"""Sparse triangular solve (SpSV) CSR/COO."""

from ._common import *

import time
import triton
import triton.language as tl

def _csr_to_dense(data, indices, indptr, shape):
    """Convert CSR (torch CUDA tensors) to dense matrix on the same device."""
    device = data.device
    dtype = data.dtype
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows == 0 or n_cols == 0:
        return torch.zeros((n_rows, n_cols), dtype=dtype, device=device)
    row_ind = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    col_ind = indices.to(torch.int64)
    coo = torch.sparse_coo_tensor(
        torch.stack([row_ind, col_ind]),
        data,
        (n_rows, n_cols),
        device=device,
    ).coalesce()
    return coo.to_dense()


def _prepare_spsv_inputs(data, indices, indptr, b, shape):
    """Validate and normalize inputs for sparse solve A x = b with CSR A."""
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, b)):
        raise TypeError("data, indices, indptr, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, b)):
        raise ValueError("data, indices, indptr, b must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if b.ndim not in (1, 2):
        raise ValueError("b must be 1D or 2D (vector or multiple RHS)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if b.ndim == 1 and b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")
    if b.ndim == 2 and b.shape[0] != n_rows:
        raise ValueError(f"b.shape[0] must equal n_rows={n_rows}")

    if data.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")

    return (
        data.contiguous(),
        indices.contiguous(),
        indptr.contiguous(),
        b.contiguous(),
        n_rows,
        n_cols,
    )


@triton.jit
def _spsv_csr_level_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_level_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    diag = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    if UNIT_DIAG:
        diag = diag + 1.0

    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        x_vals = tl.load(x_ptr + col, mask=mask, other=0.0)

        if LOWER:
            solved = col < row
        else:
            solved = col > row
        is_diag = col == row

        acc = acc + tl.sum(tl.where(mask & solved, a * x_vals, 0.0))
        if not UNIT_DIAG:
            diag = diag + tl.sum(tl.where(mask & is_diag, a, 0.0))

    rhs = tl.load(b_ptr + row)
    diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
    x_row = (rhs - acc) / diag_safe
    # Prevent NaN propagation in ill-conditioned rows.
    x_row = tl.where(x_row == x_row, x_row, 0.0)
    tl.store(x_ptr + row, x_row)


def _build_spsv_levels(indptr, indices, n_rows, lower=True):
    """Build dependency levels for triangular solve so each level can run in parallel."""
    if n_rows == 0:
        return []
    indptr_h = indptr.to(torch.int64).cpu()
    indices_h = indices.to(torch.int64).cpu()
    levels = [0] * n_rows
    if lower:
        for i in range(n_rows):
            s = int(indptr_h[i].item())
            e = int(indptr_h[i + 1].item())
            lvl = 0
            for p in range(s, e):
                c = int(indices_h[p].item())
                if c < i:
                    lvl = max(lvl, levels[c] + 1)
            levels[i] = lvl
    else:
        for i in range(n_rows - 1, -1, -1):
            s = int(indptr_h[i].item())
            e = int(indptr_h[i + 1].item())
            lvl = 0
            for p in range(s, e):
                c = int(indices_h[p].item())
                if c > i:
                    lvl = max(lvl, levels[c] + 1)
            levels[i] = lvl

    max_level = max(levels)
    buckets = [[] for _ in range(max_level + 1)]
    for r, lv in enumerate(levels):
        buckets[lv].append(r)

    device = indptr.device
    return [
        torch.tensor(rows, dtype=torch.int32, device=device)
        for rows in buckets
        if rows
    ]


def _triton_spsv_csr_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    block_nnz=256,
    diag_eps=1e-12,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    levels = _build_spsv_levels(indptr, indices, n_rows, lower=lower)
    row_lengths = indptr[1:] - indptr[:-1]
    max_nnz_per_row = int(row_lengths.max().item()) if n_rows > 0 else 0
    block_nnz_use = block_nnz
    max_segments_use = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    while max_segments_use > 2048 and block_nnz_use < 65536:
        block_nnz_use *= 2
        max_segments_use = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)

    for rows_lv in levels:
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        grid = (n_lv,)
        _spsv_csr_level_kernel[grid](
            data,
            indices,
            indptr,
            b_vec,
            x,
            rows_lv,
            n_level_rows=n_lv,
            BLOCK_NNZ=block_nnz_use,
            MAX_SEGMENTS=max_segments_use,
            LOWER=lower,
            UNIT_DIAG=unit_diagonal,
            DIAG_EPS=diag_eps,
        )
    return x


def flagsparse_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    out=None,
    return_time=False,
):
    """Sparse triangular solve using Triton level-scheduling kernels."""
    data, indices, indptr, b, n_rows, n_cols = _prepare_spsv_inputs(
        data, indices, indptr, b, shape
    )
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")
    if transpose:
        raise NotImplementedError("transpose=True is not implemented in Triton SpSV yet")
    if data.dtype not in (torch.float32, torch.float64):
        raise TypeError("Triton SpSV currently supports float32/float64")
    kernel_indices = indices.to(torch.int32) if indices.dtype != torch.int32 else indices
    kernel_indptr = indptr.to(torch.int64)
    compute_dtype = data.dtype
    data_in = data
    b_in = b
    if data.dtype == torch.float32:
        # Improve numerical stability on hard matrices.
        compute_dtype = torch.float64
        data_in = data.to(torch.float64)
        b_in = b.to(torch.float64)
    diag_eps = 1e-12 if compute_dtype == torch.float64 else 1e-6
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if b_in.ndim == 1:
        x = _triton_spsv_csr_vector(
            data_in,
            kernel_indices,
            kernel_indptr,
            b_in,
            n_rows,
            lower=lower,
            unit_diagonal=unit_diagonal,
            diag_eps=diag_eps,
        )
    else:
        cols = []
        for j in range(b_in.shape[1]):
            cols.append(
                _triton_spsv_csr_vector(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    b_in[:, j],
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    diag_eps=diag_eps,
                )
            )
        x = torch.stack(cols, dim=1)
    if compute_dtype != data.dtype:
        x = x.to(data.dtype)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out

    if return_time:
        return x, elapsed_ms
    return x


def flagsparse_spsv_coo(
    data,
    row,
    col,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    **kwargs,
):
    """COO variant: convert COO (data, row, col) to CSR then call flagsparse_spsv_csr."""
    if not all(torch.is_tensor(t) for t in (data, row, col, b)):
        raise TypeError("data, row, col, b must all be torch.Tensor")
    device = data.device
    dtype = data.dtype
    n_rows, n_cols = int(shape[0]), int(shape[1])

    # Ensure sorted by row, then col (PyTorch has no lexsort API).
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    key = row64 * max(1, n_cols) + col64
    order = torch.argsort(key)
    row_s = row64[order]
    col_s = col64[order]
    data_s = data[order].to(dtype)

    nnz = data_s.numel()
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    if nnz > 0:
        nnz_per_row = torch.bincount(row_s, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = col_s.to(torch.int64)

    return flagsparse_spsv_csr(
        data_s,
        indices,
        indptr,
        b,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        **kwargs,
    )

