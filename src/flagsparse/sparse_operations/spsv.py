"""Sparse triangular solve (SpSV) CSR/COO."""

from ._common import *

import time
import triton
import triton.language as tl

SUPPORTED_SPSV_VALUE_DTYPES = (
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
SUPPORTED_SPSV_INDEX_DTYPES = (torch.int32, torch.int64)
SPSV_NON_TRANS_PRIMARY_COMBOS = (
    (torch.float32, torch.int32),
    (torch.float64, torch.int32),
)

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


def _validate_spsv_non_trans_combo(data_dtype, index_dtype, fmt_name):
    """Validate NON_TRANS support matrix and keep error messages explicit."""
    if (data_dtype, index_dtype) in SPSV_NON_TRANS_PRIMARY_COMBOS:
        return
    if data_dtype == torch.bfloat16 and index_dtype == torch.int32:
        return
    raise TypeError(
        f"{fmt_name} SpSV currently supports NON_TRANS combinations with int32 kernel "
        "indices: (float32, int32), (float64, int32), (bfloat16, int32)"
    )


def _prepare_spsv_inputs(data, indices, indptr, b, shape, transpose=False):
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

    if data.dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: bfloat16, float32, float64"
        )
    if indices.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")
    if transpose:
        raise NotImplementedError("transpose=True is not implemented in Triton SpSV yet")

    indices64 = indices.to(torch.int64).contiguous()
    indptr64 = indptr.to(torch.int64).contiguous()
    if indices64.numel() > 0 and int(indices64.max().item()) > _INDEX_LIMIT_INT32:
        raise ValueError(
            f"int64 index value {int(indices64.max().item())} exceeds Triton int32 kernel range"
        )
    _validate_spsv_non_trans_combo(data.dtype, torch.int32, "CSR")

    if indptr64.numel() > 0:
        if int(indptr64[0].item()) != 0:
            raise ValueError("indptr[0] must be 0")
        if int(indptr64[-1].item()) != data.numel():
            raise ValueError("indptr[-1] must equal nnz")
        if bool(torch.any(indptr64[1:] < indptr64[:-1]).item()):
            raise ValueError("indptr must be non-decreasing")
    if indices64.numel() > 0:
        if bool(torch.any(indices64 < 0).item()):
            raise IndexError("indices must be non-negative")
        max_idx = int(indices64.max().item())
        if max_idx >= n_cols:
            raise IndexError(f"indices out of range for n_cols={n_cols}")

    return (
        data.contiguous(),
        indices64,
        indptr64,
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


@triton.jit
def _spsv_coo_level_kernel_real(
    data_ptr,
    row_ptr_ptr,
    col_ptr,
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
    start = tl.load(row_ptr_ptr + row)
    end = tl.load(row_ptr_ptr + row + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    diag = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    if UNIT_DIAG:
        diag = diag + 1.0

    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        col = tl.load(col_ptr + offsets, mask=mask, other=0)
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


def _auto_spsv_launch_config(indptr, block_nnz=None, max_segments=None):
    if indptr.numel() <= 1:
        max_nnz_per_row = 0
    else:
        row_lengths = indptr[1:] - indptr[:-1]
        max_nnz_per_row = int(row_lengths.max().item())

    auto_block = block_nnz is None
    if block_nnz is None:
        if max_nnz_per_row <= 64:
            block_nnz_use = 64
        elif max_nnz_per_row <= 256:
            block_nnz_use = 128
        elif max_nnz_per_row <= 1024:
            block_nnz_use = 256
        elif max_nnz_per_row <= 4096:
            block_nnz_use = 512
        elif max_nnz_per_row <= 16384:
            block_nnz_use = 1024
        else:
            block_nnz_use = 2048
    else:
        block_nnz_use = int(block_nnz)
        if block_nnz_use <= 0:
            raise ValueError("block_nnz must be a positive integer")

    required_segments = max(
        (max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1
    )
    if max_segments is None:
        max_segments_use = required_segments
        if auto_block:
            while max_segments_use > 2048 and block_nnz_use < 65536:
                block_nnz_use *= 2
                max_segments_use = max(
                    (max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1
                )
    else:
        max_segments_use = int(max_segments)
        if max_segments_use <= 0:
            raise ValueError("max_segments must be a positive integer")
        if max_segments_use < required_segments:
            raise ValueError(
                f"max_segments={max_segments_use} is too small; at least {required_segments} required"
            )

    return block_nnz_use, max_segments_use


def _triton_spsv_csr_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    levels = _build_spsv_levels(indptr, indices, n_rows, lower=lower)
    block_nnz_use, max_segments_use = _auto_spsv_launch_config(
        indptr, block_nnz=block_nnz, max_segments=max_segments
    )

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


def _prepare_spsv_coo_inputs(data, row, col, b, shape, transpose=False):
    if not all(torch.is_tensor(t) for t in (data, row, col, b)):
        raise TypeError("data, row, col, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, b)):
        raise ValueError("data, row, col, b must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if row.numel() != data.numel() or col.numel() != data.numel():
        raise ValueError("data, row, col must have the same length")
    if b.ndim not in (1, 2):
        raise ValueError("b must be 1D or 2D (vector or multiple RHS)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if b.ndim == 1 and b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")
    if b.ndim == 2 and b.shape[0] != n_rows:
        raise ValueError(f"b.shape[0] must equal n_rows={n_rows}")

    if data.dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError("data dtype must be one of: bfloat16, float32, float64")
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")
    if row.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")
    if transpose:
        raise NotImplementedError("transpose=True is not implemented in Triton SpSV yet")

    row64 = row.to(torch.int64).contiguous()
    col64 = col.to(torch.int64).contiguous()
    if col64.numel() > 0 and int(col64.max().item()) > _INDEX_LIMIT_INT32:
        raise ValueError(
            f"int64 index value {int(col64.max().item())} exceeds Triton int32 kernel range"
        )
    _validate_spsv_non_trans_combo(data.dtype, torch.int32, "COO")
    if row64.numel() > 0:
        if bool(torch.any(row64 < 0).item()):
            raise IndexError("row indices must be non-negative")
        if bool(torch.any(col64 < 0).item()):
            raise IndexError("col indices must be non-negative")
        max_row = int(row64.max().item())
        max_col = int(col64.max().item())
        if max_row >= n_rows:
            raise IndexError(f"row indices out of range for n_rows={n_rows}")
        if max_col >= n_cols:
            raise IndexError(f"col indices out of range for n_cols={n_cols}")

    return (
        data.contiguous(),
        row64,
        col64,
        b.contiguous(),
        n_rows,
        n_cols,
    )


def _coo_is_sorted_unique(row64, col64, n_cols):
    nnz = row64.numel()
    if nnz <= 1:
        return True
    key = row64 * max(1, n_cols) + col64
    is_sorted = bool(torch.all(key[1:] >= key[:-1]).item())
    is_unique = bool(torch.all(key[1:] != key[:-1]).item())
    return is_sorted and is_unique


def _build_coo_row_ptr(row_sorted, n_rows):
    row_ptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=row_sorted.device)
    if row_sorted.numel() > 0:
        nnz_per_row = torch.bincount(row_sorted, minlength=n_rows)
        row_ptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return row_ptr


def _coo_to_csr_sorted_unique(data, row64, col64, n_rows, n_cols):
    nnz = data.numel()
    if nnz == 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        indices = torch.empty(0, dtype=torch.int64, device=data.device)
        return data, indices, indptr

    key = row64 * max(1, n_cols) + col64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    key_s = key[order]
    data_s = data[order]

    unique_key, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    out_nnz = unique_key.numel()

    if data_s.dtype == torch.bfloat16:
        reduced_f32 = torch.zeros(out_nnz, dtype=torch.float32, device=data.device)
        reduced_f32.scatter_add_(0, inverse, data_s.to(torch.float32))
        data_u = reduced_f32.to(torch.bfloat16)
    else:
        data_u = torch.zeros(out_nnz, dtype=data.dtype, device=data.device)
        data_u.scatter_add_(0, inverse, data_s)

    row_u = torch.div(unique_key, max(1, n_cols), rounding_mode="floor")
    col_u = unique_key - row_u * max(1, n_cols)
    indptr = _build_coo_row_ptr(row_u, n_rows)
    indices = col_u.to(torch.int64)
    return data_u, indices, indptr


def _triton_spsv_coo_vector(
    data,
    cols,
    row_ptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    levels = _build_spsv_levels(row_ptr, cols, n_rows, lower=lower)
    block_nnz_use, max_segments_use = _auto_spsv_launch_config(
        row_ptr, block_nnz=block_nnz, max_segments=max_segments
    )

    for rows_lv in levels:
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        grid = (n_lv,)
        _spsv_coo_level_kernel_real[grid](
            data,
            row_ptr,
            cols,
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
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
):
    """Sparse triangular solve using Triton level-scheduling kernels.

    Primary NON_TRANS support matrix:
    - float32 + int32 indices
    - float64 + int32 indices
    """
    data, indices, indptr, b, n_rows, n_cols = _prepare_spsv_inputs(
        data, indices, indptr, b, shape, transpose=transpose
    )
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")
    kernel_indices = indices.to(torch.int32) if indices.dtype != torch.int32 else indices
    kernel_indptr = indptr
    compute_dtype = data.dtype
    data_in = data
    b_in = b
    if data.dtype == torch.bfloat16:
        compute_dtype = torch.float32
        data_in = data.to(torch.float32)
        b_in = b.to(torch.float32)
    elif data.dtype == torch.float32:
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
            block_nnz=block_nnz,
            max_segments=max_segments,
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
                    b_in[:, j].contiguous(),
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    block_nnz=block_nnz,
                    max_segments=max_segments,
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
    coo_mode="auto",
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
):
    """COO SpSV with dual mode:
    - direct: use COO level kernel directly (requires sorted+unique COO)
    - csr: convert COO -> CSR (sorted+deduplicated) then call flagsparse_spsv_csr
    - auto: pick direct when sorted+unique, otherwise csr

    Primary NON_TRANS support matrix:
    - float32 + int32 indices
    - float64 + int32 indices
    """
    data, row64, col64, b, n_rows, n_cols = _prepare_spsv_coo_inputs(
        data, row, col, b, shape, transpose=transpose
    )
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")

    mode = str(coo_mode).lower()
    if mode not in ("auto", "direct", "csr"):
        raise ValueError("coo_mode must be one of: 'auto', 'direct', 'csr'")

    sorted_unique = _coo_is_sorted_unique(row64, col64, n_cols)
    use_direct = mode == "direct" or (mode == "auto" and sorted_unique)
    if mode == "direct" and not sorted_unique:
        raise ValueError(
            "coo_mode='direct' requires COO sorted by (row, col) with no duplicate coordinates; "
            "use coo_mode='csr' or 'auto' for unsorted/duplicate COO input"
        )

    if not use_direct:
        data_csr, indices_csr, indptr_csr = _coo_to_csr_sorted_unique(
            data, row64, col64, n_rows, n_cols
        )
        return flagsparse_spsv_csr(
            data_csr,
            indices_csr,
            indptr_csr,
            b,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            transpose=transpose,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
        )

    kernel_cols = col64.to(torch.int32)
    row_ptr = _build_coo_row_ptr(row64, n_rows)

    compute_dtype = data.dtype
    data_in = data
    b_in = b
    if data.dtype == torch.bfloat16:
        compute_dtype = torch.float32
        data_in = data.to(torch.float32)
        b_in = b.to(torch.float32)
    elif data.dtype == torch.float32:
        compute_dtype = torch.float64
        data_in = data.to(torch.float64)
        b_in = b.to(torch.float64)
    diag_eps = 1e-12 if compute_dtype == torch.float64 else 1e-6

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if b_in.ndim == 1:
        x = _triton_spsv_coo_vector(
            data_in,
            kernel_cols,
            row_ptr,
            b_in,
            n_rows,
            lower=lower,
            unit_diagonal=unit_diagonal,
            block_nnz=block_nnz,
            max_segments=max_segments,
            diag_eps=diag_eps,
        )
    else:
        cols_out = []
        for j in range(b_in.shape[1]):
            cols_out.append(
                _triton_spsv_coo_vector(
                    data_in,
                    kernel_cols,
                    row_ptr,
                    b_in[:, j].contiguous(),
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    block_nnz=block_nnz,
                    max_segments=max_segments,
                    diag_eps=diag_eps,
                )
            )
        x = torch.stack(cols_out, dim=1)
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