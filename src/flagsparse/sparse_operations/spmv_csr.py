"""CSR SpMV: Triton baseline kernels + optimised CSR-Vector buckets."""

from ._common import *

import time
import triton
import triton.language as tl

class PreparedCsrSpmv:
    """Cached CSR metadata for repeated SpMV calls on the same sparse matrix."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "block_nnz",
        "max_segments",
        "max_row_nnz",
        "opt_buckets",
        "supports_opt",
        "_baseline_compute_dtype",
        "_baseline_data",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        n_rows,
        n_cols,
        block_nnz,
        max_segments,
        max_row_nnz,
        opt_buckets,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.block_nnz = block_nnz
        self.max_segments = max_segments
        self.max_row_nnz = max_row_nnz
        self.opt_buckets = opt_buckets
        self.supports_opt = data.dtype in (torch.float32, torch.float64)
        if data.dtype in (torch.float16, torch.bfloat16):
            self._baseline_compute_dtype = torch.float32
        elif data.dtype == torch.float32:
            self._baseline_compute_dtype = torch.float64
        else:
            self._baseline_compute_dtype = data.dtype
        self._baseline_data = None


# Performance-first CSR-Vector buckets.  num_warps*32 >= block_size.
# First bucket uses batch_rows>1: one program processes several short rows
# (fewer blocks → better occupancy on graphs with millions of low-degree rows).
_SPMV_OPT_BUCKET_CONFIGS = (
    {
        "max_row_nnz": 64,
        "block_size": 32,
        "num_warps": 1,
        "num_stages": 2,
        "batch_rows": 16,
    },
    {"max_row_nnz": 512, "block_size": 256, "num_warps": 8, "num_stages": 2},
    {"max_row_nnz": 4096, "block_size": 512, "num_warps": 16, "num_stages": 2},
    {"max_row_nnz": None, "block_size": 1024, "num_warps": 32, "num_stages": 3},
)
# fp64: extra row-length tiers + smaller tiles vs f32; batch_rows=4 for short-row kernel.
_SPMV_OPT_BUCKET_CONFIGS_FP64 = (
    {
        "max_row_nnz": 64,
        "block_size": 32,
        "num_warps": 1,
        "num_stages": 2,
        "batch_rows": 4,
    },
    {"max_row_nnz": 256, "block_size": 64, "num_warps": 2, "num_stages": 2},
    {"max_row_nnz": 2048, "block_size": 128, "num_warps": 4, "num_stages": 2},
    {"max_row_nnz": 8192, "block_size": 256, "num_warps": 8, "num_stages": 2},
    {"max_row_nnz": None, "block_size": 512, "num_warps": 16, "num_stages": 1},
)
_SPMV_OPT_ACC_MODES = ("fast", "mixed", "accurate")



@triton.jit
def _spmv_csr_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        x_vals = tl.load(x_ptr + col, mask=mask, other=0.0)
        part = tl.where(mask, a * x_vals, 0.0)
        acc = acc + tl.sum(part)
    tl.store(y_ptr + row, acc)


@triton.jit
def _spmv_csr_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc_re = tl.load(data_ri_ptr + start * 2, mask=start < end, other=0.0) * 0
    acc_im = tl.load(data_ri_ptr + start * 2 + 1, mask=start < end, other=0.0) * 0
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
        a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        x_re = tl.load(x_ri_ptr + col * 2, mask=mask, other=0.0)
        x_im = tl.load(x_ri_ptr + col * 2 + 1, mask=mask, other=0.0)
        prod_re = tl.where(mask, a_re * x_re - a_im * x_im, 0.0)
        prod_im = tl.where(mask, a_re * x_im + a_im * x_re, 0.0)
        acc_re = acc_re + tl.sum(prod_re)
        acc_im = acc_im + tl.sum(prod_im)
    tl.store(y_ri_ptr + row * 2, acc_re)
    tl.store(y_ri_ptr + row * 2 + 1, acc_im)


# ── Optimised SpMV (CSR-Vector, perf-oriented, no CuPy) ─────────────
# fp32 / fp64 native lane accum.  Batched kernel for many short rows per program.

@triton.jit
def _spmv_csr_batched_short_f32(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    rows_ptr,
    n_bucket_rows,
    BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEGS: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    for b in range(BATCH):
        ridx = pid * BATCH + b
        active = ridx < n_bucket_rows
        row = tl.load(rows_ptr + ridx, mask=active, other=0)
        start = tl.load(indptr_ptr + row, mask=active, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=active, other=0)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for seg in range(MAX_SEGS):
            offs = start + seg * BLOCK_SIZE + lane
            mask = offs < end
            a = tl.load(data_ptr + offs, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offs, mask=mask, other=0)
            xv = tl.load(x_ptr + col, mask=mask, other=0.0)
            acc += tl.where(mask, a * xv, 0.0)
        tl.store(y_ptr + row, tl.sum(acc), mask=active)

@triton.jit
def _spmv_csr_batched_short_f64(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    rows_ptr,
    n_bucket_rows,
    BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEGS: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    for b in range(BATCH):
        ridx = pid * BATCH + b
        active = ridx < n_bucket_rows
        row = tl.load(rows_ptr + ridx, mask=active, other=0)
        start = tl.load(indptr_ptr + row, mask=active, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=active, other=0)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        for seg in range(MAX_SEGS):
            offs = start + seg * BLOCK_SIZE + lane
            mask = offs < end
            a = tl.load(data_ptr + offs, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offs, mask=mask, other=0)
            xv = tl.load(x_ptr + col, mask=mask, other=0.0)
            acc += tl.where(mask, a * xv, 0.0)
        tl.store(y_ptr + row, tl.sum(acc), mask=active)

@triton.jit
def _spmv_csr_vector_rows_f32(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    rows_ptr,
    n_bucket_rows,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEGS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lane = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for seg in range(MAX_SEGS):
        offs = start + seg * BLOCK_SIZE + lane
        mask = offs < end
        a = tl.load(data_ptr + offs, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offs, mask=mask, other=0)
        xv = tl.load(x_ptr + col, mask=mask, other=0.0)
        acc = tl.where(mask, acc + a * xv, acc)
    tl.store(y_ptr + row, tl.sum(acc))

@triton.jit
def _spmv_csr_vector_rows_f64(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    rows_ptr,
    n_bucket_rows,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEGS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lane = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    for seg in range(MAX_SEGS):
        offs = start + seg * BLOCK_SIZE + lane
        mask = offs < end
        a = tl.load(data_ptr + offs, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offs, mask=mask, other=0)
        xv = tl.load(x_ptr + col, mask=mask, other=0.0)
        acc = tl.where(mask, acc + a * xv, acc)
    tl.store(y_ptr + row, tl.sum(acc))

def _build_spmv_opt_buckets(
    row_lengths,
    max_row_nnz,
    row_index_dtype,
    max_segments=None,
    fp64=False,
):
    buckets = []
    lower_bound = 0
    configs = _SPMV_OPT_BUCKET_CONFIGS_FP64 if fp64 else _SPMV_OPT_BUCKET_CONFIGS
    for spec in configs:
        upper_bound = spec["max_row_nnz"]
        if upper_bound is None:
            mask = row_lengths > lower_bound
            bucket_max_row_nnz = max_row_nnz
        elif lower_bound == 0:
            # Include nnz==0 rows in the first bucket (they still need y[i]=0).
            mask = row_lengths <= upper_bound
            bucket_max_row_nnz = upper_bound
        else:
            mask = (row_lengths > lower_bound) & (row_lengths <= upper_bound)
            bucket_max_row_nnz = upper_bound
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        if rows.numel() == 0:
            if upper_bound is not None:
                lower_bound = upper_bound
            continue
        if max_segments is None:
            max_segs = max(
                (bucket_max_row_nnz + spec["block_size"] - 1) // spec["block_size"],
                1,
            )
        else:
            max_segs = max_segments
        buckets.append(
            {
                "rows": rows.to(row_index_dtype),
                "block_size": spec["block_size"],
                "max_segs": max_segs,
                "num_warps": spec["num_warps"],
                "num_stages": spec["num_stages"],
                "batch_rows": int(spec.get("batch_rows", 1)),
            }
        )
        if upper_bound is not None:
            lower_bound = upper_bound
    return buckets

def _triton_spmv_csr_impl_opt_prepared(prepared, x):
    # First bucket includes nnz==0 rows; every row gets exactly one store.
    dtype = prepared.data.dtype
    y = torch.empty(prepared.n_rows, dtype=dtype, device=prepared.data.device)
    if prepared.n_rows == 0:
        return y
    vec_f32 = _spmv_csr_vector_rows_f32
    vec_f64 = _spmv_csr_vector_rows_f64
    bat_f32 = _spmv_csr_batched_short_f32
    bat_f64 = _spmv_csr_batched_short_f64
    for bucket in prepared.opt_buckets:
        rows = bucket["rows"]
        br = max(1, int(bucket.get("batch_rows", 1)))
        n_r = rows.numel()
        if br > 1:
            kernel = bat_f64 if dtype == torch.float64 else bat_f32
            grid = (triton.cdiv(n_r, br),)
            kernel[grid](
                prepared.data,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                x,
                y,
                rows,
                n_bucket_rows=n_r,
                BATCH=br,
                BLOCK_SIZE=bucket["block_size"],
                MAX_SEGS=bucket["max_segs"],
                num_warps=bucket["num_warps"],
                num_stages=bucket["num_stages"],
            )
        else:
            kernel = vec_f64 if dtype == torch.float64 else vec_f32
            grid = (n_r,)
            kernel[grid](
                prepared.data,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                x,
                y,
                rows,
                n_bucket_rows=n_r,
                BLOCK_SIZE=bucket["block_size"],
                MAX_SEGS=bucket["max_segs"],
                num_warps=bucket["num_warps"],
                num_stages=bucket["num_stages"],
            )
    return y


def _prepare_spmv_csr_matrix(data, indices, indptr, shape):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr)):
        raise TypeError("data, indices, indptr must all be torch.Tensor")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if not all(t.is_cuda for t in (data, indices, indptr)):
        raise ValueError("data, indices, indptr must be CUDA tensors")
    if data.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")

    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()

    if indptr.numel() > 0:
        if int(indptr[0].item()) != 0:
            raise ValueError("indptr must start at zero")
        if int(indptr[-1].item()) != data.numel():
            raise ValueError("indptr[-1] must equal nnz")
        if indptr.numel() > 1 and torch.any(indptr[1:] < indptr[:-1]).item():
            raise ValueError("indptr must be non-decreasing")

    nnz = data.numel()
    if nnz > 0:
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= n_cols:
            raise IndexError("indices out of range for n_cols")
        if max_index > _INDEX_LIMIT_INT32:
            raise ValueError(
                f"int64 column index {max_index} exceeds Triton int32 kernel range"
            )
    kernel_indices = indices.to(torch.int32) if indices.dtype == torch.int64 else indices
    kernel_indptr = (
        indptr.to(torch.int32) if nnz <= _INDEX_LIMIT_INT32 else indptr.to(torch.int64)
    )
    row_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
    max_row_nnz = int(row_lengths.max().item()) if n_rows > 0 else 0
    return (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    )


def _validate_spmv_x(x, prepared):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    if x.numel() != prepared.n_cols:
        raise ValueError(f"x length must be n_cols={prepared.n_cols}, got {x.numel()}")
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    return x.contiguous()


def prepare_spmv_csr(data, indices, indptr, shape, block_nnz=256, max_segments=None):
    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    ) = _prepare_spmv_csr_matrix(data, indices, indptr, shape)
    block_nnz_use = block_nnz
    if max_segments is None:
        max_segments_use = max((max_row_nnz + block_nnz_use - 1) // block_nnz_use, 1)
        while max_segments_use > 2048 and block_nnz_use < 65536:
            block_nnz_use *= 2
            max_segments_use = max(
                (max_row_nnz + block_nnz_use - 1) // block_nnz_use,
                1,
            )
    else:
        max_segments_use = max_segments
    row_index_dtype = torch.int32 if n_rows <= _INDEX_LIMIT_INT32 else torch.int64
    opt_buckets = _build_spmv_opt_buckets(
        row_lengths,
        max_row_nnz=max_row_nnz,
        row_index_dtype=row_index_dtype,
        max_segments=max_segments,
        fp64=data.dtype == torch.float64,
    )
    return PreparedCsrSpmv(
        data=data,
        kernel_indices=kernel_indices,
        kernel_indptr=kernel_indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        block_nnz=block_nnz_use,
        max_segments=max_segments_use,
        max_row_nnz=max_row_nnz,
        opt_buckets=opt_buckets,
    )


def _get_spmv_baseline_data(prepared):
    compute_dtype = prepared._baseline_compute_dtype
    if compute_dtype == prepared.data.dtype:
        return compute_dtype, prepared.data
    if (
        prepared._baseline_data is None
        or prepared._baseline_data.dtype != compute_dtype
    ):
        prepared._baseline_data = prepared.data.to(compute_dtype)
    return compute_dtype, prepared._baseline_data


def _triton_spmv_csr_impl_prepared(prepared, x):
    device = prepared.data.device
    dtype = prepared.data.dtype
    y = torch.empty(prepared.n_rows, dtype=dtype, device=device)
    if prepared.n_rows == 0:
        return y
    compute_dtype, data_in = _get_spmv_baseline_data(prepared)
    x_in = x
    if compute_dtype != dtype:
        x_in = x.to(compute_dtype)
    if not _is_complex_dtype(compute_dtype):
        y_out = torch.empty(prepared.n_rows, dtype=compute_dtype, device=device)
        grid = (prepared.n_rows,)
        _spmv_csr_real_kernel[grid](
            data_in,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x_in,
            y_out,
            n_rows=prepared.n_rows,
            BLOCK_NNZ=prepared.block_nnz,
            MAX_SEGMENTS=prepared.max_segments,
        )
        if dtype != compute_dtype:
            y_out = y_out.to(dtype)
        y.copy_(y_out)
        return y
    data_ri = torch.view_as_real(data_in).reshape(-1)
    x_ri = torch.view_as_real(x_in).reshape(-1)
    comp_dtype = data_ri.dtype
    y_ri = torch.empty(prepared.n_rows * 2, dtype=comp_dtype, device=device)
    grid = (prepared.n_rows,)
    _spmv_csr_complex_kernel[grid](
        data_ri,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        x_ri,
        y_ri,
        n_rows=prepared.n_rows,
        BLOCK_NNZ=prepared.block_nnz,
        MAX_SEGMENTS=prepared.max_segments,
    )
    y_ri = y_ri.reshape(prepared.n_rows, 2)
    y.copy_(torch.view_as_complex(y_ri))
    return y


def flagsparse_spmv_csr(
    data=None,
    indices=None,
    indptr=None,
    x=None,
    shape=None,
    block_nnz=256,
    max_segments=None,
    out=None,
    return_time=False,
    use_opt=False,
    prepared=None,
):
    """
    CSR SpMV: y = A @ x using Triton (cuSPARSE-aligned dtypes).
    data, indices, indptr: CSR arrays; x: dense vector; shape: (n_rows, n_cols).
    prepared: cached CSR metadata from prepare_spmv_csr for steady-state runs.
    max_segments: None = auto-compute from indptr so all NNZ per row are covered.
    use_opt: if True, use the faster CSR-Vector bucketed path (fp32/fp64 native accum).
    """
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmv_csr(
            data,
            indices,
            indptr,
            shape,
            block_nnz=block_nnz,
            max_segments=max_segments,
        )
    x = _validate_spmv_x(x, prepared)
    t0 = None
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    if use_opt and prepared.supports_opt:
        y = _triton_spmv_csr_impl_opt_prepared(prepared, x)
    else:
        y = _triton_spmv_csr_impl_prepared(prepared, x)
    elapsed_ms = None
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != y.shape or out.dtype != y.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(y)
        y = out
    if return_time:
        return y, elapsed_ms
    return y


def _coo_is_sorted_lex(row_i64, col_i64, n_cols):
    """True iff COO rows are non-decreasing lex order (row, col)."""
    n = row_i64.numel()
    if n <= 1:
        return True
    scale = max(1, int(n_cols))
    key = row_i64 * scale + col_i64
    return bool((key[1:] >= key[:-1]).all().item())


def coo_to_csr_for_spmv(data, row, col, shape, assume_sorted=False):
    """Convert COO to CSR triple (data, csr_col_indices, indptr) for SpMV."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    if row64.numel() == 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        return data, col64.to(torch.int32), indptr

    if assume_sorted or _coo_is_sorted_lex(row64, col64, n_cols):
        row_s, col_s, data_s = row64, col64, data
    else:
        key = row64 * max(1, n_cols) + col64
        order = torch.argsort(key)
        row_s = row64[order]
        col_s = col64[order]
        data_s = data[order].to(data.dtype)

    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    nnz = data_s.numel()
    if nnz > 0:
        nnz_per_row = torch.bincount(row_s, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = col_s.to(torch.int32)
    return data_s, indices, indptr


def prepare_spmv_coo_tocsr(
    data,
    row,
    col,
    shape,
    block_nnz=256,
    max_segments=None,
    assume_sorted=False,
):
    """One-time COO → CSR + bucket metadata; use with ``flagsparse_spmv_coo_tocsr(..., prepared=p)``."""
    if not all(torch.is_tensor(t) for t in (data, row, col)):
        raise TypeError("data, row, col must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col)):
        raise ValueError("data, row, col must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must all be 1D tensors")
    if data.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if row.numel() != col.numel() or data.numel() != row.numel():
        raise ValueError("data, row, col must have the same length")

    data_s, indices, indptr = coo_to_csr_for_spmv(
        data, row, col, shape, assume_sorted=assume_sorted
    )
    return prepare_spmv_csr(
        data_s,
        indices,
        indptr,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )


def flagsparse_spmv_coo_tocsr(
    data=None,
    row=None,
    col=None,
    x=None,
    shape=None,
    block_nnz=256,
    max_segments=None,
    out=None,
    return_time=False,
    use_opt=True,
    prepared=None,
    assume_sorted=False,
):
    """COO SpMV via CSR conversion: y = A @ x.

    Default ``use_opt=True`` enables the fast CSR-Vector path for float32/float64.
    If COO is already lex-sorted by (row, col), pass ``assume_sorted=True`` to skip ``argsort``.

    Steady-state: ``p = prepare_spmv_coo_tocsr(data, row, col, shape)`` then call with ``prepared=p``
    (``data``/``row``/``col`` may be omitted).
    """
    if prepared is not None:
        if x is None:
            raise TypeError("x is required")
        if shape is None:
            shape = prepared.shape
        sh = (int(shape[0]), int(shape[1]))
        if sh != prepared.shape:
            raise ValueError(f"shape {sh} does not match prepared.shape {prepared.shape}")
        return flagsparse_spmv_csr(
            x=x,
            shape=shape,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
            use_opt=use_opt,
            prepared=prepared,
        )

    if not all(torch.is_tensor(t) for t in (data, row, col, x)):
        raise TypeError("data, row, col, x must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, x)):
        raise ValueError("data, row, col, x must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1 or x.ndim != 1:
        raise ValueError("data, row, col, x must all be 1D tensors")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if x.dtype != data.dtype:
        raise TypeError("x dtype must match data dtype")

    data_s, indices, indptr = coo_to_csr_for_spmv(
        data, row, col, shape, assume_sorted=assume_sorted
    )

    return flagsparse_spmv_csr(
        data_s,
        indices,
        indptr,
        x,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        use_opt=use_opt,
    )
