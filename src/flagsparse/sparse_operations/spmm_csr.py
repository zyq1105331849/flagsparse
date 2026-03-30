"""CSR SpMM kernels, helpers, and benchmark entry points."""

from ._common import *

SUPPORTED_SPMM_VALUE_DTYPES = SUPPORTED_VALUE_DTYPES
def _spmm_relative_threshold(value_dtype):
    if value_dtype == torch.float16:
        return 5e-3
    if value_dtype == torch.bfloat16:
        return 1e-2
    if value_dtype in (torch.float32, torch.complex64):
        return 1e-6
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-12
    return 1e-6


def _spmm_coo_reference_tolerance(value_dtype):
    if value_dtype == torch.float16:
        return 2e-3, 2e-3
    if value_dtype == torch.bfloat16:
        return 1e-1, 1e-1
    if value_dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-2
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-12, 1e-10
    return 1e-6, 1e-5


def _spmm_error_metrics(candidate, reference):
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate and reference must have the same shape, got {candidate.shape} vs {reference.shape}"
        )

    if candidate.numel() == 0:
        return {
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "sum_relative_error": 0.0,
            "reference_max_magnitude": 0.0,
            "reference_sum_magnitude": 0.0,
        }

    if _is_complex_dtype(reference.dtype):
        candidate_compare = torch.abs(candidate)
        reference_compare = torch.abs(reference)
        abs_diff = torch.abs(candidate_compare - reference_compare)
    else:
        reference_compare = torch.abs(reference)
        abs_diff = torch.abs(candidate - reference)

    max_abs_error = float(torch.max(abs_diff).item())
    reference_max_magnitude = float(torch.max(reference_compare).item())
    sum_abs_error = float(torch.sum(abs_diff).item())
    reference_sum_magnitude = float(torch.sum(reference_compare).item())

    if reference_max_magnitude == 0.0:
        max_relative_error = 0.0 if max_abs_error == 0.0 else float("inf")
    else:
        max_relative_error = max_abs_error / reference_max_magnitude

    if reference_sum_magnitude == 0.0:
        sum_relative_error = 0.0 if sum_abs_error == 0.0 else float("inf")
    else:
        sum_relative_error = sum_abs_error / reference_sum_magnitude

    return {
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "sum_relative_error": sum_relative_error,
        "reference_max_magnitude": reference_max_magnitude,
        "reference_sum_magnitude": reference_sum_magnitude,
    }


def _spmm_validation_metrics(candidate, reference):
    metrics = _spmm_error_metrics(candidate, reference)
    threshold = _spmm_relative_threshold(reference.dtype)
    atol, rtol = _tolerance_for_dtype(reference.dtype)
    metrics.update(
        {
            "relative_threshold": threshold,
            "matches_threshold": metrics["max_relative_error"] <= threshold,
            "strict_allclose_match": torch.allclose(
                candidate, reference, atol=atol, rtol=rtol
            ),
        }
    )
    return metrics

@triton.jit
def _spmm_csr_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if row >= n_rows:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)

    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


# Complex-path variant of the same AlphaSparse CSR ALG1 mapping.
@triton.jit
def _spmm_csr_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    c_ri_ptr,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if row >= n_rows:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc_re = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc_im = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_re = tl.load(data_ri_ptr + idx * 2, mask=valid, other=0.0)
            a_im = tl.load(data_ri_ptr + idx * 2 + 1, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_re = tl.load(
                b_ri_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            b_im = tl.load(
                b_ri_ptr + a_col * stride_bk + offs_n * stride_bn + stride_br,
                mask=mask_n & valid,
                other=0.0,
            )
            acc_re = acc_re + a_re.to(ACC_DTYPE) * b_re.to(ACC_DTYPE) - a_im.to(ACC_DTYPE) * b_im.to(ACC_DTYPE)
            acc_im = acc_im + a_re.to(ACC_DTYPE) * b_im.to(ACC_DTYPE) + a_im.to(ACC_DTYPE) * b_re.to(ACC_DTYPE)

    tl.store(c_ri_ptr + row * stride_cm + offs_n * stride_cn, acc_re, mask=mask_n)
    tl.store(
        c_ri_ptr + row * stride_cm + offs_n * stride_cn + stride_cr,
        acc_im,
        mask=mask_n,
    )

def _prepare_spmm_csr_matrix(data, indices, indptr, shape):
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple: (n_rows, n_cols)")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, and indptr must be 1D tensors")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")

    if not all(t.is_cuda for t in (data, indices, indptr)):
        raise ValueError("data, indices, and indptr must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr)):
        raise ValueError("data, indices, and indptr must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMM_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")

    nnz = data.numel()
    if indptr.numel() > 0 and int(indptr[0].item()) != 0:
        raise ValueError("indptr[0] must be 0")
    if indptr.numel() > 0 and int(indptr[-1].item()) != nnz:
        raise ValueError(f"indptr[-1] must equal nnz={nnz}, got {int(indptr[-1].item())}")
    if indptr.numel() > 1 and bool(torch.any(indptr[1:] < indptr[:-1]).item()):
        raise ValueError("indptr must be nondecreasing")
    if nnz > 0:
        min_col = int(indices.min().item())
        max_col = int(indices.max().item())
        if min_col < 0 or max_col >= n_cols:
            raise IndexError("indices out of range for n_cols")
        if max_col > _INDEX_LIMIT_INT32:
            raise ValueError(
                "column indices exceed the int32 range supported by the Triton kernel"
            )

    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()

    kernel_indices = indices.to(torch.int32) if indices.dtype == torch.int64 else indices
    kernel_indptr = indptr.to(torch.int64)
    row_lengths = (
        kernel_indptr[1:] - kernel_indptr[:-1]
        if n_rows > 0
        else kernel_indptr.new_empty((0,))
    )
    max_row_nnz = int(row_lengths.max().item()) if n_rows > 0 else 0
    return data, kernel_indices, kernel_indptr, n_rows, n_cols, row_lengths, max_row_nnz


def _prepare_spmm_csr_inputs(data, indices, indptr, B, shape):
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")

    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        _row_lengths,
        _max_row_nnz,
    ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    if B.shape[0] != n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={n_cols}, got {B.shape[0]}")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != data.device:
        raise ValueError("B must be on the same CUDA device as the sparse matrix")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")

    B = B.contiguous()
    return data, kernel_indices, kernel_indptr, B, n_rows, n_cols, int(B.shape[1])


def _select_spmm_alg1_warp_and_factor(n_dense_cols):
    # Mirrors AlphaSparse CSR ALG1 row-major heuristics without exposing warp details publicly.
    if n_dense_cols > 64:
        return 32, 4
    if n_dense_cols > 32:
        return 32, 2
    if n_dense_cols > 16:
        return 32, 1
    if n_dense_cols > 8:
        return 16, 1
    if n_dense_cols > 4:
        return 8, 1
    return 4, 1


def _resolve_spmm_alg1_launch_config(
    n_dense_cols,
    max_row_nnz,
    block_n=None,
    block_nnz=None,
    max_segments=None,
):
    warp_size, factor = _select_spmm_alg1_warp_and_factor(n_dense_cols)

    if block_n is None:
        block_n = warp_size * factor
    if block_nnz is None:
        block_nnz = warp_size

    if block_n <= 0 or block_nnz <= 0:
        raise ValueError("block_n and block_nnz must be positive when provided")
    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive when provided")

    required_segments = triton.cdiv(max_row_nnz, block_nnz) if max_row_nnz > 0 else 0
    if max_segments is not None and required_segments > int(max_segments):
        raise ValueError(
            "row nnz requires more CSR segments than the explicit max_segments override allows: "
            f"required {required_segments}, provided {int(max_segments)}"
        )

    return {
        "block_n": int(block_n),
        "block_nnz": int(block_nnz),
        "max_segments": (None if max_segments is None else int(max_segments)),
        "required_segments": int(required_segments),
        "warp_size": int(warp_size),
        "factor": int(factor),
        "max_row_nnz": int(max_row_nnz),
        "auto_max_segments": max_segments is None,
    }


class PreparedCsrSpmmOpt:
    """Cached CSR metadata for repeated native SpMM-opt calls on the same sparse matrix."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "row_lengths",
        "max_row_nnz",
        "row_buckets",
        "supports_opt",
        "long_part_rows",
        "long_part_starts",
        "long_part_ends",
        "long_row_ids",
        "long_row_part_ptr",
        "long_row_fallback_only",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
        row_buckets,
        long_part_rows,
        long_part_starts,
        long_part_ends,
        long_row_ids,
        long_row_part_ptr,
        long_row_fallback_only,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.row_lengths = row_lengths
        self.max_row_nnz = int(max_row_nnz)
        self.row_buckets = row_buckets
        self.supports_opt = data.dtype in (torch.float32, torch.float64)
        self.long_part_rows = long_part_rows
        self.long_part_starts = long_part_starts
        self.long_part_ends = long_part_ends
        self.long_row_ids = long_row_ids
        self.long_row_part_ptr = long_row_part_ptr
        self.long_row_fallback_only = bool(long_row_fallback_only)


_SPMM_OPT_BUCKET_SPECS = (
    {"max_row_nnz": 32, "kind": "batched", "batch_rows": 8, "block_nnz": 32},
    {"max_row_nnz": 128, "kind": "batched", "batch_rows": 4, "block_nnz": 64},
    {"max_row_nnz": 512, "kind": "vector", "batch_rows": 1, "block_nnz": 128},
    {"max_row_nnz": 2048, "kind": "vector", "batch_rows": 1, "block_nnz": 128},
    {"max_row_nnz": None, "kind": "split", "batch_rows": 1, "block_nnz": 256},
)
_SPMM_OPT_LONG_ROW_THRESHOLD = 2048


def _select_spmm_opt_block_n(n_dense_cols):
    if n_dense_cols <= 8:
        return 8
    if n_dense_cols <= 16:
        return 16
    if n_dense_cols <= 32:
        return 32
    if n_dense_cols <= 64:
        return 64
    return 128


def _build_spmm_opt_split_metadata(kernel_indptr, long_rows, part_block_nnz):
    device = kernel_indptr.device
    row_values = []
    part_starts = []
    part_ends = []
    row_part_ptr = [0]
    long_rows_cpu = long_rows.to(torch.int64).cpu().tolist()
    indptr_cpu = kernel_indptr.to(torch.int64).cpu().tolist()
    for row in long_rows_cpu:
        start = int(indptr_cpu[row])
        end = int(indptr_cpu[row + 1])
        cursor = start
        while cursor < end:
            row_values.append(row)
            part_starts.append(cursor)
            cursor = min(cursor + int(part_block_nnz), end)
            part_ends.append(cursor)
        row_part_ptr.append(len(row_values))
    row_dtype = long_rows.dtype if long_rows.numel() > 0 else torch.int32
    return (
        torch.tensor(row_values, dtype=row_dtype, device=device),
        torch.tensor(part_starts, dtype=torch.int64, device=device),
        torch.tensor(part_ends, dtype=torch.int64, device=device),
        torch.tensor(row_part_ptr, dtype=torch.int64, device=device),
    )


def _build_spmm_opt_buckets(row_lengths):
    device = row_lengths.device
    max_row_index = int(row_lengths.numel()) - 1
    row_index_dtype = (
        torch.int32 if max_row_index <= _INDEX_LIMIT_INT32 else torch.int64
    )
    buckets = []
    lower = 0
    long_rows = torch.empty((0,), dtype=row_index_dtype, device=device)
    for spec in _SPMM_OPT_BUCKET_SPECS:
        upper = spec["max_row_nnz"]
        if upper is None:
            mask = row_lengths > lower
        elif lower == 0:
            mask = row_lengths <= upper
        else:
            mask = (row_lengths > lower) & (row_lengths <= upper)
        rows = torch.nonzero(mask, as_tuple=False).flatten().to(row_index_dtype)
        if rows.numel() == 0:
            if upper is not None:
                lower = upper
            continue
        bucket = {
            "kind": spec["kind"],
            "rows": rows,
            "batch_rows": int(spec["batch_rows"]),
            "block_nnz": int(spec["block_nnz"]),
        }
        buckets.append(bucket)
        if spec["kind"] == "split":
            long_rows = rows
        if upper is not None:
            lower = upper
    return buckets, long_rows


def prepare_spmm_csr_opt(data, indices, indptr, shape):
    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    row_buckets, long_rows = _build_spmm_opt_buckets(row_lengths)
    long_part_rows = torch.empty((0,), dtype=long_rows.dtype, device=data.device)
    long_part_starts = torch.empty((0,), dtype=torch.int64, device=data.device)
    long_part_ends = torch.empty((0,), dtype=torch.int64, device=data.device)
    long_row_part_ptr = torch.zeros(1, dtype=torch.int64, device=data.device)
    if long_rows.numel() > 0:
        (
            long_part_rows,
            long_part_starts,
            long_part_ends,
            long_row_part_ptr,
        ) = _build_spmm_opt_split_metadata(
            kernel_indptr,
            long_rows,
            part_block_nnz=256,
        )
    return PreparedCsrSpmmOpt(
        data=data,
        kernel_indices=kernel_indices,
        kernel_indptr=kernel_indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        row_lengths=row_lengths,
        max_row_nnz=max_row_nnz,
        row_buckets=row_buckets,
        long_part_rows=long_part_rows,
        long_part_starts=long_part_starts,
        long_part_ends=long_part_ends,
        long_row_ids=long_rows,
        long_row_part_ptr=long_row_part_ptr,
        long_row_fallback_only=False,
    )


def _triton_spmm_csr_impl(
    data,
    indices,
    indptr,
    B,
    n_rows,
    n_dense_cols,
    block_n,
    block_nnz,
):
    device = data.device
    dtype = data.dtype
    if n_rows == 0 or n_dense_cols == 0 or B.shape[0] == 0:
        return torch.zeros((n_rows, n_dense_cols), dtype=dtype, device=device)

    if not _is_complex_dtype(dtype):
        compute_dtype = dtype
        data_in = data
        B_in = B
        if dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
            data_in = data.to(torch.float32)
            B_in = B.to(torch.float32)
        elif dtype == torch.float32:
            compute_dtype = torch.float64
            data_in = data.to(torch.float64)
            B_in = B.to(torch.float64)

        C_compute = torch.empty((n_rows, n_dense_cols), dtype=compute_dtype, device=device)
        grid = (n_rows, triton.cdiv(n_dense_cols, block_n))
        acc_dtype = tl.float64 if compute_dtype == torch.float64 else tl.float32
        _spmm_csr_real_kernel[grid](
            data_in,
            indices,
            indptr,
            B_in,
            C_compute,
            n_rows,
            n_dense_cols,
            B_in.stride(0),
            B_in.stride(1),
            C_compute.stride(0),
            C_compute.stride(1),
            BLOCK_N=block_n,
            BLOCK_NNZ=block_nnz,
            ACC_DTYPE=acc_dtype,
        )
        if compute_dtype != dtype:
            C_compute = C_compute.to(dtype)
        return C_compute

    data_ri = torch.view_as_real(data).contiguous().reshape(-1)
    B_ri = torch.view_as_real(B).contiguous()
    C_ri = torch.empty((n_rows, n_dense_cols, 2), dtype=B_ri.dtype, device=device)
    grid = (n_rows, triton.cdiv(n_dense_cols, block_n))
    acc_dtype = tl.float64 if B_ri.dtype == torch.float64 else tl.float32
    _spmm_csr_complex_kernel[grid](
        data_ri,
        indices,
        indptr,
        B_ri,
        C_ri,
        n_rows,
        n_dense_cols,
        B_ri.stride(0),
        B_ri.stride(1),
        B_ri.stride(2),
        C_ri.stride(0),
        C_ri.stride(1),
        C_ri.stride(2),
        BLOCK_N=block_n,
        BLOCK_NNZ=block_nnz,
        ACC_DTYPE=acc_dtype,
    )
    return torch.view_as_complex(C_ri.contiguous())


@triton.jit
def _spmm_csr_batched_rows_f32_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    rows_ptr,
    n_bucket_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BATCH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    for batch_idx in tl.static_range(0, BATCH):
        ridx = pid_row * BATCH + batch_idx
        active = ridx < n_bucket_rows
        row = tl.load(rows_ptr + ridx, mask=active, other=0)
        start = tl.load(indptr_ptr + row, mask=active, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=active, other=0)
        row_nnz = end - start
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = start + chunk_start + kk
                valid = active & (idx < end)
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                acc = acc + a_val * b_vals
        tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n & active)


@triton.jit
def _spmm_csr_batched_rows_f64_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    rows_ptr,
    n_bucket_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BATCH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    for batch_idx in tl.static_range(0, BATCH):
        ridx = pid_row * BATCH + batch_idx
        active = ridx < n_bucket_rows
        row = tl.load(rows_ptr + ridx, mask=active, other=0)
        start = tl.load(indptr_ptr + row, mask=active, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=active, other=0)
        row_nnz = end - start
        acc = tl.zeros([BLOCK_N], dtype=tl.float64)
        for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = start + chunk_start + kk
                valid = active & (idx < end)
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                acc = acc + a_val * b_vals
        tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n & active)


@triton.jit
def _spmm_csr_vector_rows_f32_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    rows_ptr,
    n_bucket_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val * b_vals
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_vector_rows_f64_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    rows_ptr,
    n_bucket_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val * b_vals
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_split_part_f32_kernel(
    data_ptr,
    indices_ptr,
    b_ptr,
    workspace_ptr,
    part_starts_ptr,
    part_ends_ptr,
    n_parts,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_wm,
    stride_wn,
    BLOCK_N: tl.constexpr,
):
    pid_part = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_part >= n_parts:
        return
    start = tl.load(part_starts_ptr + pid_part)
    end = tl.load(part_ends_ptr + pid_part)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for idx in tl.range(start, end):
        a_val = tl.load(data_ptr + idx)
        a_col = tl.load(indices_ptr + idx)
        b_vals = tl.load(
            b_ptr + a_col * stride_bk + offs_n * stride_bn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + a_val * b_vals
    tl.store(workspace_ptr + pid_part * stride_wm + offs_n * stride_wn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_split_part_f64_kernel(
    data_ptr,
    indices_ptr,
    b_ptr,
    workspace_ptr,
    part_starts_ptr,
    part_ends_ptr,
    n_parts,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_wm,
    stride_wn,
    BLOCK_N: tl.constexpr,
):
    pid_part = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_part >= n_parts:
        return
    start = tl.load(part_starts_ptr + pid_part)
    end = tl.load(part_ends_ptr + pid_part)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for idx in tl.range(start, end):
        a_val = tl.load(data_ptr + idx)
        a_col = tl.load(indices_ptr + idx)
        b_vals = tl.load(
            b_ptr + a_col * stride_bk + offs_n * stride_bn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + a_val * b_vals
    tl.store(workspace_ptr + pid_part * stride_wm + offs_n * stride_wn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_split_reduce_f32_kernel(
    workspace_ptr,
    out_ptr,
    long_rows_ptr,
    row_part_ptr_ptr,
    n_long_rows,
    n_dense_cols,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_long_rows:
        return
    row = tl.load(long_rows_ptr + pid_row)
    part_start = tl.load(row_part_ptr_ptr + pid_row)
    part_end = tl.load(row_part_ptr_ptr + pid_row + 1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for rel_part in tl.range(0, part_end - part_start):
        part_id = part_start + rel_part
        vals = tl.load(
            workspace_ptr + part_id * stride_wm + offs_n * stride_wn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + vals
    tl.store(out_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_split_reduce_f64_kernel(
    workspace_ptr,
    out_ptr,
    long_rows_ptr,
    row_part_ptr_ptr,
    n_long_rows,
    n_dense_cols,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_long_rows:
        return
    row = tl.load(long_rows_ptr + pid_row)
    part_start = tl.load(row_part_ptr_ptr + pid_row)
    part_end = tl.load(row_part_ptr_ptr + pid_row + 1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for rel_part in tl.range(0, part_end - part_start):
        part_id = part_start + rel_part
        vals = tl.load(
            workspace_ptr + part_id * stride_wm + offs_n * stride_wn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + vals
    tl.store(out_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


def _run_spmm_opt_bucket(prepared, bucket, B, C_out, block_n):
    rows = bucket["rows"]
    if rows.numel() == 0:
        return
    dtype = prepared.data.dtype
    kernel_map = {
        ("batched", torch.float32): _spmm_csr_batched_rows_f32_kernel,
        ("batched", torch.float64): _spmm_csr_batched_rows_f64_kernel,
        ("vector", torch.float32): _spmm_csr_vector_rows_f32_kernel,
        ("vector", torch.float64): _spmm_csr_vector_rows_f64_kernel,
    }
    if bucket["kind"] == "batched":
        batch_rows = int(bucket["batch_rows"])
        grid = (triton.cdiv(rows.numel(), batch_rows), triton.cdiv(B.shape[1], block_n))
        kernel = kernel_map[(bucket["kind"], dtype)]
        kernel[grid](
            prepared.data,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            B,
            C_out,
            rows,
            rows.numel(),
            B.shape[1],
            B.stride(0),
            B.stride(1),
            C_out.stride(0),
            C_out.stride(1),
            BATCH=batch_rows,
            BLOCK_N=block_n,
            BLOCK_NNZ=bucket["block_nnz"],
        )
        return

    grid = (rows.numel(), triton.cdiv(B.shape[1], block_n))
    kernel = kernel_map[(bucket["kind"], dtype)]
    kernel[grid](
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        B,
        C_out,
        rows,
        rows.numel(),
        B.shape[1],
        B.stride(0),
        B.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        BLOCK_N=block_n,
        BLOCK_NNZ=bucket["block_nnz"],
    )


def _run_spmm_opt_split_bucket(prepared, B, C_out, block_n):
    if prepared.long_part_rows.numel() == 0:
        return False
    workspace = torch.empty(
        (prepared.long_part_rows.numel(), B.shape[1]),
        dtype=B.dtype,
        device=B.device,
    )
    split_kernel = (
        _spmm_csr_split_part_f64_kernel
        if B.dtype == torch.float64
        else _spmm_csr_split_part_f32_kernel
    )
    reduce_kernel = (
        _spmm_csr_split_reduce_f64_kernel
        if B.dtype == torch.float64
        else _spmm_csr_split_reduce_f32_kernel
    )
    split_grid = (
        prepared.long_part_rows.numel(),
        triton.cdiv(B.shape[1], block_n),
    )
    split_kernel[split_grid](
        prepared.data,
        prepared.kernel_indices,
        B,
        workspace,
        prepared.long_part_starts,
        prepared.long_part_ends,
        prepared.long_part_rows.numel(),
        B.shape[1],
        B.stride(0),
        B.stride(1),
        workspace.stride(0),
        workspace.stride(1),
        BLOCK_N=block_n,
    )
    reduce_grid = (
        prepared.long_row_ids.numel(),
        triton.cdiv(B.shape[1], block_n),
    )
    reduce_kernel[reduce_grid](
        workspace,
        C_out,
        prepared.long_row_ids,
        prepared.long_row_part_ptr,
        prepared.long_row_ids.numel(),
        B.shape[1],
        workspace.stride(0),
        workspace.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        BLOCK_N=block_n,
    )
    return False


def _triton_spmm_csr_impl_opt_prepared(prepared, B):
    if not prepared.supports_opt:
        raise TypeError("spmm opt only supports float32 and float64")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as the sparse matrix")
    if B.dtype != prepared.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if B.shape[0] != prepared.n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={prepared.n_cols}, got {B.shape[0]}")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    B = B.contiguous()
    block_n = _select_spmm_opt_block_n(int(B.shape[1]))
    C_out = torch.zeros((prepared.n_rows, int(B.shape[1])), dtype=B.dtype, device=B.device)
    long_row_fallback_used = False
    for bucket in prepared.row_buckets:
        if bucket["kind"] == "split":
            long_row_fallback_used = _run_spmm_opt_split_bucket(prepared, B, C_out, block_n)
            continue
        _run_spmm_opt_bucket(prepared, bucket, B, C_out, block_n)
    return C_out, long_row_fallback_used

def flagsparse_spmm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
):
    """CSR SpMM: C = A @ B using Triton.

    A is provided as CSR arrays; B is a dense CUDA tensor with shape (n_cols, n_dense_cols).
    This staged implementation is the row-major, non-transpose subset of
    AlphaSparse CSR ALG1 (`csrspmm_rb_sr`) expressed in Triton.
    """
    if block_n is not None and block_n <= 0:
        raise ValueError("block_n must be positive when provided")
    if block_nnz is not None and block_nnz <= 0:
        raise ValueError("block_nnz must be positive when provided")
    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive when provided")

    data, kernel_indices, kernel_indptr, B, n_rows, _, n_dense_cols = _prepare_spmm_csr_inputs(
        data, indices, indptr, B, shape
    )
    max_row_nnz = (
        int(torch.max(kernel_indptr[1:] - kernel_indptr[:-1]).item())
        if n_rows > 0
        else 0
    )
    launch = _resolve_spmm_alg1_launch_config(
        n_dense_cols,
        max_row_nnz,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )

    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != data.device:
            raise ValueError("out must be on the same CUDA device as the inputs")
        if out.shape != (n_rows, n_dense_cols) or out.dtype != data.dtype:
            raise ValueError("out shape/dtype must match result")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C = _triton_spmm_csr_impl(
        data,
        kernel_indices,
        kernel_indptr,
        B,
        n_rows,
        n_dense_cols,
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if out is not None:
        out.copy_(C)
        C = out
    if return_time:
        return C, elapsed_ms
    return C


def flagsparse_spmm_csr_opt(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    prepared=None,
    out=None,
    return_time=False,
):
    """CSR SpMM-opt: native float32/float64 bucketed path for CSR @ dense."""
    if prepared is not None and not isinstance(prepared, PreparedCsrSpmmOpt):
        raise TypeError("prepared must be a PreparedCsrSpmmOpt instance")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmm_csr_opt(data, indices, indptr, shape)
    elif shape is not None:
        resolved_shape = (int(shape[0]), int(shape[1]))
        if resolved_shape != prepared.shape:
            raise ValueError(
                f"shape {resolved_shape} does not match prepared.shape {prepared.shape}"
            )
    if B is None:
        raise ValueError("B is required")
    if not prepared.supports_opt:
        raise TypeError("flagsparse_spmm_csr_opt only supports float32 and float64")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as sparse matrix data")
    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != prepared.data.device:
            raise ValueError("out must be on the same CUDA device as sparse matrix data")
        if out.shape != (prepared.n_rows, int(B.shape[1])) or out.dtype != prepared.data.dtype:
            raise ValueError("out shape/dtype must match result")
    t0 = None
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    C, _long_row_fallback_used = _triton_spmm_csr_impl_opt_prepared(prepared, B)
    elapsed_ms = None
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        out.copy_(C)
        C = out
    if return_time:
        return C, elapsed_ms
    return C


def _spmm_opt_reference_error(candidate, reference, value_dtype):
    atol, rtol = _spmm_coo_reference_tolerance(value_dtype)
    if candidate.numel() == 0:
        return 0.0
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    return float(torch.max(diff / denom).item())


def benchmark_spmm_opt_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
):
    """Benchmark SpMM base vs opt against the same high-precision PyTorch reference."""
    if value_dtype not in (torch.float32, torch.float64):
        raise TypeError("benchmark_spmm_opt_case only supports float32 and float64")
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    B = _build_random_dense((n_cols, n_dense_cols), value_dtype, device)
    shape = (n_rows, n_cols)
    prepared = prepare_spmm_csr_opt(data, indices, indptr, shape)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    base_values = flagsparse_spmm_csr(data, indices, indptr, B, shape)
    torch.cuda.synchronize()
    base_first_call_ms = (time.perf_counter() - t0) * 1000.0
    base_values, base_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr(data, indices, indptr, B, shape),
        warmup=warmup,
        iters=iters,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    opt_values = flagsparse_spmm_csr_opt(B=B, prepared=prepared)
    torch.cuda.synchronize()
    opt_first_call_ms = (time.perf_counter() - t0) * 1000.0
    opt_values, opt_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr_opt(B=B, prepared=prepared),
        warmup=warmup,
        iters=iters,
    )

    indptr64 = indptr.to(torch.int64)
    indices64 = indices.to(torch.int64)
    csr_ref = torch.sparse_csr_tensor(
        indptr64,
        indices64,
        data.to(torch.float64 if value_dtype == torch.float32 else value_dtype),
        size=shape,
        device=device,
    )
    ref = torch.sparse.mm(
        csr_ref,
        B.to(torch.float64 if value_dtype == torch.float32 else value_dtype),
    ).to(value_dtype)

    pt_ms = None
    try:
        pt_sparse = torch.sparse_csr_tensor(
            indptr64,
            indices64,
            data,
            size=shape,
            device=device,
        )
        pt_op = lambda: torch.sparse.mm(pt_sparse, B)
        _, pt_ms = _benchmark_cuda_op(pt_op, warmup=warmup, iters=iters)
    except Exception:
        pt_ms = None

    cu_ms = None
    if run_cusparse and cp is not None and cpx_sparse is not None:
        try:
            data_cp = _cupy_from_torch(data)
            indices_cp = _cupy_from_torch(indices.to(torch.int64))
            indptr_cp = _cupy_from_torch(indptr)
            B_cp = _cupy_from_torch(B)
            A_csr = cpx_sparse.csr_matrix((data_cp, indices_cp, indptr_cp), shape=shape)
            _, cu_ms = _benchmark_cuda_op(lambda: A_csr @ B_cp, warmup=warmup, iters=iters)
        except Exception:
            cu_ms = None

    err_base = _spmm_opt_reference_error(base_values, ref, value_dtype)
    err_opt = _spmm_opt_reference_error(opt_values, ref, value_dtype)
    return {
        "parameters": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "n_dense_cols": n_dense_cols,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
        },
        "performance": {
            "base_ms": base_ms,
            "base_first_call_ms": base_first_call_ms,
            "opt_ms": opt_ms,
            "opt_first_call_ms": opt_first_call_ms,
            "pt_ms": pt_ms,
            "cu_ms": cu_ms,
            "opt_vs_base": (base_ms / opt_ms if opt_ms and opt_ms > 0 else None),
            "opt_vs_pt": (pt_ms / opt_ms if pt_ms is not None and opt_ms > 0 else None),
            "opt_vs_cu": (cu_ms / opt_ms if cu_ms is not None and opt_ms > 0 else None),
        },
        "verification": {
            "err_base": err_base,
            "err_opt": err_opt,
            "base_ok": err_base <= 1.0,
            "opt_ok": err_opt <= 1.0,
            "status": ("PASS" if err_opt <= 1.0 else "FAIL"),
        },
        "backend_status": {
            "long_row_fallback_used": bool(prepared.long_row_fallback_only),
            "flagsparse_internal_route": "csr-opt-bucketed",
        },
        "samples": {
            "base": base_values,
            "opt": opt_values,
            "reference": ref,
        },
    }


def benchmark_spmm_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    run_cusparse=True,
):
    """Benchmark Triton CSR SpMM vs PyTorch sparse.mm and CuPy/cuSPARSE CSR @ dense."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    B = _build_random_dense((n_cols, n_dense_cols), value_dtype, device)
    shape = (n_rows, n_cols)
    max_row_nnz = int(torch.max(indptr[1:] - indptr[:-1]).item()) if n_rows > 0 else 0
    launch = _resolve_spmm_alg1_launch_config(
        n_dense_cols,
        max_row_nnz,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )

    triton_kwargs = {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "B": B,
        "shape": shape,
        "block_n": launch["block_n"],
        "block_nnz": launch["block_nnz"],
        "max_segments": launch["max_segments"],
        "return_time": False,
    }

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = flagsparse_spmm_csr(**triton_kwargs)
    torch.cuda.synchronize()
    triton_first_call_ms = (time.perf_counter() - t0) * 1000.0
    triton_C, triton_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr(**triton_kwargs),
        warmup=warmup,
        iters=iters,
    )

    indptr64 = indptr.to(torch.int64)
    indices64 = indices.to(torch.int64)
    row_indices = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )

    pytorch_reason = None
    pytorch_values = None
    pytorch_ms = None
    pytorch_format = "CSR"
    try:
        csr_pt = torch.sparse_csr_tensor(indptr64, indices64, data, size=shape, device=device)
        pytorch_op = lambda: torch.sparse.mm(csr_pt, B)
        if value_dtype in (torch.float16, torch.bfloat16):
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.float32), size=shape, device=device)
            expected = torch.sparse.mm(csr_ref, B.to(torch.float32)).to(value_dtype)
        elif value_dtype == torch.float32:
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.float64), size=shape, device=device)
            expected = torch.sparse.mm(csr_ref, B.to(torch.float64)).to(value_dtype)
        elif value_dtype == torch.complex64:
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.complex128), size=shape, device=device)
            expected = torch.sparse.mm(csr_ref, B.to(torch.complex128)).to(value_dtype)
        else:
            expected = torch.sparse.mm(csr_pt, B)
    except Exception as exc:
        pytorch_format = "COO"
        pytorch_reason = f"CSR fallback: {exc}"
        coo = torch.sparse_coo_tensor(
            torch.stack([row_indices, indices64]),
            data,
            shape,
            device=device,
        ).coalesce()
        pytorch_op = lambda: torch.sparse.mm(coo, B)
        if value_dtype in (torch.float16, torch.bfloat16):
            expected = torch.sparse.mm(coo.to(torch.float32), B.to(torch.float32)).to(value_dtype)
        elif value_dtype == torch.float32:
            expected = torch.sparse.mm(coo.to(torch.float64), B.to(torch.float64)).to(value_dtype)
        elif value_dtype == torch.complex64:
            expected = torch.sparse.mm(coo.to(torch.complex128), B.to(torch.complex128)).to(value_dtype)
        else:
            expected = torch.sparse.mm(coo, B)

    pytorch_values = expected
    try:
        pytorch_values, pytorch_ms = _benchmark_cuda_op(
            pytorch_op, warmup=warmup, iters=iters
        )
    except Exception as exc:
        pytorch_reason = str(exc) if pytorch_reason is None else f"{pytorch_reason}; timing: {exc}"

    triton_metrics = _spmm_validation_metrics(triton_C, expected)
    triton_match = triton_metrics["strict_allclose_match"]

    cusparse_ms = None
    cusparse_match = None
    cusparse_reason = None
    cusparse_values = None
    cusparse_metrics = None
    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    if run_cusparse:
        if cp is None or cpx_sparse is None:
            cusparse_reason = "CuPy/cuSPARSE is not available"
        elif value_dtype not in _cupy_supported_dtypes:
            cusparse_reason = "float16/bfloat16 not supported by CuPy sparse; skipped"
        else:
            try:
                data_cp = _cupy_from_torch(data)
                indices_cp = _cupy_from_torch(indices.to(torch.int64))
                indptr_cp = _cupy_from_torch(indptr)
                B_cp = _cupy_from_torch(B)
                A_csr = cpx_sparse.csr_matrix(
                    (data_cp, indices_cp, indptr_cp), shape=shape
                )
                cusparse_values_cp, cusparse_ms = _benchmark_cuda_op(
                    lambda: A_csr @ B_cp, warmup=warmup, iters=iters
                )
                cusparse_values = _torch_from_cupy(cusparse_values_cp)
                cusparse_metrics = _spmm_validation_metrics(cusparse_values, expected)
                cusparse_match = cusparse_metrics["strict_allclose_match"]
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if (pytorch_ms is not None and triton_ms > 0) else None
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms if (cusparse_ms is not None and triton_ms > 0) else None
    )
    threshold = _spmm_relative_threshold(value_dtype)
    return {
        "parameters": {
            "format": "csr",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "n_dense_cols": n_dense_cols,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "max_segments": launch["max_segments"],
            "required_segments": launch["required_segments"],
            "alg1_warp_size": launch["warp_size"],
            "alg1_factor": launch["factor"],
            "auto_max_segments": launch["auto_max_segments"],
            "run_cusparse": run_cusparse,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "triton_first_call_ms": triton_first_call_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_reference": triton_match,
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_metrics["max_abs_error"],
            "triton_max_abs_error": triton_metrics["max_abs_error"],
            "triton_max_relative_error": triton_metrics["max_relative_error"],
            "triton_sum_relative_error": triton_metrics["sum_relative_error"],
            "triton_relative_threshold": triton_metrics["relative_threshold"],
            "triton_strict_allclose_match": triton_metrics["strict_allclose_match"],
            "pytorch_match_reference": True,
            "pytorch_max_error": 0.0,
            "pytorch_max_abs_error": 0.0,
            "pytorch_max_relative_error": 0.0,
            "pytorch_sum_relative_error": 0.0,
            "pytorch_relative_threshold": threshold,
            "cusparse_match_reference": cusparse_match,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": (cusparse_metrics["max_abs_error"] if cusparse_metrics is not None else None),
            "cusparse_max_abs_error": (cusparse_metrics["max_abs_error"] if cusparse_metrics is not None else None),
            "cusparse_max_relative_error": (cusparse_metrics["max_relative_error"] if cusparse_metrics is not None else None),
            "cusparse_sum_relative_error": (cusparse_metrics["sum_relative_error"] if cusparse_metrics is not None else None),
            "cusparse_relative_threshold": (cusparse_metrics["relative_threshold"] if cusparse_metrics is not None else threshold),
            "cusparse_strict_allclose_match": (cusparse_metrics["strict_allclose_match"] if cusparse_metrics is not None else None),
        },
        "backend_status": {
            "pytorch_unavailable_reason": pytorch_reason,
            "pytorch_sparse_format": pytorch_format,
            "cusparse_unavailable_reason": cusparse_reason,
            "flagsparse_internal_route": "csr-alg1",
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_C,
            "reference": expected,
            "cusparse": cusparse_values,
        },
    }
def comprehensive_spmm_test(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    run_cusparse=True,
):
    """Full SpMM benchmark entry for one configuration."""
    return benchmark_spmm_case(
        n_rows=n_rows,
        n_cols=n_cols,
        nnz=nnz,
        n_dense_cols=n_dense_cols,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
        run_cusparse=run_cusparse,
    )
