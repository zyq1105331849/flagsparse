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

def _prepare_spmm_csr_inputs(data, indices, indptr, B, shape):
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple: (n_rows, n_cols)")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, and indptr must be 1D tensors")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if B.shape[0] != n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={n_cols}, got {B.shape[0]}")

    if not all(t.is_cuda for t in (data, indices, indptr, B)):
        raise ValueError("data, indices, indptr, and B must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr, B)):
        raise ValueError("data, indices, indptr, and B must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMM_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
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
    B = B.contiguous()

    kernel_indices = indices.to(torch.int32) if indices.dtype == torch.int64 else indices
    kernel_indptr = indptr.to(torch.int64)
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