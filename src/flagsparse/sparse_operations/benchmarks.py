"""Benchmarks for gather, scatter, and SpMV."""

from ._common import *

from .gather_scatter import (
    _cusparse_spmv,
    _make_gather_selector_matrix,
    _make_scatter_selector_matrix,
    _pytorch_scatter_impl,
    _triton_gather_impl,
    _triton_scatter_impl,
)
from .spmv_csr import flagsparse_spmv_csr, prepare_spmv_csr

def benchmark_gather_case(
    dense_size=65536,
    nnz=4096,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_size=1024,
    run_cusparse=True,
):
    """Benchmark Triton vs PyTorch indexing vs cuSPARSE-backed COO SpMV."""
    device = torch.device("cuda")
    dense_vector = _build_random_dense(dense_size, value_dtype, device)
    indices = _build_indices(nnz, dense_size, index_dtype, device, unique=False)

    dense_vector, indices, kernel_indices = _prepare_inputs(dense_vector, indices)
    expected = dense_vector[indices]

    pytorch_op = lambda: dense_vector[indices]
    triton_op = lambda: _triton_gather_impl(dense_vector, kernel_indices, block_size=block_size)

    pytorch_values, pytorch_ms = _benchmark_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    triton_values, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_values, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_values - expected)).item())
        if nnz > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if run_cusparse:
        skip_reason = _cusparse_baseline_skip_reason(value_dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                selector_matrix = _make_gather_selector_matrix(
                    indices, dense_vector.numel(), dense_vector.dtype
                )
                cusparse_op = lambda: _cusparse_spmv(selector_matrix, dense_vector)
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if nnz > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_values,
        },
    }


def benchmark_scatter_case(
    dense_size=65536,
    nnz=4096,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_size=1024,
    run_cusparse=True,
    unique_indices=True,
):
    """Benchmark Triton scatter vs PyTorch index_copy vs cuSPARSE-backed COO SpMV."""
    device = torch.device("cuda")
    sparse_values = _build_random_dense(nnz, value_dtype, device)
    indices = _build_indices(nnz, dense_size, index_dtype, device, unique=unique_indices)

    sparse_values, indices, kernel_indices, dense_size = _prepare_scatter_inputs(
        sparse_values, indices, dense_size=dense_size, out=None
    )
    expected = _pytorch_scatter_impl(sparse_values, indices, dense_size)

    pytorch_op = lambda: _pytorch_scatter_impl(sparse_values, indices, dense_size)
    triton_op = lambda: _triton_scatter_impl(
        sparse_values,
        kernel_indices,
        dense_size=dense_size,
        out=None,
        block_size=block_size,
    )

    pytorch_values, pytorch_ms = _benchmark_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    triton_values, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_values, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_values - expected)).item())
        if dense_size > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if run_cusparse:
        skip_reason = _cusparse_baseline_skip_reason(value_dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                selector_matrix = _make_scatter_selector_matrix(
                    indices, dense_size, sparse_values.dtype
                )
                cusparse_op = lambda: _cusparse_spmv(selector_matrix, sparse_values)
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if dense_size > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "unique_indices": unique_indices,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_values,
        },
    }


def benchmark_spmv_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_nnz=256,
    max_segments=None,
    run_cusparse=True,
):
    """Benchmark Triton CSR SpMV vs cuSPARSE (CuPy CSR @ x)."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    x = _build_random_dense(n_cols, value_dtype, device)
    shape = (n_rows, n_cols)
    prepared = prepare_spmv_csr(
        data,
        indices,
        indptr,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )
    triton_op = lambda: flagsparse_spmv_csr(
        x=x,
        prepared=prepared,
        return_time=False,
    )
    triton_y, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)
    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    if (
        cp is not None
        and cpx_sparse is not None
        and value_dtype in _cupy_supported_dtypes
    ):
        data_cp = _cupy_from_torch(data)
        indices_cp = _cupy_from_torch(indices.to(torch.int64))
        indptr_cp = _cupy_from_torch(indptr)
        x_cp = _cupy_from_torch(x)
        A_csr = cpx_sparse.csr_matrix(
            (data_cp, indices_cp, indptr_cp), shape=shape
        )
        ref_y = A_csr @ x_cp
        expected = _torch_from_cupy(ref_y)
    else:
        row_indices = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        col_ind = indices.to(torch.int64)
        coo = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_ind]),
            data,
            shape,
            device=device,
        ).coalesce()
        x_2d = x.unsqueeze(1)
        if value_dtype in (torch.float16, torch.bfloat16):
            coo_f32 = coo.to(torch.float32)
            x_2d_f32 = x_2d.to(torch.float32)
            expected = torch.sparse.mm(coo_f32, x_2d_f32).squeeze(1).to(value_dtype)
        else:
            expected = torch.sparse.mm(coo, x_2d).squeeze(1)
    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_y, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_y - expected)).item())
        if n_rows > 0
        else 0.0
    )
    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if (
        run_cusparse
        and cp is not None
        and cpx_sparse is not None
        and value_dtype in _cupy_supported_dtypes
    ):
        skip_reason = _cusparse_baseline_skip_reason(value_dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                cusparse_op = lambda: _torch_from_cupy(
                    A_csr @ _cupy_from_torch(x)
                )
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if n_rows > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)
    elif run_cusparse and value_dtype not in _cupy_supported_dtypes:
        cusparse_reason = (
            "float16/bfloat16 not supported by CuPy sparse; skipped"
        )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )
    return {
        "parameters": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_reference": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_reference": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {"triton": triton_y, "reference": expected},
    }


def benchmark_performance(
    dense_size=65536,
    nnz=4096,
    dtype=torch.float32,
    index_dtype=torch.int32,
):
    """Backward-compatible benchmark entry."""
    result = benchmark_gather_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=10,
        iters=100,
        run_cusparse=False,
    )
    return {
        "triton_time_ms": result["performance"]["triton_ms"],
        "results_match": result["verification"]["triton_match_pytorch"],
        "dtype": str(dtype),
        "index_dtype": str(index_dtype),
        "dense_size": dense_size,
        "nnz": nnz,
    }


def comprehensive_gather_test(
    dense_size=100000,
    nnz=10000,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
):
    """Full test entry for one configuration."""
    return benchmark_gather_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        run_cusparse=run_cusparse,
    )


def comprehensive_scatter_test(
    dense_size=100000,
    nnz=10000,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
    unique_indices=True,
):
    """Full scatter test entry for one configuration."""
    return benchmark_scatter_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        run_cusparse=run_cusparse,
        unique_indices=unique_indices,
    )
