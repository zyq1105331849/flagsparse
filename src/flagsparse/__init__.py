"""FlagSparse package."""

__version__ = "1.0.0"

__all__ = [
    "flagsparse_gather",
    "flagsparse_scatter",
    "pytorch_index_gather",
    "pytorch_index_scatter",
    "cusparse_spmv_gather",
    "cusparse_spmv_scatter",
    "benchmark_gather_case",
    "benchmark_scatter_case",
    "benchmark_performance",
    "comprehensive_gather_test",
    "comprehensive_scatter_test",
    "PreparedCoo",
    "PreparedCsrSpmv",
    "prepare_spmv_csr",
    "prepare_spmv_coo",
    "prepare_spmv_coo_tocsr",
    "flagsparse_spmv_csr",
    "flagsparse_spmv_coo",
    "flagsparse_spmv_coo_tocsr",
    "flagsparse_spsv_csr",
    "flagsparse_spsv_coo",
    "benchmark_spmv_case",
    "create_csr_matrix",
    "create_coo_matrix",
    "create_csc_matrix",
    "create_bsr_matrix",
    "create_sell_matrix",
    "create_blocked_ell_matrix",
    "coo_to_csr",
    "coo_to_csc",
    "coo_to_bsr",
    "coo_to_sell",
    "coo_to_blocked_ell",
    "CSRMatrix",
    "COOMatrix",
    "CSCMatrix",
    "BSRMatrix",
    "SELLMatrix",
    "BLOCKEDELLMatrix",
    "generate_random_sparse_matrix",
    "read_mtx_file",
]

_OPS_EXPORTS = {
    "flagsparse_gather",
    "flagsparse_scatter",
    "pytorch_index_gather",
    "pytorch_index_scatter",
    "cusparse_spmv_gather",
    "cusparse_spmv_scatter",
    "benchmark_gather_case",
    "benchmark_scatter_case",
    "benchmark_performance",
    "comprehensive_gather_test",
    "comprehensive_scatter_test",
    "PreparedCoo",
    "PreparedCsrSpmv",
    "prepare_spmv_csr",
    "prepare_spmv_coo",
    "prepare_spmv_coo_tocsr",
    "flagsparse_spmv_csr",
    "flagsparse_spmv_coo",
    "flagsparse_spmv_coo_tocsr",
    "flagsparse_spsv_csr",
    "flagsparse_spsv_coo",
    "benchmark_spmv_case",
}

_FORMAT_EXPORTS = {
    "create_csr_matrix",
    "create_coo_matrix",
    "create_csc_matrix",
    "create_bsr_matrix",
    "create_sell_matrix",
    "create_blocked_ell_matrix",
    "coo_to_csr",
    "coo_to_csc",
    "coo_to_bsr",
    "coo_to_sell",
    "coo_to_blocked_ell",
    "CSRMatrix",
    "COOMatrix",
    "CSCMatrix",
    "BSRMatrix",
    "SELLMatrix",
    "BLOCKEDELLMatrix",
    "generate_random_sparse_matrix",
    "read_mtx_file",
}


def __getattr__(name):
    if name in _OPS_EXPORTS:
        from . import sparse_operations as _ops
        return getattr(_ops, name)
    if name in _FORMAT_EXPORTS:
        from . import sparse_formats as _formats
        return getattr(_formats, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _OPS_EXPORTS | _FORMAT_EXPORTS)
