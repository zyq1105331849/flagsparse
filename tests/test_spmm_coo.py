"""
COO SpMM tests: load SuiteSparse .mtx, batch run, output error and performance.
Supports: multi .mtx files, value_dtype / index_dtype, CSV export, synthetic cases,
API validation checks, and PyTorch / CuPy comparison baselines.

This test module targets the current FlagSparse native COO SpMM implementation.
The default public route is a sorted row-run Triton COO kernel. A second native
atomic COO route is retained for internal parity checks and debug.
"""
import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as ast
import flagsparse.sparse_operations.spmm_coo as ast_ops



VALUE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]
INDEX_DTYPES = [torch.int32, torch.int64]
CSV_VALUE_DTYPES = [torch.float32, torch.float64]
CSV_INDEX_DTYPES = [torch.int32]
TEST_CASES = [
    (512, 512, 4096, 16),
    (1024, 1024, 16384, 32),
    (2048, 2048, 65536, 64),
    (4096, 4096, 131072, 64),
]
COO_TILE_CASES = [
    (256, 256, 4096, 4),
    (256, 256, 4096, 5),
    (256, 256, 4096, 12),
    (256, 256, 4096, 24),
    (256, 256, 4096, 48),
    (256, 256, 4096, 96),
]
WARMUP = 10
ITERS = 50
DEFAULT_BLOCK_N = None
DEFAULT_BLOCK_NNZ = 256
DUPLICATE_CASE_DENSE_COLS = 48




def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(value):
    return "N/A" if value is None else f"{value:.4f}"


def _fmt_speedup(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return "N/A"
    return f"{other_ms / triton_ms:.2f}x"


def _fmt_err(value):
    return "N/A" if value is None else f"{value:.2e}"


def _fmt_check(value):
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"

def _status_label(value):
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"

def _normalize_csv_path(csv_path):
    csv_path = str(csv_path)
    if not csv_path.lower().endswith(".csv"):
        csv_path = f"{csv_path}.csv"
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return csv_path

def _fmt_launch_value(value):
    return "auto" if value is None else str(value)


def _build_values(length, value_dtype, device):
    shape = (length,)
    if value_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=value_dtype, device=device)
    if value_dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if value_dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"Unsupported value dtype: {value_dtype}")


def _build_dense_matrix(n_rows, n_cols, value_dtype, device):
    shape = (n_rows, n_cols)
    if value_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=value_dtype, device=device)
    if value_dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if value_dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"Unsupported value dtype: {value_dtype}")


def _tolerance_for_dtype(value_dtype):
    if value_dtype == torch.float16:
        return 2e-3, 2e-3
    if value_dtype == torch.bfloat16:
        return 1e-1, 1e-1
    if value_dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-2
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-12, 1e-10
    return 1e-6, 1e-5


def _scaled_allclose_error(candidate, reference, value_dtype=None):
    if candidate.numel() == 0:
        return 0.0
    dtype = reference.dtype if value_dtype is None else value_dtype
    atol, rtol = _tolerance_for_dtype(dtype)
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    return float(torch.max(diff / denom).item())

def load_mtx_to_coo_torch(file_path, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    mm_field = "real"
    mm_symmetry = "general"
    data_lines = []
    header_info = None
    for line in lines:
        line = line.strip()
        if line.startswith("%%MatrixMarket"):
            parts = line.split()
            if len(parts) >= 5:
                mm_field = parts[3].lower()
                mm_symmetry = parts[4].lower()
            continue
        if line.startswith("%"):
            continue
        if not header_info and line:
            parts = line.split()
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            header_info = (n_rows, n_cols, nnz)
            continue
        if line:
            data_lines.append(line)

    if header_info is None:
        raise ValueError(f"Cannot parse .mtx header: {file_path}")

    n_rows, n_cols, nnz = header_info
    if nnz == 0:
        empty_index = torch.tensor([], dtype=torch.int64, device=device)
        data = torch.tensor([], dtype=dtype, device=device)
        return data, empty_index, empty_index.clone(), (n_rows, n_cols)

    if mm_field == "complex" and dtype not in (torch.complex64, torch.complex128):
        raise TypeError(
            f"Matrix Market file {file_path} stores complex values but requested dtype {dtype}"
        )

    is_pattern = mm_field == "pattern"
    is_complex = mm_field == "complex"
    is_symmetric = mm_symmetry == "symmetric"
    is_skew = mm_symmetry == "skew-symmetric"
    is_hermitian = mm_symmetry == "hermitian"

    entries = {}

    def _accumulate(row_idx, col_idx, value):
        key = (row_idx, col_idx)
        entries[key] = entries.get(key, 0.0) + value

    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        row_idx = int(parts[0]) - 1
        col_idx = int(parts[1]) - 1
        if not (0 <= row_idx < n_rows and 0 <= col_idx < n_cols):
            continue

        if is_pattern:
            value = 1.0
        elif is_complex:
            if len(parts) < 4:
                raise ValueError(f"Complex Matrix Market entry is missing an imaginary part: {line}")
            value = complex(float(parts[2]), float(parts[3]))
        else:
            if len(parts) < 3:
                raise ValueError(f"Matrix Market entry is missing a numeric value: {line}")
            value = float(parts[2])

        _accumulate(row_idx, col_idx, value)
        if row_idx != col_idx:
            if is_symmetric and 0 <= col_idx < n_rows and 0 <= row_idx < n_cols:
                _accumulate(col_idx, row_idx, value)
            elif is_skew and 0 <= col_idx < n_rows and 0 <= row_idx < n_cols:
                _accumulate(col_idx, row_idx, -value)
            elif is_hermitian and 0 <= col_idx < n_rows and 0 <= row_idx < n_cols:
                twin = value.conjugate() if isinstance(value, complex) else value
                _accumulate(col_idx, row_idx, twin)

    sorted_entries = sorted(entries.items(), key=lambda item: item[0])
    rows = [key[0] for key, _ in sorted_entries]
    cols = [key[1] for key, _ in sorted_entries]
    vals = [value for _, value in sorted_entries]

    data = torch.tensor(vals, dtype=dtype, device=device)
    row = torch.tensor(rows, dtype=torch.int64, device=device)
    col = torch.tensor(cols, dtype=torch.int64, device=device)
    return data, row, col, (n_rows, n_cols)

def _normalize_route(route):
    route = str(route).strip().lower()
    if route not in ("rowrun", "atomic", "compare"):
        raise ValueError("route must be one of: rowrun, atomic, compare")
    return route



def _selected_route(route):
    route = _normalize_route(route)
    return "rowrun" if route == "compare" else route



def _route_label(route):
    labels = {
        "rowrun": "COO native row-run",
        "atomic": "COO native atomic",
        "compare": "COO native row-run (compare mode)",
    }
    if route not in labels:
        raise ValueError(f"Unsupported route label: {route}")
    return labels[route]



def _empty_pairwise_summary():
    return {
        "match": None,
        "error_ratio": None,
        "max_abs_error": None,
        "max_relative_error": None,
        "sum_relative_error": None,
    }



def _prepare_canonical_case(data, row, col, shape, B):
    native_data, native_row, native_col, native_B, n_rows, n_cols, n_dense_cols = ast_ops._prepare_spmm_coo_inputs(
        data, row, col, B, shape
    )
    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        _,
        _,
        _,
        output_dtype,
        _,
    ) = ast_ops._prepare_spmm_coo_canonical_prepared(
        native_data,
        native_row,
        native_col,
        native_B,
        n_rows,
        n_cols,
        n_dense_cols,
    )
    native_coo = ast_ops._build_torch_sparse_coo(native_data, native_row, native_col, shape)
    return {
        "native_data": native_data,
        "native_row": native_row,
        "native_col": native_col,
        "native_B": native_B,
        "native_coo": native_coo,
        "canonical_data": canonical_data,
        "canonical_row": canonical_row,
        "canonical_col": canonical_col,
        "canonical_B": canonical_B,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_dense_cols": n_dense_cols,
        "output_dtype": output_dtype,
    }



def _build_pytorch_reference(data, row, col, shape, B, prepared=None):
    prepared = _prepare_canonical_case(data, row, col, shape, B) if prepared is None else prepared
    expected = ast_ops._build_spmm_coo_pytorch_reference_from_canonical(
        prepared["canonical_data"],
        prepared["canonical_row"],
        prepared["canonical_col"],
        prepared["canonical_B"],
        shape,
        prepared["output_dtype"],
    )
    pytorch_op = lambda: torch.sparse.mm(prepared["native_coo"], prepared["native_B"])
    return expected, pytorch_op, "COO", None



def _benchmark_spmm_coo_route(
    data,
    row,
    col,
    B,
    shape,
    warmup,
    iters,
    route="rowrun",
    block_n=None,
    block_nnz=DEFAULT_BLOCK_NNZ,
    prepared=None,
):
    selected_route = _selected_route(route)
    prepared = _prepare_canonical_case(data, row, col, shape, B) if prepared is None else prepared
    return ast_ops._benchmark_spmm_coo_canonical_route(
        prepared["canonical_data"],
        prepared["canonical_row"],
        prepared["canonical_col"],
        prepared["canonical_B"],
        prepared["n_rows"],
        prepared["n_dense_cols"],
        prepared["output_dtype"],
        warmup,
        iters,
        block_n,
        block_nnz,
        selected_route,
    )

def _summarize_route_output(values, reference, value_dtype, ms=None, first_call_ms=None, cusparse_values=None):
    metrics = ast_ops._spmm_validation_metrics(values, reference)
    atol, rtol = _tolerance_for_dtype(value_dtype)
    summary = {
        "ms": ms,
        "first_call_ms": first_call_ms,
        "ok_pt": torch.allclose(values, reference, atol=atol, rtol=rtol),
        "err_pt": _scaled_allclose_error(values, reference, value_dtype),
        "max_abs_error": metrics["max_abs_error"],
        "max_relative_error": metrics["max_relative_error"],
        "ok_cu": None,
        "err_cu": None,
        "error": None,
    }
    if cusparse_values is not None:
        summary["ok_cu"] = torch.allclose(values, cusparse_values, atol=atol, rtol=rtol)
        summary["err_cu"] = _scaled_allclose_error(values, cusparse_values, value_dtype)
    return summary



def _pairwise_route_summary(candidate, reference, value_dtype):
    return ast_ops._spmm_coo_pairwise_summary(candidate, reference, value_dtype)



def _format_debug_scalar(value):
    if value is None:
        return "-"
    if torch.is_tensor(value):
        value = value.item()
    if isinstance(value, complex):
        return f"{value.real:.16e}{value.imag:+.16e}j"
    return f"{float(value):.16e}"



def _build_compare_debug_summary(row, reference, route_outputs, cusparse_values, value_dtype):
    if reference is None or reference.numel() == 0:
        return None

    atol, rtol = _tolerance_for_dtype(value_dtype)
    candidates = []
    for label in ("rowrun", "atomic"):
        values = route_outputs.get(label)
        if values is not None:
            candidates.append((label, values))
    if cusparse_values is not None:
        candidates.append(("cusparse", cusparse_values))

    best = None
    for label, candidate in candidates:
        if candidate is None or candidate.shape != reference.shape or candidate.numel() == 0:
            continue
        diff = torch.abs(candidate - reference)
        denom = atol + rtol * torch.abs(reference)
        ratio = diff / denom
        flat_idx = int(torch.argmax(ratio).item())
        error_ratio = float(ratio.reshape(-1)[flat_idx].item())
        if best is None or error_ratio > best["error_ratio"]:
            row_idx = flat_idx // reference.shape[1]
            dense_col = flat_idx % reference.shape[1]
            best = {
                "route": label,
                "row": row_idx,
                "dense_col": dense_col,
                "error_ratio": error_ratio,
            }

    if best is None:
        return None

    row_idx = best["row"]
    dense_col = best["dense_col"]
    row64 = row.to(torch.int64)
    row_nnz = int((row64 == row_idx).sum().item())

    def _scalar_at(values):
        if values is None:
            return None
        return values[row_idx, dense_col]

    return {
        "route": best["route"],
        "row": row_idx,
        "dense_col": dense_col,
        "row_nnz": row_nnz,
        "error_ratio": best["error_ratio"],
        "rowrun": _format_debug_scalar(_scalar_at(route_outputs.get("rowrun"))),
        "atomic": _format_debug_scalar(_scalar_at(route_outputs.get("atomic"))),
        "pt": _format_debug_scalar(reference[row_idx, dense_col]),
        "cu": _format_debug_scalar(_scalar_at(cusparse_values)),
    }
def _assert_spmm_coo_matches_reference(data, row, col, B, shape, value_dtype, out=None, block_n=None, block_nnz=DEFAULT_BLOCK_NNZ):
    result = ast.flagsparse_spmm_coo(
        data,
        row,
        col,
        B,
        shape,
        block_n=block_n,
        block_nnz=block_nnz,
        out=out,
    )
    ref_C, _, _, _ = _build_pytorch_reference(data, row, col, shape, B)
    atol, rtol = _tolerance_for_dtype(value_dtype)
    if not torch.allclose(result, ref_C, atol=atol, rtol=rtol):
        metrics = ast_ops._spmm_validation_metrics(result, ref_C)
        raise AssertionError(
            "reference mismatch: "
            f"err={_scaled_allclose_error(result, ref_C, value_dtype):.3e}, "
            f"max_abs={metrics['max_abs_error']:.3e}, "
            f"atol={atol:.3e}, "
            f"rtol={rtol:.3e}"
        )
    if out is not None and result.data_ptr() != out.data_ptr():
        raise AssertionError("flagsparse_spmm_coo did not return the provided out tensor")
    return result, ref_C



def _build_duplicate_unsorted_case(value_dtype, index_dtype, device, n_dense_cols=DUPLICATE_CASE_DENSE_COLS):
    shape = (4, 6)
    row = torch.tensor([2, 0, 2, 1, 2, 0, 3, 2], dtype=index_dtype, device=device)
    col = torch.tensor([1, 4, 1, 3, 0, 4, 2, 5], dtype=index_dtype, device=device)
    data = _build_values(row.numel(), value_dtype, device)
    B = _build_dense_matrix(shape[1], n_dense_cols, value_dtype, device)
    return data, row, col, B, shape



def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    n_dense_cols=32,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    route="rowrun",
):
    route = _normalize_route(route)
    selected_route = _selected_route(route)
    device = torch.device("cuda")
    data, row, col, shape = load_mtx_to_coo_torch(mtx_path, dtype=value_dtype, device=device)
    row = row.to(index_dtype)
    col = col.to(index_dtype)
    n_rows, n_cols = shape
    nnz = data.numel()
    B = _build_dense_matrix(n_cols, n_dense_cols, value_dtype, device)
    prepared = None
    atol, rtol = _tolerance_for_dtype(value_dtype)

    result = {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz,
        "dense_cols": n_dense_cols,
        "route": selected_route,
        "error": None,
        "triton_ms": None,
        "triton_first_call_ms": None,
        "cusparse_ms": None,
        "pytorch_ms": None,
        "err_pt": None,
        "err_cu": None,
        "triton_abs_err": None,
        "cusparse_abs_err": None,
        "triton_relative_error_diag": None,
        "cusparse_relative_error_diag": None,
        "triton_ok_pt": None,
        "triton_ok_cu": None,
        "cusparse_reason": None,
        "pytorch_reason": None,
        "pytorch_format": None,
        "status": "UNKNOWN",
        "compare": None,
    }

    try:
        prepared = _prepare_canonical_case(data, row, col, shape, B)
        ref_C, pytorch_op, pytorch_format, pytorch_reason = _build_pytorch_reference(data, row, col, shape, B, prepared=prepared)
        result["pytorch_format"] = pytorch_format
        result["pytorch_reason"] = pytorch_reason
    except Exception as exc:
        result["error"] = f"ref: {exc}"
        result["status"] = "REF_FAIL"
        return result

    triton_C = None
    try:
        triton_C, triton_ms, triton_first_call_ms = _benchmark_spmm_coo_route(
            data,
            row,
            col,
            B,
            shape,
            warmup,
            iters,
            route=selected_route,
            block_n=block_n,
            block_nnz=block_nnz,
            prepared=prepared,
        )
        result["triton_ms"] = triton_ms
        result["triton_first_call_ms"] = triton_first_call_ms
    except Exception as exc:
        # Continue to PyTorch / CuPy timing when Triton fails (same as CSR SpMM test).
        result["error"] = f"triton: {exc}"
        result["triton_ok_pt"] = False

    if triton_C is not None:
        triton_summary = _summarize_route_output(triton_C, ref_C, value_dtype)
        result["triton_abs_err"] = triton_summary["max_abs_error"]
        result["triton_relative_error_diag"] = triton_summary["max_relative_error"]
        result["err_pt"] = triton_summary["err_pt"]
        result["triton_ok_pt"] = triton_summary["ok_pt"]
    else:
        result["triton_ok_pt"] = False

    try:
        _, result["pytorch_ms"] = ast_ops._benchmark_cuda_op(
            pytorch_op,
            warmup=warmup,
            iters=iters,
        )
    except Exception as exc:
        reason = str(exc)
        if result["pytorch_reason"]:
            result["pytorch_reason"] = f"{result['pytorch_reason']}; timing: {reason}"
        else:
            result["pytorch_reason"] = reason

    cs_C_t = None
    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    if run_cusparse:
        if value_dtype not in _cupy_supported_dtypes:
            result["cusparse_reason"] = "float16/bfloat16 not supported by CuPy sparse; skipped"
        else:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cpx

                data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(prepared["native_data"]))
                row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(prepared["native_row"].to(torch.int64)))
                col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(prepared["native_col"].to(torch.int64)))
                B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(prepared["native_B"]))
                A_coo = cpx.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)

                torch.cuda.synchronize()
                for _ in range(warmup):
                    _ = A_coo @ B_cp
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    _ = A_coo @ B_cp
                end.record()
                torch.cuda.synchronize()
                result["cusparse_ms"] = start.elapsed_time(end) / iters

                cs_C = A_coo @ B_cp
                cs_C_t = torch.utils.dlpack.from_dlpack(cs_C.toDlpack())
                cusparse_metrics = ast_ops._spmm_validation_metrics(cs_C_t, ref_C)
                result["cusparse_abs_err"] = cusparse_metrics["max_abs_error"]
                result["cusparse_relative_error_diag"] = cusparse_metrics["max_relative_error"]
                if triton_C is not None:
                    result["err_cu"] = _scaled_allclose_error(triton_C, cs_C_t, value_dtype)
                    result["triton_ok_cu"] = torch.allclose(triton_C, cs_C_t, atol=atol, rtol=rtol)
            except Exception as exc:
                result["cusparse_ms"] = None
                result["err_cu"] = None
                result["cusparse_abs_err"] = None
                result["cusparse_relative_error_diag"] = None
                result["triton_ok_cu"] = None
                result["cusparse_reason"] = str(exc)

    if route == "compare":
        route_outputs = {}
        route_summaries = {}
        if triton_C is not None:
            route_outputs[selected_route] = triton_C
            route_summaries[selected_route] = _summarize_route_output(
                triton_C,
                ref_C,
                value_dtype,
                ms=triton_ms,
                first_call_ms=triton_first_call_ms,
                cusparse_values=cs_C_t,
            )
        for extra_route in ("rowrun", "atomic"):
            if extra_route in route_outputs:
                continue
            try:
                extra_C, extra_ms, extra_first_call_ms = _benchmark_spmm_coo_route(
                    data,
                    row,
                    col,
                    B,
                    shape,
                    warmup,
                    iters,
                    route=extra_route,
                    block_n=block_n,
                    block_nnz=block_nnz,
                    prepared=prepared,
                )
                route_outputs[extra_route] = extra_C
                route_summaries[extra_route] = _summarize_route_output(
                    extra_C,
                    ref_C,
                    value_dtype,
                    ms=extra_ms,
                    first_call_ms=extra_first_call_ms,
                    cusparse_values=cs_C_t,
                )
            except Exception as exc:
                route_summaries[extra_route] = {
                    "ms": None,
                    "first_call_ms": None,
                    "ok_pt": False,
                    "err_pt": None,
                    "max_abs_error": None,
                    "max_relative_error": None,
                    "ok_cu": None,
                    "err_cu": None,
                    "error": str(exc),
                }

        parity = {
            "rowrun_vs_atomic": _empty_pairwise_summary(),
        }
        if "rowrun" in route_outputs and "atomic" in route_outputs:
            parity["rowrun_vs_atomic"] = _pairwise_route_summary(route_outputs["rowrun"], route_outputs["atomic"], value_dtype)

        cu_match = None if cs_C_t is None else torch.allclose(cs_C_t, ref_C, atol=atol, rtol=rtol)
        compare_debug = None
        rowrun_summary = route_summaries.get("rowrun") or {}
        atomic_summary = route_summaries.get("atomic") or {}
        if rowrun_summary.get("ok_pt") is False or atomic_summary.get("ok_pt") is False or cu_match is False:
            compare_debug = _build_compare_debug_summary(prepared["canonical_row"], ref_C, route_outputs, cs_C_t, value_dtype)

        result["compare"] = {
            "routes": route_summaries,
            "parity": parity,
            "cusparse_reference_match": cu_match,
            "cusparse_reference_error": (
                None if cs_C_t is None else _scaled_allclose_error(cs_C_t, ref_C, value_dtype)
            ),
            "debug": compare_debug,
        }

    result["status"] = "PASS" if (result["triton_ok_pt"] or result["triton_ok_cu"]) else "FAIL"
    return result



def run_mtx_batch(
    paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    n_dense_cols=32,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    route="rowrun",
    on_result=None,
):
    results = []
    for path in paths:
        entry = run_one_mtx(
            path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=warmup,
            iters=iters,
            run_cusparse=run_cusparse,
            n_dense_cols=n_dense_cols,
            block_n=block_n,
            block_nnz=block_nnz,
            route=route,
        )
        results.append(entry)
        if on_result is not None:
            on_result(entry)
    return results


def _print_spmm_coo_mtx_header(value_dtype, index_dtype, route):
    route = _normalize_route(route)
    print(f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}")
    print(f"Formats: FlagSparse={_route_label(route)}, cuSPARSE=COO dense-mm, PyTorch=COO.")
    print("Timing stays in native dtype. For float32, correctness references use float64 compute then cast.")
    print("PT/CU show per-reference correctness. Err(PT)/Err(CU)=max(|diff| / (atol + rtol*|ref|)).")
    print("PyTorch uses COO sparse.mm as the only correctness reference path.")
    if route == "compare":
        print("Compare mode also benchmarks native atomic (debug-only) after the main table.")
    print("-" * 186)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} "
        f"{'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>13} {'PyTorch(ms)':>11} "
        f"{'FS/CU':>7} {'FS/PT':>7} {'PT':>6} {'CU':>6} {'Err(PT)':>10} {'Err(CU)':>10}"
    )
    print("-" * 186)


def _print_spmm_coo_mtx_row(entry):
    name = os.path.basename(entry["path"])[:27]
    n_rows, n_cols = entry["shape"]
    triton_ms = entry.get("triton_ms")
    cu_ms = entry.get("cusparse_ms")
    pt_ms = entry.get("pytorch_ms")
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {entry['nnz']:>10} {entry['dense_cols']:>8} "
        f"{_fmt_ms(triton_ms):>14} {_fmt_ms(cu_ms):>13} {_fmt_ms(pt_ms):>11} "
        f"{_fmt_speedup(cu_ms, triton_ms):>7} {_fmt_speedup(pt_ms, triton_ms):>7} "
        f"{_fmt_check(entry.get('triton_ok_pt')):>6} {_fmt_check(entry.get('triton_ok_cu')):>6} "
        f"{_fmt_err(entry.get('err_pt')):>10} {_fmt_err(entry.get('err_cu')):>10}"
    )
    err = entry.get("error")
    if err:
        msg = str(err).replace("\n", " ")
        if len(msg) > 200:
            msg = msg[:197] + "..."
        print(f"  NOTE: {msg}")


def print_mtx_results(results, value_dtype, index_dtype, route="rowrun"):
    route = _normalize_route(route)
    _print_spmm_coo_mtx_header(value_dtype, index_dtype, route)
    for entry in results:
        _print_spmm_coo_mtx_row(entry)
    print("-" * 186)



def print_compare_results(results, value_dtype, index_dtype):
    if not any(entry.get("compare") for entry in results):
        return

    print("Compare details (PT-COO / CU-COO / native parity)")
    print("Row/PT is the main default-route diagnostic; Atomic/PT is debug-only.")
    print("-" * 166)
    print(
        f"{'Matrix':<28} {'Row/PT':>7} {'Atomic/PT':>9} {'CU/PT':>7} {'Row/Atomic':>11} "
        f"{'Err(Row/PT)':>12} {'Err(Atomic/PT)':>14} {'Err(CU/PT)':>10} {'Err(Row/Atomic)':>15}"
    )
    print("-" * 166)
    for entry in results:
        compare = entry.get("compare") or {}
        routes = compare.get("routes") or {}
        parity = compare.get("parity") or {}
        rowrun = routes.get("rowrun") or {}
        atomic = routes.get("atomic") or {}
        row_atomic = parity.get("rowrun_vs_atomic") or {}
        print(
            f"{os.path.basename(entry['path'])[:27]:<28} "
            f"{_fmt_check(rowrun.get('ok_pt')):>7} {_fmt_check(atomic.get('ok_pt')):>9} {_fmt_check(compare.get('cusparse_reference_match')):>7} "
            f"{_fmt_check(row_atomic.get('match')):>11} "
            f"{_fmt_err(rowrun.get('err_pt')):>12} {_fmt_err(atomic.get('err_pt')):>14} {_fmt_err(compare.get('cusparse_reference_error')):>10} "
            f"{_fmt_err(row_atomic.get('error_ratio')):>15}"
        )
    print("-" * 166)

    debug_rows = []
    for entry in results:
        compare = entry.get("compare") or {}
        debug = compare.get("debug")
        if debug is not None:
            debug_rows.append((os.path.basename(entry["path"])[:27], debug))
    if not debug_rows:
        return

    print("Worst mismatch summary for failing compare cases")
    print("-" * 178)
    print(
        f"{'Matrix':<28} {'Route':>8} {'Row':>8} {'DenseCol':>9} {'RowNNZ':>8} {'Err':>10} "
        f"{'Rowrun':>18} {'Atomic':>18} {'PT':>18} {'CU':>18}"
    )
    print("-" * 178)
    for name, debug in debug_rows:
        print(
            f"{name:<28} {debug['route']:>8} {debug['row']:>8} {debug['dense_col']:>9} {debug['row_nnz']:>8} {debug['error_ratio']:>10.2e} "
            f"{debug['rowrun']:>18} {debug['atomic']:>18} {debug['pt']:>18} {debug['cu']:>18}"
        )
    print("-" * 178)
def run_all_dtypes_export_csv(
    paths,
    csv_path,
    warmup=10,
    iters=50,
    run_cusparse=True,
    n_dense_cols=32,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    route="rowrun",
):
    route = _normalize_route(route)
    if route == "compare":
        raise ValueError("CSV export only supports route='rowrun' or route='atomic'")
    selected_route = _selected_route(route)
    csv_path = _normalize_csv_path(csv_path)
    rows = []
    for value_dtype in CSV_VALUE_DTYPES:
        for index_dtype in CSV_INDEX_DTYPES:
            print("=" * 150)
            _print_spmm_coo_mtx_header(value_dtype, index_dtype, route)
            results = run_mtx_batch(
                paths,
                value_dtype=value_dtype,
                index_dtype=index_dtype,
                warmup=warmup,
                iters=iters,
                run_cusparse=run_cusparse,
                n_dense_cols=n_dense_cols,
                block_n=block_n,
                block_nnz=block_nnz,
                route=selected_route,
                on_result=_print_spmm_coo_mtx_row,
            )
            print("-" * 186)
            for entry in results:
                n_rows, n_cols = entry["shape"]
                rows.append({
                    "matrix": os.path.basename(entry["path"]),
                    "value_dtype": _dtype_name(value_dtype),
                    "index_dtype": _dtype_name(index_dtype),
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "nnz": entry["nnz"],
                    "triton_ms": entry.get("triton_ms"),
                    "cusparse_ms": entry.get("cusparse_ms"),
                    "pytorch_ms": entry.get("pytorch_ms"),
                    "pt_status": _status_label(entry.get("triton_ok_pt")),
                    "cu_status": _status_label(entry.get("triton_ok_cu")),
                    "status": (
                        "PASS"
                        if (entry.get("triton_ok_pt") or entry.get("triton_ok_cu"))
                        else "FAIL"
                    ),
                    "err_pt": entry.get("err_pt"),
                    "err_cu": entry.get("err_cu"),
                    "error": entry.get("error"),
                })
    fieldnames = [
        "matrix", "value_dtype", "index_dtype", "n_rows", "n_cols", "nnz",
        "triton_ms", "cusparse_ms", "pytorch_ms",
        "pt_status", "cu_status", "status", "err_pt", "err_cu", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    print(f"Wrote {len(rows)} rows to {csv_path}")

def run_api_validation_checks():
    if not torch.cuda.is_available():
        print("API checks skipped: CUDA is not available.")
        return 0

    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    B = torch.randn((2, 4), dtype=torch.float32, device=device)
    dup_data, dup_row, dup_col, dup_B, dup_shape = _build_duplicate_unsorted_case(
        torch.float32, torch.int32, device
    )

    negative_cases = [
        ("shape must be 2D", lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2,)), ValueError),
        ("B must be 2D", lambda: ast.flagsparse_spmm_coo(data, row, col, B[0], (2, 2)), ValueError),
        ("dtype mismatch", lambda: ast.flagsparse_spmm_coo(data, row, col, B.to(torch.float64), (2, 2)), TypeError),
        ("shape mismatch", lambda: ast.flagsparse_spmm_coo(data, row, col, torch.randn((3, 4), dtype=torch.float32, device=device), (2, 2)), ValueError),
        ("row length mismatch", lambda: ast.flagsparse_spmm_coo(data, row[:-1], col, B, (2, 2)), ValueError),
        ("col length mismatch", lambda: ast.flagsparse_spmm_coo(data, row, col[:-1], B, (2, 2)), ValueError),
        ("row out of range", lambda: ast.flagsparse_spmm_coo(data, torch.tensor([0, 2, 1], dtype=torch.int32, device=device), col, B, (2, 2)), IndexError),
        ("col out of range", lambda: ast.flagsparse_spmm_coo(data, row, torch.tensor([0, 3, 1], dtype=torch.int32, device=device), B, (2, 2)), IndexError),
        ("block_n positive", lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2, 2), block_n=0), ValueError),
        ("block_nnz positive", lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2, 2), block_nnz=0), ValueError),
        ("out shape mismatch", lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2, 2), out=torch.empty((3, 4), dtype=torch.float32, device=device)), ValueError),
        ("out device mismatch", lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2, 2), out=torch.empty((2, 4), dtype=torch.float32)), ValueError),
    ]

    failed = 0
    print("-" * 96)
    print("API validation checks")
    print("-" * 96)
    for name, fn, exc_type in negative_cases:
        try:
            fn()
            print(f"FAIL  {name:<32} expected {exc_type.__name__}")
            failed += 1
        except exc_type:
            print(f"PASS  {name:<32} raised {exc_type.__name__}")
        except Exception as exc:
            print(f"FAIL  {name:<32} raised {type(exc).__name__}: {exc}")
            failed += 1

    positive_checks = []

    def _positive_out_path():
        out = torch.empty((2, 4), dtype=torch.float32, device=device)
        _assert_spmm_coo_matches_reference(data, row, col, B, (2, 2), torch.float32, out=out)

    positive_checks.append(("out path success", _positive_out_path))

    def _positive_empty_matrix():
        empty_data = torch.tensor([], dtype=torch.float32, device=device)
        empty_row = torch.tensor([], dtype=torch.int32, device=device)
        empty_col = torch.tensor([], dtype=torch.int32, device=device)
        dense = torch.randn((2, 4), dtype=torch.float32, device=device)
        result, _ = _assert_spmm_coo_matches_reference(
            empty_data,
            empty_row,
            empty_col,
            dense,
            (2, 2),
            torch.float32,
        )
        if result.shape != (2, 4):
            raise AssertionError(f"unexpected empty-matrix result shape: {tuple(result.shape)}")

    positive_checks.append(("empty matrix success", _positive_empty_matrix))

    def _positive_empty_dense_cols():
        dense = torch.empty((2, 0), dtype=torch.float32, device=device)
        result, _ = _assert_spmm_coo_matches_reference(
            data,
            row,
            col,
            dense,
            (2, 2),
            torch.float32,
        )
        if result.shape != (2, 0):
            raise AssertionError(f"unexpected empty-dense result shape: {tuple(result.shape)}")

    positive_checks.append(("empty dense cols success", _positive_empty_dense_cols))

    def _positive_noncontiguous_b():
        dense = _build_dense_matrix(4, 2, torch.float32, device).transpose(0, 1)
        if dense.is_contiguous():
            raise AssertionError("expected non-contiguous test matrix")
        _assert_spmm_coo_matches_reference(data, row, col, dense, (2, 2), torch.float32)

    positive_checks.append(("noncontiguous B success", _positive_noncontiguous_b))

    def _positive_unsorted_duplicate():
        _assert_spmm_coo_matches_reference(
            dup_data,
            dup_row,
            dup_col,
            dup_B,
            dup_shape,
            torch.float32,
        )

    positive_checks.append(("unsorted duplicate success", _positive_unsorted_duplicate))

    for name, fn in positive_checks:
        try:
            fn()
            print(f"PASS  {name:<32} returned correct result")
        except Exception as exc:
            print(f"FAIL  {name:<32} raised {type(exc).__name__}: {exc}")
            failed += 1

    print("-" * 96)
    return failed

def run_coo_tile_branch_coverage(warmup=WARMUP, iters=ITERS, run_cusparse=True):
    if not torch.cuda.is_available():
        print("COO branch coverage skipped: CUDA is not available.")
        return 0

    print("=" * 144)
    print("COO native row-run dense-column coverage")
    print("=" * 144)
    print(
        f"{'DenseN':>8} {'BLOCK_N':>8} {'NNZTile':>8} {'Runs':>7} {'Tiles':>7} {'Warp':>6} {'Factor':>7} "
        f"{'PyTorch(ms)':>12} {'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>12} {'PT':>6} {'CU':>6} {'Err(FS)':>11}"
    )
    print("-" * 144)

    failed = 0
    note = None
    for n_rows, n_cols, nnz, n_dense_cols in COO_TILE_CASES:
        result = ast_ops.benchmark_spmm_coo_case(
            n_rows=n_rows,
            n_cols=n_cols,
            nnz=nnz,
            n_dense_cols=n_dense_cols,
            value_dtype=torch.float32,
            index_dtype=torch.int32,
            warmup=warmup,
            iters=iters,
            block_n=DEFAULT_BLOCK_N,
            block_nnz=DEFAULT_BLOCK_NNZ,
            run_cusparse=run_cusparse,
        )
        params = result["parameters"]
        perf = result["performance"]
        verify = result["verification"]
        backend = result["backend_status"]
        samples = result["samples"]
        triton_ok = verify.get("triton_strict_allclose_match", verify.get("triton_match_reference"))
        cusparse_ok = verify.get("cusparse_strict_allclose_match", verify.get("cusparse_match_reference"))
        status = "PASS" if triton_ok else "FAIL"
        if status != "PASS":
            failed += 1
        if backend.get("cusparse_unavailable_reason"):
            note = backend["cusparse_unavailable_reason"]
        triton_err = _scaled_allclose_error(samples["triton"], samples["reference"], torch.float32)
        print(
            f"{n_dense_cols:>8} {params['block_n']:>8} {params['block_nnz']:>8} {params['n_row_runs']:>7} {params['required_nnz_tiles']:>7} {params['heuristic_warp_size']:>6} {params['heuristic_factor']:>7} "
            f"{_fmt_ms(perf.get('pytorch_ms')):>12} {_fmt_ms(perf.get('triton_ms')):>14} {_fmt_ms(perf.get('cusparse_ms')):>12} "
            f"{_fmt_check(triton_ok):>6} {_fmt_check(cusparse_ok):>6} {_fmt_err(triton_err):>11}"
        )
    print("-" * 144)
    if note:
        print(f"cuSPARSE note: {note}")
    print()
    return failed



def _print_synthetic_compare_results(compare_rows):
    if not compare_rows:
        return

    print("Compare details (PT-COO / CU-COO / native parity)")
    print("Row/PT is the main default-route diagnostic; Atomic/PT is debug-only.")
    print("-" * 160)
    print(
        f"{'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} {'Row/PT':>7} {'Atomic/PT':>9} {'CU/PT':>7} {'Row/Atomic':>11} "
        f"{'Err(Row/PT)':>12} {'Err(Atomic/PT)':>14} {'Err(CU/PT)':>10} {'Err(Row/Atomic)':>15}"
    )
    print("-" * 160)
    for entry in compare_rows:
        print(
            f"{entry['n_rows']:>7} {entry['n_cols']:>7} {entry['nnz']:>10} {entry['dense_cols']:>8} "
            f"{_fmt_check(entry.get('row_pt')):>7} {_fmt_check(entry.get('atomic_pt')):>9} {_fmt_check(entry.get('cu_pt')):>7} "
            f"{_fmt_check(entry.get('row_atomic')):>11} "
            f"{_fmt_err(entry.get('err_row_pt')):>12} {_fmt_err(entry.get('err_atomic_pt')):>14} {_fmt_err(entry.get('err_cu_pt')):>10} "
            f"{_fmt_err(entry.get('err_row_atomic')):>15}"
        )
    print("-" * 160)
    print()
def run_comprehensive_synthetic(
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    run_api_checks=True,
    run_coo_coverage=True,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    route="rowrun",
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    route = _normalize_route(route)
    selected_route = _selected_route(route)

    print("=" * 150)
    print("FLAGSPARSE SpMM BENCHMARK (synthetic COO @ dense)")
    print("=" * 150)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  |  Warmup: {warmup}  Iters: {iters}  "
        f"BLOCK_N: {_fmt_launch_value(block_n)}  BLOCK_NNZ: {_fmt_launch_value(block_nnz)}  Route: {route}"
    )
    print(f"Formats: FlagSparse={_route_label(route)}, cuSPARSE=COO dense-mm (when supported), PyTorch=COO.")
    print("For float32, PT checks the float64-based correctness reference while CU reflects native cuSPARSE float32 consistency.")
    if route == "compare":
        print("Compare mode also benchmarks native atomic (debug-only) for each synthetic case.")
    print()

    total = 0
    failed = 0
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            compare_rows = []
            print("-" * 150)
            print(f"Value dtype: {_dtype_name(value_dtype):<12}  |  Index dtype: {_dtype_name(index_dtype):<6}")
            print("-" * 150)
            print(
                f"{'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} {'BN':>4} {'BNNZ':>6} {'Runs':>5} {'Tiles':>5} "
                f"{'PyTorch(ms)':>12} {'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>12} {'FS/PT':>8} {'FS/CU':>8} {'PT':>6} {'CU':>6} {'Err(FS)':>11} {'Err(CU)':>12}"
            )
            print("-" * 150)
            combo_reason = None
            for n_rows, n_cols, nnz, n_dense_cols in TEST_CASES:
                result = ast_ops.benchmark_spmm_coo_case(
                    n_rows=n_rows,
                    n_cols=n_cols,
                    nnz=nnz,
                    n_dense_cols=n_dense_cols,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=warmup,
                    iters=iters,
                    block_n=block_n,
                    block_nnz=block_nnz,
                    run_cusparse=run_cusparse,
                    route=selected_route,
                    compare_routes=(route == "compare"),
                )
                total += 1
                params = result["parameters"]
                perf = result["performance"]
                verify = result["verification"]
                backend = result["backend_status"]
                samples = result["samples"]
                triton_ok = verify.get("triton_strict_allclose_match", verify.get("triton_match_reference"))
                cusparse_ok = verify.get("cusparse_strict_allclose_match", verify.get("cusparse_match_reference"))
                status = "PASS" if triton_ok else "FAIL"
                if status != "PASS":
                    failed += 1
                if backend.get("cusparse_unavailable_reason"):
                    combo_reason = backend["cusparse_unavailable_reason"]
                triton_err = _scaled_allclose_error(samples["triton"], samples["reference"], value_dtype)
                cusparse_err = None
                if samples.get("cusparse") is not None:
                    cusparse_err = _scaled_allclose_error(samples["triton"], samples["cusparse"], value_dtype)
                print(
                    f"{n_rows:>7} {n_cols:>7} {nnz:>10} {n_dense_cols:>8} {params['block_n']:>4} {params['block_nnz']:>6} {params['n_row_runs']:>5} {params['required_nnz_tiles']:>5} "
                    f"{_fmt_ms(perf.get('pytorch_ms')):>12} {_fmt_ms(perf.get('triton_ms')):>14} {_fmt_ms(perf.get('cusparse_ms')):>12} "
                    f"{_fmt_speedup(perf.get('pytorch_ms'), perf.get('triton_ms')):>8} {_fmt_speedup(perf.get('cusparse_ms'), perf.get('triton_ms')):>8} "
                    f"{_fmt_check(triton_ok):>6} {_fmt_check(cusparse_ok):>6} {_fmt_err(triton_err):>11} {_fmt_err(cusparse_err):>12}"
                )
                if route == "compare":
                    route_results = result.get("route_results") or {}
                    parity = result.get("parity") or {}
                    compare_rows.append({
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "nnz": nnz,
                        "dense_cols": n_dense_cols,
                        "row_pt": (route_results.get("rowrun") or {}).get("match_reference"),
                        "atomic_pt": (route_results.get("atomic") or {}).get("match_reference"),
                        "cu_pt": verify.get("cusparse_match_reference"),
                        "row_atomic": (parity.get("rowrun_vs_atomic") or {}).get("match"),
                        "err_row_pt": (route_results.get("rowrun") or {}).get("error_ratio"),
                        "err_atomic_pt": (route_results.get("atomic") or {}).get("error_ratio"),
                        "err_cu_pt": (verify.get("cusparse_max_relative_error") if verify.get("cusparse_match_reference") is not None else None),
                        "err_row_atomic": (parity.get("rowrun_vs_atomic") or {}).get("error_ratio"),
                    })
            print("-" * 150)
            if combo_reason:
                print(f"  cuSPARSE: {combo_reason}")
            print()
            if route == "compare":
                _print_synthetic_compare_results(compare_rows)

    coo_failed = 0
    if run_coo_coverage:
        if route == "rowrun":
            coo_failed = run_coo_tile_branch_coverage(warmup=warmup, iters=iters, run_cusparse=run_cusparse)
        else:
            print(f"COO dense-column coverage is row-run specific; skipped for route {route}.")
            print()
    api_failed = run_api_validation_checks() if run_api_checks else 0
    print("=" * 150)
    print(
        f"Total synthetic cases: {total}  Failed synthetic cases: {failed}  "
        f"Failed COO branch cases: {coo_failed}  Failed API checks: {api_failed}"
    )
    print("=" * 150)


def main():
    parser = argparse.ArgumentParser(description="COO SpMM test: SuiteSparse .mtx batch run, error and performance.")
    parser.add_argument("mtx", nargs="*", help=".mtx file path(s), or directory(ies) to glob for *.mtx")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic benchmark instead of .mtx")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32", "float64", "complex64", "complex128"], help="Value dtype (default: float32)")
    parser.add_argument("--index-dtype", default="int32", choices=["int32", "int64"], help="Index dtype (default: int32)")
    parser.add_argument("--dense-cols", type=int, default=32, help="Dense RHS column count")
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N, help="Output column tile override (default: auto from dense-column heuristic)")
    parser.add_argument("--block-nnz", type=int, default=DEFAULT_BLOCK_NNZ, help="COO nnz tile width override (default: 256)")
    parser.add_argument("--route", default="rowrun", choices=["rowrun", "atomic", "compare"], help="Native COO route to benchmark/test (default: rowrun)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--no-cusparse", action="store_true", help="Skip cuSPARSE baseline")
    parser.add_argument("--skip-api-checks", action="store_true", help="Skip API validation checks in synthetic mode")
    parser.add_argument("--skip-coo-coverage", action="store_true", help="Skip dense-column COO heuristic coverage in synthetic mode")
    parser.add_argument("--csv", type=str, default=None, metavar="FILE", help="Run float32/float64 with int32 indices on all .mtx and write results to one CSV")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    index_map = {"int32": torch.int32, "int64": torch.int64}
    value_dtype = dtype_map[args.dtype]
    index_dtype = index_map[args.index_dtype]

    if args.synthetic:
        run_comprehensive_synthetic(
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            run_api_checks=not args.skip_api_checks,
            run_coo_coverage=not args.skip_coo_coverage,
            block_n=args.block_n,
            block_nnz=args.block_nnz,
            route=args.route,
        )
        return

    paths = []
    for path in args.mtx:
        if os.path.isfile(path) and path.endswith(".mtx"):
            paths.append(path)
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))

    if not paths and not args.csv:
        print("No .mtx files given. Use: python test_spmm_coo.py <file.mtx> [file2.mtx ...] or <dir/>")
        print("Or run synthetic: python test_spmm_coo.py --synthetic")
        print("Or run all dtypes and export CSV: python test_spmm_coo.py <dir/> --csv results.csv")
        return

    if args.csv is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        if args.route == "compare":
            print("CSV export only supports --route rowrun or --route atomic.")
            return
        csv_path = _normalize_csv_path(args.csv)
        print("=" * 100)
        print("FLAGSPARSE COO SpMM - f32/f64 with int32, export to CSV")
        print("=" * 100)
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  DenseN: {args.dense_cols}  |  Route: {args.route}  |  CSV: {csv_path}")
        if args.dtype != "float32" or args.index_dtype != "int32":
            print("Note: --csv export ignores --dtype/--index-dtype and always writes float32/float64 with int32 indices.")
        run_all_dtypes_export_csv(
            paths,
            csv_path,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            n_dense_cols=args.dense_cols,
            block_n=args.block_n,
            block_nnz=args.block_nnz,
            route=args.route,
        )
        return

    print("=" * 140)
    print("FLAGSPARSE COO SpMM - SuiteSparse .mtx batch (error + performance)")
    print("=" * 140)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(
        f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  dense_cols: {args.dense_cols}  "
        f"warmup: {args.warmup}  iters: {args.iters}  block_n: {_fmt_launch_value(args.block_n)}  "
        f"block_nnz: {_fmt_launch_value(args.block_nnz)}  route: {args.route}"
    )
    print()
    results = run_mtx_batch(
        paths,
        value_dtype=value_dtype,
        index_dtype=index_dtype,
        warmup=args.warmup,
        iters=args.iters,
        run_cusparse=not args.no_cusparse,
        n_dense_cols=args.dense_cols,
        block_n=args.block_n,
        block_nnz=args.block_nnz,
        route=args.route,
    )
    print_mtx_results(results, value_dtype, index_dtype, route=args.route)
    if args.route == "compare":
        print_compare_results(results, value_dtype, index_dtype)


if __name__ == "__main__":
    main()