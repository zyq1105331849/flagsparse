"""
SpGEMM tests: load SuiteSparse .mtx, run CSR SpGEMM(A@B), and report
error/performance in a SpMM-like table and CSV format.
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
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import flagsparse as ast
import flagsparse.sparse_operations.spgemm_csr as ast_ops
from test_spmm import load_mtx_to_csr_torch

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
CSV_VALUE_DTYPES = [torch.float32, torch.float64]
CSV_INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 50
DEFAULT_INPUT_MODE = "auto"
TARGET_TIMED_WINDOW_SECONDS = 8.0


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


def _append_error(current, message):
    msg = str(message)
    if not current:
        return msg
    return f"{current}; {msg}"


def _allclose_error_ratio(actual, reference, atol, rtol):
    if actual.numel() == 0:
        return 0.0
    diff = torch.abs(actual - reference).to(torch.float64)
    tol = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / tol).item())


def _spgemm_compare_metrics(candidate, reference, value_dtype):
    c_data, c_indices, c_indptr, c_shape = candidate
    r_data, r_indices, r_indptr, r_shape = reference
    if c_shape != r_shape:
        return {
            "pattern_ok": False,
            "pass": False,
            "err_ratio": float("inf"),
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "reason": f"shape mismatch {c_shape} vs {r_shape}",
        }

    c_keys, c_vals = ast_ops._csr_to_sorted_pairs(c_data, c_indices, c_indptr, c_shape[1])
    r_keys, r_vals = ast_ops._csr_to_sorted_pairs(r_data, r_indices, r_indptr, r_shape[1])
    if c_keys.numel() != r_keys.numel():
        return {
            "pattern_ok": False,
            "pass": False,
            "err_ratio": float("inf"),
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "reason": f"nnz mismatch {c_keys.numel()} vs {r_keys.numel()}",
        }
    if c_keys.numel() > 0 and not torch.equal(c_keys, r_keys):
        return {
            "pattern_ok": False,
            "pass": False,
            "err_ratio": float("inf"),
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "reason": "sparsity pattern mismatch",
        }

    if c_vals.numel() == 0:
        return {
            "pattern_ok": True,
            "pass": True,
            "err_ratio": 0.0,
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "reason": "ok",
        }

    atol, rtol = ast_ops._tolerance_for_dtype(value_dtype)
    err_ratio = _allclose_error_ratio(c_vals, r_vals, atol, rtol)
    abs_diff = torch.abs(c_vals - r_vals)
    max_abs = float(torch.max(abs_diff).item())
    ref_max = float(torch.max(torch.abs(r_vals)).item())
    max_rel = 0.0 if ref_max == 0.0 else max_abs / ref_max
    ok = (not torch.isnan(torch.tensor(err_ratio)).item()) and err_ratio <= 1.0
    return {
        "pattern_ok": True,
        "pass": bool(ok),
        "err_ratio": err_ratio,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "reason": "ok" if ok else "value mismatch",
    }


def _classify_reference_reason(*messages):
    merged = " ".join(str(m).lower() for m in messages if m)
    if not merged:
        return "REF_UNAVAILABLE"
    if "out of memory" in merged:
        return "REF_OOM"
    if (
        "insufficient resources" in merged
        or "cusparsespgemm_workestimation" in merged
        or "cusparsespgemm_compute" in merged
        or "resource" in merged
    ):
        return "REF_RESOURCE"
    return "REF_UNAVAILABLE"


def _normalize_csv_path(csv_path):
    csv_path = str(csv_path)
    if not csv_path.lower().endswith(".csv"):
        csv_path = f"{csv_path}.csv"
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return csv_path


def _log_stage(path, stage, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"[SpGEMM][{os.path.basename(path)}] {stage} (elapsed={elapsed:.2f}s)", flush=True)


def _resolve_input_mode(requested_mode, shape):
    n_rows, n_cols = shape
    if requested_mode == "a_equals_b":
        if n_rows != n_cols:
            raise ValueError(
                f"input_mode=a_equals_b requires square matrix, got {n_rows}x{n_cols}"
            )
        return "A_EQUALS_B"
    if requested_mode == "a_at":
        return "A_AT"
    if requested_mode == "auto":
        return "A_EQUALS_B" if n_rows == n_cols else "A_AT"
    raise ValueError(f"unsupported input_mode: {requested_mode}")


def _build_spgemm_rhs(a_data, a_indices, a_indptr, a_shape, mode):
    if mode == "A_EQUALS_B":
        return a_data, a_indices, a_indptr, a_shape
    if mode != "A_AT":
        raise ValueError(f"unsupported resolved mode: {mode}")
    a_t = ast_ops._to_torch_csr(a_data, a_indices, a_indptr, a_shape)
    # CSR^T may materialize as CSC; convert through COO so downstream always receives CSR.
    b_t = a_t.transpose(0, 1).to_sparse_coo().coalesce()
    return ast_ops._torch_sparse_to_csr(b_t)


def _build_torch_spgemm_reference(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    a_csr = torch.sparse_csr_tensor(
        a_indptr.to(torch.int64),
        a_indices.to(torch.int64),
        a_data,
        size=a_shape,
        device=a_data.device,
    )
    b_csr = torch.sparse_csr_tensor(
        b_indptr.to(torch.int64),
        b_indices.to(torch.int64),
        b_data,
        size=b_shape,
        device=b_data.device,
    )
    ref_format = "CSR"
    try:
        op = lambda: torch.sparse.mm(a_csr, b_csr)
        ref_sparse = op()
    except Exception:
        ref_format = "COO"
        a_coo = a_csr.to_sparse_coo().coalesce()
        b_coo = b_csr.to_sparse_coo().coalesce()
        op = lambda: torch.sparse.mm(a_coo, b_coo)
        ref_sparse = op()
    if ref_sparse.layout not in (torch.sparse_coo, torch.sparse_csr):
        raise RuntimeError(f"Unexpected torch sparse.mm result layout: {ref_sparse.layout}")
    return ast_ops._torch_sparse_to_csr(ref_sparse), ref_format, op


def _pick_effective_benchmark_loops(warmup, iters, first_call_ms, target_window_seconds):
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))
    per_call_s = max(float(first_call_ms) / 1000.0, 1e-4)
    target_iters = max(1, int(float(target_window_seconds) / per_call_s))
    eff_iters = min(iters, target_iters)
    warmup_cap = max(1, eff_iters // 2)
    eff_warmup = min(warmup, warmup_cap)
    return eff_warmup, eff_iters


def _benchmark_flagsparse_spgemm(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    warmup,
    iters,
    adaptive_loops,
    target_window_seconds,
    start_time,
    mtx_path,
):
    _log_stage(mtx_path, "prepare", start_time)
    torch.cuda.synchronize()
    t_prepare0 = time.perf_counter()
    prepared = ast.prepare_spgemm_csr(
        a_data, a_indices, a_indptr, a_shape,
        b_data, b_indices, b_indptr, b_shape,
    )
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0

    _log_stage(mtx_path, "first-call", start_time)
    torch.cuda.synchronize()
    t_first0 = time.perf_counter()
    first_result, first_meta = ast.flagsparse_spgemm_csr(prepared=prepared, return_meta=True)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t_first0) * 1000.0
    first_meta = dict(first_meta)
    first_meta["prepare_ms"] = prepare_ms

    if adaptive_loops:
        eff_warmup, eff_iters = _pick_effective_benchmark_loops(
            warmup=warmup,
            iters=iters,
            first_call_ms=first_call_ms,
            target_window_seconds=target_window_seconds,
        )
    else:
        eff_warmup = max(0, int(warmup))
        eff_iters = max(1, int(iters))
    first_meta["effective_warmup"] = eff_warmup
    first_meta["effective_iters"] = eff_iters

    _log_stage(mtx_path, f"timed-run warmup={eff_warmup} iters={eff_iters}", start_time)
    triton_result, triton_ms = ast_ops._benchmark_cuda_op(
        lambda: ast.flagsparse_spgemm_csr(prepared=prepared),
        warmup=eff_warmup,
        iters=eff_iters,
    )
    return triton_result, triton_ms, first_call_ms, first_meta


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    input_mode=DEFAULT_INPUT_MODE,
    adaptive_loops=False,
    target_window_seconds=TARGET_TIMED_WINDOW_SECONDS,
):
    case_start = time.perf_counter()
    device = torch.device("cuda")

    _log_stage(mtx_path, "load", case_start)
    a_data, a_indices, a_indptr, a_shape = load_mtx_to_csr_torch(
        mtx_path, dtype=value_dtype, device=device
    )
    a_indices = a_indices.to(index_dtype)
    resolved_mode = _resolve_input_mode(input_mode, a_shape)
    b_data, b_indices, b_indptr, b_shape = _build_spgemm_rhs(
        a_data, a_indices, a_indptr, a_shape, resolved_mode
    )
    b_indices = b_indices.to(index_dtype)

    nnz_a = int(a_data.numel())
    nnz_b = int(b_data.numel())
    print(
        f"[SpGEMM][{os.path.basename(mtx_path)}] preflight mode={resolved_mode} "
        f"shape_a={a_shape} shape_b={b_shape} nnz_a={nnz_a} nnz_b={nnz_b}",
        flush=True,
    )
    result = {
        "path": mtx_path,
        "shape": a_shape,
        "shape_a": a_shape,
        "shape_b": b_shape,
        "nnz": nnz_a,
        "nnz_a": nnz_a,
        "nnz_b": nnz_b,
        "nnz_c": None,
        "input_mode": resolved_mode,
        "error": None,
        "triton_started": False,
        "ref_started": False,
        "ref_reason_code": None,
        "triton_ms": None,
        "triton_first_call_ms": None,
        "prepare_ms": None,
        "count_ms": None,
        "fill_ms": None,
        "effective_warmup": None,
        "effective_iters": None,
        "pytorch_ms": None,
        "cusparse_ms": None,
        "err_pt": None,
        "err_cu": None,
        "max_abs_err_pt": None,
        "max_rel_err_pt": None,
        "max_abs_err_cu": None,
        "max_rel_err_cu": None,
        "triton_ok_pt": None,
        "triton_ok_cu": None,
        "pytorch_reason": None,
        "cusparse_reason": None,
        "pytorch_format": None,
        "status": "UNKNOWN",
    }

    triton_result = None
    try:
        result["triton_started"] = True
        triton_result, triton_ms, triton_first_ms, meta = _benchmark_flagsparse_spgemm(
            a_data,
            a_indices,
            a_indptr,
            a_shape,
            b_data,
            b_indices,
            b_indptr,
            b_shape,
            warmup=warmup,
            iters=iters,
            adaptive_loops=adaptive_loops,
            target_window_seconds=target_window_seconds,
            start_time=case_start,
            mtx_path=mtx_path,
        )
        result["triton_ms"] = triton_ms
        result["triton_first_call_ms"] = triton_first_ms
        result["prepare_ms"] = meta.get("prepare_ms")
        result["count_ms"] = meta.get("count_ms")
        result["fill_ms"] = meta.get("fill_ms")
        result["effective_warmup"] = meta.get("effective_warmup")
        result["effective_iters"] = meta.get("effective_iters")
        result["nnz_c"] = int(triton_result[0].numel()) if triton_result is not None else None
    except Exception as exc:
        result["error"] = _append_error(result["error"], f"triton: {exc}")

    _log_stage(mtx_path, "reference", case_start)
    result["ref_started"] = True
    ref_result = None
    pt_compared = False
    cu_compared = False
    ref_warmup = result["effective_warmup"] if result["effective_warmup"] is not None else warmup
    ref_iters = result["effective_iters"] if result["effective_iters"] is not None else iters
    ref_warmup = max(0, int(ref_warmup))
    ref_iters = max(1, int(ref_iters))

    try:
        ref_result, ref_format, ref_op = _build_torch_spgemm_reference(
            a_data,
            a_indices,
            a_indptr,
            a_shape,
            b_data,
            b_indices,
            b_indptr,
            b_shape,
        )
        result["pytorch_format"] = ref_format
        _, result["pytorch_ms"] = ast_ops._benchmark_cuda_op(
            ref_op,
            warmup=ref_warmup,
            iters=ref_iters,
        )
        if triton_result is not None:
            pt_metrics = _spgemm_compare_metrics(triton_result, ref_result, value_dtype)
            result["triton_ok_pt"] = pt_metrics["pass"]
            result["err_pt"] = pt_metrics["err_ratio"]
            result["max_abs_err_pt"] = pt_metrics["max_abs_error"]
            result["max_rel_err_pt"] = pt_metrics["max_relative_error"]
            if not pt_metrics["pattern_ok"]:
                result["error"] = _append_error(result["error"], f"pt_ref: {pt_metrics['reason']}")
            pt_compared = True
    except Exception as exc:
        result["pytorch_reason"] = str(exc)
        result["error"] = _append_error(result["error"], f"pt_ref: {exc}")

    if run_cusparse:
        if ast_ops.cp is None or ast_ops.cpx_sparse is None:
            result["cusparse_reason"] = "CuPy/cuSPARSE is not available"
        else:
            try:
                a_cp = ast_ops.cpx_sparse.csr_matrix(
                    (
                        ast_ops._cupy_from_torch(a_data),
                        ast_ops._cupy_from_torch(a_indices.to(torch.int64)),
                        ast_ops._cupy_from_torch(a_indptr.to(torch.int64)),
                    ),
                    shape=a_shape,
                )
                b_cp = ast_ops.cpx_sparse.csr_matrix(
                    (
                        ast_ops._cupy_from_torch(b_data),
                        ast_ops._cupy_from_torch(b_indices.to(torch.int64)),
                        ast_ops._cupy_from_torch(b_indptr.to(torch.int64)),
                    ),
                    shape=b_shape,
                )
                ref_warmup = result["effective_warmup"] if result["effective_warmup"] is not None else warmup
                ref_iters = result["effective_iters"] if result["effective_iters"] is not None else iters
                ref_warmup = max(0, int(ref_warmup))
                ref_iters = max(1, int(ref_iters))
                c_cp, result["cusparse_ms"] = ast_ops._benchmark_cuda_op(
                    lambda: a_cp @ b_cp,
                    warmup=ref_warmup,
                    iters=ref_iters,
                )
                c_coo = c_cp.tocoo()
                rows = ast_ops._torch_from_cupy(c_coo.row).to(torch.int64)
                cols = ast_ops._torch_from_cupy(c_coo.col).to(torch.int64)
                vals = ast_ops._torch_from_cupy(c_coo.data).to(value_dtype)
                c_t = torch.sparse_coo_tensor(
                    torch.stack([rows, cols]), vals, (a_shape[0], b_shape[1]), device=device
                ).coalesce()
                c_ref = ast_ops._torch_sparse_to_csr(c_t)
                if triton_result is not None:
                    cu_metrics = _spgemm_compare_metrics(triton_result, c_ref, value_dtype)
                    result["triton_ok_cu"] = cu_metrics["pass"]
                    result["err_cu"] = cu_metrics["err_ratio"]
                    result["max_abs_err_cu"] = cu_metrics["max_abs_error"]
                    result["max_rel_err_cu"] = cu_metrics["max_relative_error"]
                    if not cu_metrics["pattern_ok"]:
                        result["error"] = _append_error(result["error"], f"cu_ref: {cu_metrics['reason']}")
                    cu_compared = True
            except Exception as exc:
                result["cusparse_reason"] = str(exc)
                result["error"] = _append_error(result["error"], f"cu_ref: {exc}")

    _log_stage(mtx_path, "compare", case_start)
    if triton_result is None:
        result["status"] = "FAIL"
        result["ref_reason_code"] = _classify_reference_reason(
            result.get("pytorch_reason"),
            result.get("cusparse_reason"),
            result.get("error"),
        )
        return result

    if pt_compared or cu_compared:
        result["status"] = "PASS" if (result["triton_ok_pt"] or result["triton_ok_cu"]) else "FAIL"
    else:
        ref_code = _classify_reference_reason(
            result.get("pytorch_reason"),
            result.get("cusparse_reason"),
        )
        result["ref_reason_code"] = ref_code
        result["status"] = ref_code
    return result


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    input_mode=DEFAULT_INPUT_MODE,
    adaptive_loops=False,
    target_window_seconds=TARGET_TIMED_WINDOW_SECONDS,
    on_result=None,
):
    results = []
    total = len(mtx_paths)
    for idx, path in enumerate(mtx_paths, start=1):
        print(f"[SpGEMM] ({idx}/{total}) {path}", flush=True)
        entry = run_one_mtx(
            path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=warmup,
            iters=iters,
            run_cusparse=run_cusparse,
            input_mode=input_mode,
            adaptive_loops=adaptive_loops,
            target_window_seconds=target_window_seconds,
        )
        results.append(entry)
        if on_result is not None:
            on_result(entry)
    return results


def _print_spgemm_mtx_header(value_dtype, index_dtype):
    print(f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}")
    print("Formats: FlagSparse=CSR SpGEMM(A@B), cuSPARSE=CSR@CSR, PyTorch=sparse.mm.")
    print("Err(PT/CU)=max(|diff|/(atol+rtol*|ref|)); MaxRel=max(|diff|)/max(|ref|).")
    print("-" * 320)
    print(
        f"{'Matrix':<28} {'Mode':<10} {'A_rows':>7} {'A_cols':>7} {'B_cols':>7} {'NNZ_A':>10} {'NNZ_B':>10} {'NNZ_C':>10} "
        f"{'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>13} {'PyTorch(ms)':>11} "
        f"{'FS/CU':>7} {'FS/PT':>7} {'PT':>6} {'CU':>6} {'Status':>13} {'RefCode':>14} "
        f"{'Err(PT)':>10} {'Err(CU)':>10} {'MaxAbs(PT)':>12} {'MaxRel(PT)':>12} {'MaxAbs(CU)':>12} {'MaxRel(CU)':>12} "
        f"{'Prep(ms)':>9} {'Count(ms)':>10} {'Fill(ms)':>9}"
    )
    print("-" * 320)


def _print_spgemm_mtx_row(entry):
    name = os.path.basename(entry["path"])[:27]
    a_rows, a_cols = entry["shape_a"]
    b_cols = entry["shape_b"][1]
    print(
        f"{name:<28} {entry.get('input_mode', 'N/A'):<10} {a_rows:>7} {a_cols:>7} {b_cols:>7} "
        f"{entry['nnz_a']:>10} {entry['nnz_b']:>10} {str(entry['nnz_c'] if entry['nnz_c'] is not None else 'N/A'):>10} "
        f"{_fmt_ms(entry.get('triton_ms')):>14} {_fmt_ms(entry.get('cusparse_ms')):>13} {_fmt_ms(entry.get('pytorch_ms')):>11} "
        f"{_fmt_speedup(entry.get('cusparse_ms'), entry.get('triton_ms')):>7} {_fmt_speedup(entry.get('pytorch_ms'), entry.get('triton_ms')):>7} "
        f"{_fmt_check(entry.get('triton_ok_pt')):>6} {_fmt_check(entry.get('triton_ok_cu')):>6} {entry.get('status', 'N/A'):>13} {str(entry.get('ref_reason_code') or 'N/A'):>14} "
        f"{_fmt_err(entry.get('err_pt')):>10} {_fmt_err(entry.get('err_cu')):>10} "
        f"{_fmt_err(entry.get('max_abs_err_pt')):>12} {_fmt_err(entry.get('max_rel_err_pt')):>12} {_fmt_err(entry.get('max_abs_err_cu')):>12} {_fmt_err(entry.get('max_rel_err_cu')):>12} "
        f"{_fmt_ms(entry.get('prepare_ms')):>9} {_fmt_ms(entry.get('count_ms')):>10} {_fmt_ms(entry.get('fill_ms')):>9}"
    )
    err = entry.get("error")
    if err:
        msg = str(err).replace("\n", " ")
        if len(msg) > 320:
            msg = msg[:317] + "..."
        print(f"  NOTE: {msg}")


def print_mtx_results(results, value_dtype, index_dtype):
    _print_spgemm_mtx_header(value_dtype, index_dtype)
    for entry in results:
        _print_spgemm_mtx_row(entry)
    print("-" * 320)


def run_all_dtypes_export_csv(
    paths,
    csv_path,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    input_mode=DEFAULT_INPUT_MODE,
    adaptive_loops=False,
    target_window_seconds=TARGET_TIMED_WINDOW_SECONDS,
):
    csv_path = _normalize_csv_path(csv_path)
    rows = []
    for value_dtype in CSV_VALUE_DTYPES:
        for index_dtype in CSV_INDEX_DTYPES:
            print("=" * 180)
            _print_spgemm_mtx_header(value_dtype, index_dtype)
            results = run_mtx_batch(
                paths,
                value_dtype=value_dtype,
                index_dtype=index_dtype,
                warmup=warmup,
                iters=iters,
                run_cusparse=run_cusparse,
                input_mode=input_mode,
                adaptive_loops=adaptive_loops,
                target_window_seconds=target_window_seconds,
                on_result=_print_spgemm_mtx_row,
            )
            print("-" * 320)
            for entry in results:
                n_rows, n_cols = entry["shape"]
                rows.append(
                    {
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
                        "status": entry.get("status"),
                        "ref_reason_code": entry.get("ref_reason_code"),
                        "err_pt": entry.get("err_pt"),
                        "err_cu": entry.get("err_cu"),
                        "max_abs_err_pt": entry.get("max_abs_err_pt"),
                        "max_rel_err_pt": entry.get("max_rel_err_pt"),
                        "max_abs_err_cu": entry.get("max_abs_err_cu"),
                        "max_rel_err_cu": entry.get("max_rel_err_cu"),
                        "pytorch_reason": entry.get("pytorch_reason"),
                        "cusparse_reason": entry.get("cusparse_reason"),
                        "error": entry.get("error"),
                        "nnz_a": entry.get("nnz_a"),
                        "nnz_b": entry.get("nnz_b"),
                        "nnz_c": entry.get("nnz_c"),
                        "input_mode": entry.get("input_mode"),
                        "shape_a": str(entry.get("shape_a")),
                        "shape_b": str(entry.get("shape_b")),
                        "prepare_ms": entry.get("prepare_ms"),
                        "count_ms": entry.get("count_ms"),
                        "fill_ms": entry.get("fill_ms"),
                        "triton_started": entry.get("triton_started"),
                        "ref_started": entry.get("ref_started"),
                        "effective_warmup": entry.get("effective_warmup"),
                        "effective_iters": entry.get("effective_iters"),
                    }
                )
    fieldnames = [
        "matrix", "value_dtype", "index_dtype", "n_rows", "n_cols", "nnz",
        "triton_ms", "cusparse_ms", "pytorch_ms",
        "pt_status", "cu_status", "status", "ref_reason_code", "err_pt", "err_cu",
        "max_abs_err_pt", "max_rel_err_pt", "max_abs_err_cu", "max_rel_err_cu",
        "pytorch_reason", "cusparse_reason", "error",
        "nnz_a", "nnz_b", "nnz_c", "input_mode", "shape_a", "shape_b",
        "prepare_ms", "count_ms", "fill_ms", "triton_started", "ref_started",
        "effective_warmup", "effective_iters",
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
    a_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    a_indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    a_indptr = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    shape = (2, 2)
    c_data, c_indices, c_indptr, _ = ast.flagsparse_spgemm_csr(
        a_data, a_indices, a_indptr, shape,
        a_data, a_indices, a_indptr, shape,
    )
    negative_cases = [
        (
            "shape mismatch",
            lambda: ast.flagsparse_spgemm_csr(
                a_data, a_indices, a_indptr, (2, 3),
                a_data, a_indices, a_indptr, shape,
            ),
            ValueError,
        ),
        (
            "dtype mismatch",
            lambda: ast.flagsparse_spgemm_csr(
                a_data, a_indices, a_indptr, shape,
                a_data.to(torch.float64), a_indices, a_indptr, shape,
            ),
            TypeError,
        ),
        (
            "indices dtype must int32",
            lambda: ast.flagsparse_spgemm_csr(
                a_data, a_indices.to(torch.int64), a_indptr, shape,
                a_data, a_indices, a_indptr, shape,
            ),
            TypeError,
        ),
        (
            "out data must be CUDA",
            lambda: ast.flagsparse_spgemm_csr(
                a_data, a_indices, a_indptr, shape,
                a_data, a_indices, a_indptr, shape,
                out=(
                    torch.empty(c_data.shape, dtype=c_data.dtype),
                    torch.empty(c_indices.shape, dtype=c_indices.dtype, device=device),
                    torch.empty(c_indptr.shape, dtype=c_indptr.dtype, device=device),
                ),
            ),
            ValueError,
        ),
    ]
    failed = 0
    print("-" * 96)
    print("API validation checks (SpGEMM)")
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

    try:
        out = ast.flagsparse_spgemm_csr(
            a_data, a_indices, a_indptr, shape,
            a_data, a_indices, a_indptr, shape,
        )
        if len(out) != 4:
            raise AssertionError("unexpected result tuple length")
        print("PASS  positive path returned CSR tuple")
    except Exception as exc:
        print(f"FAIL  positive path raised {type(exc).__name__}: {exc}")
        failed += 1
    print("-" * 96)
    return failed


def _expand_mtx_paths(raw_paths):
    paths = []
    for p in raw_paths:
        if os.path.isfile(p) and p.lower().endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    seen = set()
    uniq = []
    for path in paths:
        ap = os.path.abspath(path)
        if ap not in seen:
            uniq.append(ap)
            seen.add(ap)
    return uniq


def main():
    parser = argparse.ArgumentParser(description="FlagSparse SpGEMM CSR tests")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--index-dtype", type=str, default="int32", choices=["int32"])
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument(
        "--adaptive-loops",
        action="store_true",
        help="enable adaptive effective_warmup/effective_iters based on first-call runtime",
    )
    parser.add_argument(
        "--target-window-seconds",
        type=float,
        default=TARGET_TIMED_WINDOW_SECONDS,
        help="adaptive target runtime window per matrix (only used with --adaptive-loops)",
    )
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--csv", type=str, default=None, metavar="FILE")
    parser.add_argument(
        "--run-api-checks",
        action="store_true",
        help="run API validation checks before matrix benchmark (disabled by default)",
    )
    parser.add_argument("--skip-api-checks", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--input-mode",
        type=str,
        default=DEFAULT_INPUT_MODE,
        choices=["auto", "a_equals_b", "a_at"],
        help="extra option: auto(square->A@A, rectangular->A@A^T) to avoid shape mismatch on non-square matrices",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    if args.run_api_checks and not args.skip_api_checks:
        failed = run_api_validation_checks()
        if failed > 0:
            raise SystemExit(1)

    value_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    index_dtype = torch.int32
    paths = _expand_mtx_paths(args.mtx)
    if not paths and not args.csv:
        print("No .mtx files given. Use: python test_spgemm.py <file.mtx> [file2.mtx ...] or <dir/>")
        print("Or run all dtypes and export CSV: python test_spgemm.py <dir/> --csv results.csv")
        return

    if args.csv is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        csv_path = _normalize_csv_path(args.csv)
        print("=" * 120)
        print("FLAGSPARSE SpGEMM - f32/f64 with int32, export to CSV")
        print("=" * 120)
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  "
            f"input_mode: {args.input_mode}  |  adaptive_loops: {args.adaptive_loops}  |  CSV: {csv_path}"
        )
        run_all_dtypes_export_csv(
            paths,
            csv_path,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            input_mode=args.input_mode,
            adaptive_loops=args.adaptive_loops,
            target_window_seconds=args.target_window_seconds,
        )
        return

    print("=" * 160)
    print("FLAGSPARSE SpGEMM - SuiteSparse .mtx batch (CSR)")
    print("=" * 160)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(
        f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  warmup: {args.warmup}  "
        f"iters: {args.iters}  adaptive_loops: {args.adaptive_loops}  input_mode: {args.input_mode}"
    )
    print()
    results = run_mtx_batch(
        paths,
        value_dtype=value_dtype,
        index_dtype=index_dtype,
        warmup=args.warmup,
        iters=args.iters,
        run_cusparse=not args.no_cusparse,
        input_mode=args.input_mode,
        adaptive_loops=args.adaptive_loops,
        target_window_seconds=args.target_window_seconds,
    )
    print_mtx_results(results, value_dtype, index_dtype)


if __name__ == "__main__":
    main()
