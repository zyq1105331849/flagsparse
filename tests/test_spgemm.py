"""
SpGEMM tests: load SuiteSparse .mtx, run CSR SpGEMM(A@B) with A=B, and report
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
WARMUP = 5
ITERS = 20
INPUT_MODE = "A_EQUALS_B"


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


def _build_torch_spgemm_reference(a_data, a_indices, a_indptr, shape):
    n_rows, n_cols = shape
    a_csr = torch.sparse_csr_tensor(
        a_indptr.to(torch.int64),
        a_indices.to(torch.int64),
        a_data,
        size=shape,
        device=a_data.device,
    )
    ref_format = "CSR"
    try:
        op = lambda: torch.sparse.mm(a_csr, a_csr)
        ref_sparse = op()
    except Exception:
        ref_format = "COO"
        a_coo = a_csr.to_sparse_coo().coalesce()
        op = lambda: torch.sparse.mm(a_coo, a_coo)
        ref_sparse = op()
    if ref_sparse.layout != torch.sparse_coo and ref_sparse.layout != torch.sparse_csr:
        raise RuntimeError(f"Unexpected torch sparse.mm result layout: {ref_sparse.layout}")
    return ast_ops._torch_sparse_to_csr(ref_sparse), ref_format, op


def _benchmark_triton_spgemm(a_data, a_indices, a_indptr, shape, warmup, iters):
    torch.cuda.synchronize()
    t_prepare0 = time.perf_counter()
    prepared = ast.prepare_spgemm_csr(
        a_data, a_indices, a_indptr, shape,
        a_data, a_indices, a_indptr, shape,
    )
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0

    torch.cuda.synchronize()
    t_first0 = time.perf_counter()
    _ = ast.flagsparse_spgemm_csr(prepared=prepared)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t_first0) * 1000.0

    triton_result, triton_ms = ast_ops._benchmark_cuda_op(
        lambda: ast.flagsparse_spgemm_csr(prepared=prepared),
        warmup=warmup,
        iters=iters,
    )
    _, meta = ast.flagsparse_spgemm_csr(prepared=prepared, return_meta=True)
    meta["prepare_ms"] = prepare_ms
    return triton_result, triton_ms, first_call_ms, meta


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
):
    device = torch.device("cuda")
    a_data, a_indices, a_indptr, shape = load_mtx_to_csr_torch(mtx_path, dtype=value_dtype, device=device)
    a_indices = a_indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz_a = int(a_data.numel())
    result = {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz_a,
        "nnz_a": nnz_a,
        "nnz_b": nnz_a,
        "nnz_c": None,
        "input_mode": INPUT_MODE,
        "error": None,
        "triton_ms": None,
        "triton_first_call_ms": None,
        "prepare_ms": None,
        "count_ms": None,
        "fill_ms": None,
        "pytorch_ms": None,
        "cusparse_ms": None,
        "err_pt": None,
        "err_cu": None,
        "triton_ok_pt": None,
        "triton_ok_cu": None,
        "pytorch_reason": None,
        "cusparse_reason": None,
        "pytorch_format": None,
        "status": "UNKNOWN",
    }

    triton_result = None
    try:
        triton_result, triton_ms, triton_first_ms, meta = _benchmark_triton_spgemm(
            a_data, a_indices, a_indptr, shape, warmup=warmup, iters=iters
        )
        result["triton_ms"] = triton_ms
        result["triton_first_call_ms"] = triton_first_ms
        result["prepare_ms"] = meta.get("prepare_ms")
        result["count_ms"] = meta.get("count_ms")
        result["fill_ms"] = meta.get("fill_ms")
        result["nnz_c"] = int(triton_result[0].numel())
    except Exception as exc:
        result["error"] = f"triton: {exc}"

    try:
        ref_result, ref_format, ref_op = _build_torch_spgemm_reference(a_data, a_indices, a_indptr, shape)
        result["pytorch_format"] = ref_format
        _, result["pytorch_ms"] = ast_ops._benchmark_cuda_op(ref_op, warmup=warmup, iters=iters)
    except Exception as exc:
        result["error"] = str(exc) if result["error"] is None else f"{result['error']}; ref: {exc}"
        result["status"] = "REF_FAIL"
        return result

    if triton_result is not None:
        summary = ast_ops._spgemm_pairwise_summary(triton_result, ref_result, value_dtype)
        result["triton_ok_pt"] = summary["match"]
        result["err_pt"] = summary["max_relative_error"]
    else:
        result["triton_ok_pt"] = False

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
                    shape=shape,
                )
                c_cp, result["cusparse_ms"] = ast_ops._benchmark_cuda_op(
                    lambda: a_cp @ a_cp,
                    warmup=warmup,
                    iters=iters,
                )
                c_coo = c_cp.tocoo()
                rows = ast_ops._torch_from_cupy(c_coo.row).to(torch.int64)
                cols = ast_ops._torch_from_cupy(c_coo.col).to(torch.int64)
                vals = ast_ops._torch_from_cupy(c_coo.data).to(value_dtype)
                c_t = torch.sparse_coo_tensor(
                    torch.stack([rows, cols]), vals, shape, device=device
                ).coalesce()
                c_ref = ast_ops._torch_sparse_to_csr(c_t)
                if triton_result is not None:
                    cu_summary = ast_ops._spgemm_pairwise_summary(triton_result, c_ref, value_dtype)
                    result["triton_ok_cu"] = cu_summary["match"]
                    result["err_cu"] = cu_summary["max_relative_error"]
            except Exception as exc:
                result["cusparse_reason"] = str(exc)

    result["status"] = "PASS" if (result["triton_ok_pt"] or result["triton_ok_cu"]) else "FAIL"
    return result


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    on_result=None,
):
    results = []
    for path in mtx_paths:
        entry = run_one_mtx(
            path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=warmup,
            iters=iters,
            run_cusparse=run_cusparse,
        )
        results.append(entry)
        if on_result is not None:
            on_result(entry)
    return results


def _print_spgemm_mtx_header(value_dtype, index_dtype):
    print(f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}")
    print("Formats: FlagSparse=CSR SpGEMM(A@B, A=B), cuSPARSE=CSR@CSR, PyTorch=sparse.mm.")
    print("Timing stays in native dtype; Err fields are max relative error vs each baseline.")
    print("-" * 206)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ_A':>10} {'NNZ_B':>10} {'NNZ_C':>10} "
        f"{'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>13} {'PyTorch(ms)':>11} "
        f"{'FS/CU':>7} {'FS/PT':>7} {'PT':>6} {'CU':>6} {'Err(PT)':>10} {'Err(CU)':>10} "
        f"{'Prep(ms)':>9} {'Count(ms)':>10} {'Fill(ms)':>9}"
    )
    print("-" * 206)


def _print_spgemm_mtx_row(entry):
    name = os.path.basename(entry["path"])[:27]
    n_rows, n_cols = entry["shape"]
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {entry['nnz_a']:>10} {entry['nnz_b']:>10} {str(entry['nnz_c'] if entry['nnz_c'] is not None else 'N/A'):>10} "
        f"{_fmt_ms(entry.get('triton_ms')):>14} {_fmt_ms(entry.get('cusparse_ms')):>13} {_fmt_ms(entry.get('pytorch_ms')):>11} "
        f"{_fmt_speedup(entry.get('cusparse_ms'), entry.get('triton_ms')):>7} {_fmt_speedup(entry.get('pytorch_ms'), entry.get('triton_ms')):>7} "
        f"{_fmt_check(entry.get('triton_ok_pt')):>6} {_fmt_check(entry.get('triton_ok_cu')):>6} "
        f"{_fmt_err(entry.get('err_pt')):>10} {_fmt_err(entry.get('err_cu')):>10} "
        f"{_fmt_ms(entry.get('prepare_ms')):>9} {_fmt_ms(entry.get('count_ms')):>10} {_fmt_ms(entry.get('fill_ms')):>9}"
    )
    err = entry.get("error")
    if err:
        msg = str(err).replace("\n", " ")
        if len(msg) > 220:
            msg = msg[:217] + "..."
        print(f"  NOTE: {msg}")


def print_mtx_results(results, value_dtype, index_dtype):
    _print_spgemm_mtx_header(value_dtype, index_dtype)
    for entry in results:
        _print_spgemm_mtx_row(entry)
    print("-" * 206)


def run_all_dtypes_export_csv(
    paths,
    csv_path,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
):
    csv_path = _normalize_csv_path(csv_path)
    rows = []
    for value_dtype in CSV_VALUE_DTYPES:
        for index_dtype in CSV_INDEX_DTYPES:
            print("=" * 170)
            _print_spgemm_mtx_header(value_dtype, index_dtype)
            results = run_mtx_batch(
                paths,
                value_dtype=value_dtype,
                index_dtype=index_dtype,
                warmup=warmup,
                iters=iters,
                run_cusparse=run_cusparse,
                on_result=_print_spgemm_mtx_row,
            )
            print("-" * 206)
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
                        "status": "PASS" if (entry.get("triton_ok_pt") or entry.get("triton_ok_cu")) else "FAIL",
                        "err_pt": entry.get("err_pt"),
                        "err_cu": entry.get("err_cu"),
                        "error": entry.get("error"),
                        "nnz_a": entry.get("nnz_a"),
                        "nnz_b": entry.get("nnz_b"),
                        "nnz_c": entry.get("nnz_c"),
                        "input_mode": entry.get("input_mode"),
                        "prepare_ms": entry.get("prepare_ms"),
                        "count_ms": entry.get("count_ms"),
                        "fill_ms": entry.get("fill_ms"),
                    }
                )
    fieldnames = [
        "matrix", "value_dtype", "index_dtype", "n_rows", "n_cols", "nnz",
        "triton_ms", "cusparse_ms", "pytorch_ms",
        "pt_status", "cu_status", "status", "err_pt", "err_cu", "error",
        "nnz_a", "nnz_b", "nnz_c", "input_mode", "prepare_ms", "count_ms", "fill_ms",
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
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--csv", type=str, default=None, metavar="FILE")
    parser.add_argument("--skip-api-checks", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    if not args.skip_api_checks:
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
        print("=" * 108)
        print("FLAGSPARSE SpGEMM - f32/f64 with int32, export to CSV (A=B)")
        print("=" * 108)
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  CSV: {csv_path}")
        run_all_dtypes_export_csv(
            paths,
            csv_path,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
        )
        return

    print("=" * 146)
    print("FLAGSPARSE SpGEMM - SuiteSparse .mtx batch (A=B, CSR)")
    print("=" * 146)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(
        f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  warmup: {args.warmup}  iters: {args.iters}  input_mode: {INPUT_MODE}"
    )
    print()
    results = run_mtx_batch(
        paths,
        value_dtype=value_dtype,
        index_dtype=index_dtype,
        warmup=args.warmup,
        iters=args.iters,
        run_cusparse=not args.no_cusparse,
    )
    print_mtx_results(results, value_dtype, index_dtype)


if __name__ == "__main__":
    main()
