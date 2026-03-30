"""
SDDMM float32 A/B test: baseline_f32 vs acc64_f32, with float64->float32
PyTorch reference and optional CuPy sampled-dot timing.

Usage:
    python tests/test_sddmm_f32or64.py <dir/>
    python tests/test_sddmm_f32or64.py <dir/> --k 256 --csv sddmm_f32or64.csv
"""

import argparse
import csv
import glob
import math
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

import flagsparse as fs
import flagsparse.sparse_operations.sddmm_csr as ast_ops
from test_sddmm import (
    _benchmark_cupy_sampled_reference,
    _expand_mtx_paths,
    _fmt_check,
    _fmt_err,
    _fmt_ms,
    _fmt_speedup,
    _is_resource_error,
    _normalize_csv_path,
    run_api_validation_checks,
)
from test_spmm import load_mtx_to_csr_torch

WARMUP = 5
ITERS = 20
DEFAULT_K = 64
BASELINE_ATOL = 1e-4
BASELINE_RTOL = 1e-2
ACC64_ATOL = 1e-6
ACC64_RTOL = 1e-5


def _scaled_error(candidate, reference, atol, rtol):
    if candidate.numel() == 0:
        return 0.0
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    return float(torch.max(diff / denom).item())


def _timed_triton_variant(prepared, data, x, y, alpha, beta, variant, warmup, iters):
    x_in = x.contiguous()
    y_in = y.contiguous()
    data_in = data.contiguous() if data is not None else None

    torch.cuda.synchronize()
    t_first0 = time.perf_counter()
    _first, meta = ast_ops._run_sddmm_prepared(
        prepared,
        x_in,
        y_in,
        data_in,
        alpha,
        beta,
        out=None,
        allow_fallback=False,
        variant=variant,
    )
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t_first0) * 1000.0

    op = lambda: ast_ops._run_sddmm_prepared(
        prepared,
        x_in,
        y_in,
        data_in,
        alpha,
        beta,
        out=None,
        allow_fallback=False,
        variant=variant,
    )[0]
    values, elapsed_ms = ast_ops._benchmark_cuda_op(op, warmup=warmup, iters=iters)
    return values, elapsed_ms, first_call_ms, meta


def _timed_reference_f64_to_f32(indices, indptr, x, y, data, alpha, beta, warmup, iters):
    x_ref = x.to(torch.float64)
    y_ref = y.to(torch.float64)
    data_ref = data.to(torch.float64) if data is not None else None
    indptr64 = indptr.to(torch.int64)

    op = lambda: ast_ops._sddmm_reference(indices, indptr64, x_ref, y_ref, data_ref, alpha, beta).to(torch.float32)
    values, elapsed_ms = ast_ops._benchmark_cuda_op(op, warmup=warmup, iters=iters)
    return values, elapsed_ms


def run_one_mtx(path, warmup=WARMUP, iters=ITERS, k_dim=DEFAULT_K, alpha=1.0, beta=0.0, run_cupy=True):
    device = torch.device("cuda")
    _pattern_values, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=torch.float32, device=device)
    indices = indices.to(torch.int32)
    n_rows, n_cols = shape
    nnz = int(indices.numel())
    data = torch.randn(nnz, dtype=torch.float32, device=device)
    x = torch.randn((n_rows, k_dim), dtype=torch.float32, device=device)
    y = torch.randn((n_cols, k_dim), dtype=torch.float32, device=device)

    torch.cuda.synchronize()
    t_prepare0 = time.perf_counter()
    prepared = fs.prepare_sddmm_csr(indices, indptr, shape, k_hint=int(k_dim))
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0

    result = {
        "path": path,
        "shape": shape,
        "nnz": nnz,
        "k": int(k_dim),
        "alpha": float(alpha),
        "beta": float(beta),
        "prepare_ms": prepare_ms,
        "baseline_ms": None,
        "baseline_first_call_ms": None,
        "acc64_ms": None,
        "acc64_first_call_ms": None,
        "pt_ms": None,
        "cupy_ms": None,
        "err_baseline": None,
        "err_acc64": None,
        "baseline_ok": None,
        "acc64_ok": None,
        "status": "UNKNOWN",
        "cu_status": "PERF_ONLY",
        "cu_reason": None,
        "error": None,
        "baseline_acc_dtype": None,
        "acc64_acc_dtype": None,
        "baseline_out_dtype": None,
        "acc64_out_dtype": None,
    }

    y_ref = None
    try:
        y_ref, pt_ms = _timed_reference_f64_to_f32(
            indices=indices,
            indptr=indptr,
            x=x,
            y=y,
            data=data,
            alpha=alpha,
            beta=beta,
            warmup=warmup,
            iters=iters,
        )
        result["pt_ms"] = pt_ms
    except Exception as exc:
        result["error"] = f"ref: {exc}"
        result["status"] = "REF_FAIL"
        return result

    y_base = None
    try:
        y_base, base_ms, base_first_ms, meta = _timed_triton_variant(
            prepared=prepared,
            data=data,
            x=x,
            y=y,
            alpha=alpha,
            beta=beta,
            variant="baseline",
            warmup=warmup,
            iters=iters,
        )
        result["baseline_ms"] = base_ms
        result["baseline_first_call_ms"] = base_first_ms
        result["baseline_acc_dtype"] = meta.get("acc_dtype")
        result["baseline_out_dtype"] = meta.get("out_dtype")
    except Exception as exc:
        result["error"] = f"baseline: {exc}"

    y_acc64 = None
    try:
        y_acc64, acc64_ms, acc64_first_ms, meta = _timed_triton_variant(
            prepared=prepared,
            data=data,
            x=x,
            y=y,
            alpha=alpha,
            beta=beta,
            variant="acc64",
            warmup=warmup,
            iters=iters,
        )
        result["acc64_ms"] = acc64_ms
        result["acc64_first_call_ms"] = acc64_first_ms
        result["acc64_acc_dtype"] = meta.get("acc_dtype")
        result["acc64_out_dtype"] = meta.get("out_dtype")
    except Exception as exc:
        msg = f"acc64: {exc}"
        result["error"] = msg if result["error"] is None else f"{result['error']}; {msg}"

    if y_base is not None:
        result["err_baseline"] = _scaled_error(y_base, y_ref, BASELINE_ATOL, BASELINE_RTOL)
        result["baseline_ok"] = bool(torch.allclose(y_base, y_ref, atol=BASELINE_ATOL, rtol=BASELINE_RTOL))

    if y_acc64 is not None:
        result["err_acc64"] = _scaled_error(y_acc64, y_ref, ACC64_ATOL, ACC64_RTOL)
        result["acc64_ok"] = bool(torch.allclose(y_acc64, y_ref, atol=ACC64_ATOL, rtol=ACC64_RTOL))

    if run_cupy:
        if ast_ops.cp is None:
            result["cu_reason"] = "CuPy is not available"
        else:
            try:
                _cu_vals, cupy_ms = _benchmark_cupy_sampled_reference(
                    indices=indices,
                    indptr=indptr,
                    x=x,
                    y=y,
                    data_in=data,
                    alpha=alpha,
                    beta=beta,
                    warmup=warmup,
                    iters=iters,
                )
                result["cupy_ms"] = cupy_ms
            except Exception as exc:
                result["cu_status"] = "PERF_UNAVAILABLE" if not _is_resource_error(exc) else "PERF_RESOURCE"
                result["cu_reason"] = str(exc)
    else:
        result["cu_reason"] = "CuPy timing is disabled by CLI"

    result["status"] = "PASS" if bool(result["acc64_ok"]) else "FAIL"
    return result


def print_header():
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'K':>6}  "
        f"{'Base(ms)':>9} {'Acc64(ms)':>10} {'PT(ms)':>9} {'CU(ms)':>9}  "
        f"{'Base/Acc64':>10} {'Base/PT':>8} {'Acc64/PT':>8} {'Base/CU':>8} {'Acc64/CU':>8}  "
        f"{'Err(Base)':>10} {'Err(Acc64)':>12} {'Base':>6} {'Acc64':>6} {'Status':>6}"
    )


def print_row(row):
    n_rows, n_cols = row["shape"]
    name = os.path.basename(row["path"])[:27]
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {row['nnz']:>10} {row['k']:>6}  "
        f"{_fmt_ms(row['baseline_ms']):>9} {_fmt_ms(row['acc64_ms']):>10} {_fmt_ms(row['pt_ms']):>9} {_fmt_ms(row['cupy_ms']):>9}  "
        f"{_fmt_speedup(row['baseline_ms'], row['acc64_ms']):>10} "
        f"{_fmt_speedup(row['pt_ms'], row['baseline_ms']):>8} {_fmt_speedup(row['pt_ms'], row['acc64_ms']):>8} "
        f"{_fmt_speedup(row['cupy_ms'], row['baseline_ms']):>8} {_fmt_speedup(row['cupy_ms'], row['acc64_ms']):>8}  "
        f"{_fmt_err(row['err_baseline']):>10} {_fmt_err(row['err_acc64']):>12} "
        f"{_fmt_check(row['baseline_ok']):>6} {_fmt_check(row['acc64_ok']):>6} {row['status']:>6}"
    )
    if row.get("error"):
        print(f"  NOTE: {str(row['error']).replace(chr(10), ' ')}")
    if row.get("cu_reason"):
        print(f"  CU_NOTE: {str(row['cu_reason']).replace(chr(10), ' ')}")


def run_batch(paths, warmup=WARMUP, iters=ITERS, k_dim=DEFAULT_K, alpha=1.0, beta=0.0, run_cupy=True):
    results = []
    for path in paths:
        try:
            row = run_one_mtx(
                path,
                warmup=warmup,
                iters=iters,
                k_dim=k_dim,
                alpha=alpha,
                beta=beta,
                run_cupy=run_cupy,
            )
        except Exception as exc:
            print(f"  ERROR on {os.path.basename(path)}: {exc}")
            continue
        results.append(row)
        print_row(row)
    return results


def export_csv(results, csv_path):
    csv_path = _normalize_csv_path(csv_path)
    fields = [
        "matrix",
        "n_rows",
        "n_cols",
        "nnz",
        "k",
        "alpha",
        "beta",
        "prepare_ms",
        "baseline_ms",
        "baseline_first_call_ms",
        "acc64_ms",
        "acc64_first_call_ms",
        "pt_ms",
        "cupy_ms",
        "baseline_vs_acc64",
        "baseline_vs_pt",
        "acc64_vs_pt",
        "baseline_vs_cu",
        "acc64_vs_cu",
        "err_baseline",
        "err_acc64",
        "baseline_ok",
        "acc64_ok",
        "baseline_acc_dtype",
        "acc64_acc_dtype",
        "baseline_out_dtype",
        "acc64_out_dtype",
        "cu_status",
        "cu_reason",
        "status",
        "error",
    ]
    rows = []
    for row in results:
        n_rows, n_cols = row["shape"]
        rows.append(
            {
                "matrix": os.path.basename(row["path"]),
                "n_rows": n_rows,
                "n_cols": n_cols,
                "nnz": row["nnz"],
                "k": row["k"],
                "alpha": row["alpha"],
                "beta": row["beta"],
                "prepare_ms": row["prepare_ms"],
                "baseline_ms": row["baseline_ms"],
                "baseline_first_call_ms": row["baseline_first_call_ms"],
                "acc64_ms": row["acc64_ms"],
                "acc64_first_call_ms": row["acc64_first_call_ms"],
                "pt_ms": row["pt_ms"],
                "cupy_ms": row["cupy_ms"],
                "baseline_vs_acc64": (row["baseline_ms"] / row["acc64_ms"] if row["baseline_ms"] and row["acc64_ms"] and row["acc64_ms"] > 0 else None),
                "baseline_vs_pt": (row["pt_ms"] / row["baseline_ms"] if row["pt_ms"] and row["baseline_ms"] and row["baseline_ms"] > 0 else None),
                "acc64_vs_pt": (row["pt_ms"] / row["acc64_ms"] if row["pt_ms"] and row["acc64_ms"] and row["acc64_ms"] > 0 else None),
                "baseline_vs_cu": (row["cupy_ms"] / row["baseline_ms"] if row["cupy_ms"] and row["baseline_ms"] and row["baseline_ms"] > 0 else None),
                "acc64_vs_cu": (row["cupy_ms"] / row["acc64_ms"] if row["cupy_ms"] and row["acc64_ms"] and row["acc64_ms"] > 0 else None),
                "err_baseline": row["err_baseline"],
                "err_acc64": row["err_acc64"],
                "baseline_ok": row["baseline_ok"],
                "acc64_ok": row["acc64_ok"],
                "baseline_acc_dtype": row["baseline_acc_dtype"],
                "acc64_acc_dtype": row["acc64_acc_dtype"],
                "baseline_out_dtype": row["baseline_out_dtype"],
                "acc64_out_dtype": row["acc64_out_dtype"],
                "cu_status": row["cu_status"],
                "cu_reason": row["cu_reason"],
                "status": row["status"],
                "error": row["error"],
            }
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    print(f"Wrote {len(rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SDDMM float32 A/B: baseline_f32 vs acc64_f32, with float64->float32 PyTorch reference."
    )
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--csv", type=str, default=None, metavar="FILE", help="Export rows to CSV")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Dense feature dimension K")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--no-cupy-ref", action="store_true", help="Skip CuPy sampled-dot timing")
    parser.add_argument("--no-cusparse", action="store_true", help="Deprecated alias of --no-cupy-ref")
    parser.add_argument("--skip-api-checks", action="store_true")
    args = parser.parse_args()

    if args.k < 0:
        raise ValueError("--k must be non-negative")
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    if not args.skip_api_checks:
        failed = run_api_validation_checks()
        if failed > 0:
            raise SystemExit(1)

    paths = _expand_mtx_paths(args.mtx)
    if not paths and not args.csv:
        print("No .mtx files. Usage: python test_sddmm_f32or64.py <dir/> [--csv out.csv]")
        return
    if args.csv is not None and not paths:
        paths = sorted(glob.glob("*.mtx"))
    if not paths:
        print("No .mtx files found. Specify files or a directory.")
        return

    run_cupy = not (args.no_cupy_ref or args.no_cusparse)

    print("=" * 190)
    print("FLAGSPARSE SDDMM float32 A/B test")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  K: {args.k}  |  alpha: {args.alpha}  |  beta: {args.beta}")
    print(
        "Reference = PyTorch sampled-dot in float64, cast back to float32. "
        "Baseline = native float32 accumulate with relaxed tolerance. "
        "Acc64 = float64 accumulate, float32 output, checked with strict tolerance. "
        "CuPy is performance-only."
    )
    print("-" * 190)
    print_header()
    print("-" * 190)

    results = run_batch(
        paths,
        warmup=args.warmup,
        iters=args.iters,
        k_dim=args.k,
        alpha=args.alpha,
        beta=args.beta,
        run_cupy=run_cupy,
    )
    print("-" * 190)
    passed = sum(1 for row in results if row["status"] == "PASS")
    print(f"Passed(acc64): {passed} / {len(results)}")

    if args.csv:
        export_csv(results, args.csv)


if __name__ == "__main__":
    main()
