"""
SDDMM tests: load SuiteSparse .mtx as CSR pattern and benchmark
out = alpha * dot(X[row], Y[col]) + beta * in.
CuPy baseline uses sampled-dot on CSR pattern (not dense X@Y^T).

acc_mode notes:
- acc_mode=f32 keeps the native float32 accumulate path for float32 inputs.
- acc_mode=f64 upgrades only the internal accumulation of float32 inputs to
  float64 while still returning float32 outputs.
- float64 inputs always keep the existing float64 route; acc_mode only affects
  float32 runs in this test harness.
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
import flagsparse.sparse_operations.sddmm_csr as ast_ops
from test_spmm import load_mtx_to_csr_torch

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
WARMUP = 5
ITERS = 20
DEFAULT_K = 64
BASELINE_ATOL = 1e-4
BASELINE_RTOL = 1e-2
ACC64_ATOL = 1e-6
ACC64_RTOL = 1e-5


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
    if isinstance(value, str):
        return value
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def _is_resource_error(message):
    text = str(message).lower()
    resource_tokens = (
        "out of memory",
        "cudaerroroutofmemory",
        "cuda error out of memory",
        "insufficient resources",
        "resource exhausted",
        "memoryerror",
        "cublas_status_alloc_failed",
        "cusparse_status_insufficient_resources",
    )
    return any(token in text for token in resource_tokens)


def _cupy_sampled_dot_chunked(x_cp, y_cp, row_ids_cp, col_ids_cp, chunk_nnz):
    nnz = int(row_ids_cp.size)
    out_cp = ast_ops.cp.empty((nnz,), dtype=x_cp.dtype)
    for start in range(0, nnz, chunk_nnz):
        end = min(nnz, start + chunk_nnz)
        rows = row_ids_cp[start:end]
        cols = col_ids_cp[start:end]
        out_cp[start:end] = ast_ops.cp.sum(x_cp[rows] * y_cp[cols], axis=1)
    return out_cp


def _benchmark_cupy_sampled_reference(indices, indptr, x, y, data_in, alpha, beta, warmup, iters):
    n_rows = int(indptr.numel()) - 1
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, dtype=torch.int64, device=x.device),
        indptr.to(torch.int64)[1:] - indptr.to(torch.int64)[:-1],
    )
    x_cp = ast_ops._cupy_from_torch(x)
    y_cp = ast_ops._cupy_from_torch(y)
    row_ids_cp = ast_ops._cupy_from_torch(row_ids)
    col_ids_cp = ast_ops._cupy_from_torch(indices.to(torch.int64))
    nnz = max(1, int(indices.numel()))
    chunk_nnz = min(262144, nnz)
    sampled_cp, cupy_ms = ast_ops._benchmark_cuda_op(
        lambda: _cupy_sampled_dot_chunked(x_cp, y_cp, row_ids_cp, col_ids_cp, chunk_nnz),
        warmup=warmup,
        iters=iters,
    )
    sampled = ast_ops._torch_from_cupy(sampled_cp)
    sampled = sampled * alpha
    if beta != 0.0:
        sampled = sampled + beta * data_in
    return sampled, cupy_ms


def _normalize_csv_path(csv_path):
    csv_path = str(csv_path)
    if not csv_path.lower().endswith(".csv"):
        csv_path = f"{csv_path}.csv"
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return csv_path


def _resolve_tolerance(value_dtype, acc_mode):
    if value_dtype == torch.float32:
        if acc_mode == "f64":
            return ACC64_ATOL, ACC64_RTOL
        return BASELINE_ATOL, BASELINE_RTOL
    return ast_ops._tolerance_for_dtype(value_dtype)


def _scaled_allclose_error(candidate, reference, atol, rtol):
    if candidate.numel() == 0:
        return 0.0
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    return float(torch.max(diff / denom).item())


def _benchmark_reference_sddmm(data, indices, indptr, x, y, alpha, beta, value_dtype, warmup, iters):
    indptr64 = indptr.to(torch.int64)
    if value_dtype == torch.float32:
        x_ref = x.to(torch.float64)
        y_ref = y.to(torch.float64)
        data_ref = data.to(torch.float64) if data is not None else None

        op = lambda: ast_ops._sddmm_reference(indices, indptr64, x_ref, y_ref, data_ref, alpha, beta).to(torch.float32)
    else:
        op = lambda: ast_ops._sddmm_reference(indices, indptr64, x, y, data, alpha, beta)
    ref_values, ref_ms = ast_ops._benchmark_cuda_op(op, warmup=warmup, iters=iters)
    return ref_values, ref_ms


def _benchmark_triton_sddmm(data, indices, indptr, shape, x, y, alpha, beta, warmup, iters, acc_mode):
    torch.cuda.synchronize()
    t_prepare0 = time.perf_counter()
    prepared = ast.prepare_sddmm_csr(indices, indptr, shape, k_hint=int(x.shape[1]))
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0

    torch.cuda.synchronize()
    t_first0 = time.perf_counter()
    if x.dtype == torch.float32 and acc_mode == "f64":
        _ = ast_ops._run_sddmm_prepared(
            prepared,
            x.contiguous(),
            y.contiguous(),
            data.contiguous() if data is not None else None,
            alpha,
            beta,
            out=None,
            allow_fallback=False,
            variant="acc64",
        )[0]
    else:
        _ = ast.flagsparse_sddmm_csr(data=data, x=x, y=y, alpha=alpha, beta=beta, prepared=prepared)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t_first0) * 1000.0

    if x.dtype == torch.float32 and acc_mode == "f64":
        op = lambda: ast_ops._run_sddmm_prepared(
            prepared,
            x.contiguous(),
            y.contiguous(),
            data.contiguous() if data is not None else None,
            alpha,
            beta,
            out=None,
            allow_fallback=False,
            variant="acc64",
        )[0]
        triton_values, triton_ms = ast_ops._benchmark_cuda_op(op, warmup=warmup, iters=iters)
        _, meta = ast_ops._run_sddmm_prepared(
            prepared,
            x.contiguous(),
            y.contiguous(),
            data.contiguous() if data is not None else None,
            alpha,
            beta,
            out=None,
            allow_fallback=False,
            variant="acc64",
        )
    else:
        triton_values, triton_ms = ast_ops._benchmark_cuda_op(
            lambda: ast.flagsparse_sddmm_csr(data=data, x=x, y=y, alpha=alpha, beta=beta, prepared=prepared),
            warmup=warmup,
            iters=iters,
        )
        _, meta = ast.flagsparse_sddmm_csr(
            data=data, x=x, y=y, alpha=alpha, beta=beta, prepared=prepared, return_meta=True
        )
    meta["prepare_ms"] = prepare_ms
    return triton_values, triton_ms, first_call_ms, meta


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    k_dim=DEFAULT_K,
    alpha=1.0,
    beta=0.0,
    run_cusparse=True,
    acc_mode="f32",
):
    device = torch.device("cuda")
    _pattern_values, indices, indptr, shape = load_mtx_to_csr_torch(mtx_path, dtype=value_dtype, device=device)
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz = int(indices.numel())
    data_in = torch.randn(nnz, dtype=value_dtype, device=device)
    x = torch.randn((n_rows, k_dim), dtype=value_dtype, device=device)
    y = torch.randn((n_cols, k_dim), dtype=value_dtype, device=device)

    result = {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz,
        "nnz_pattern": nnz,
        "k": int(k_dim),
        "alpha": float(alpha),
        "beta": float(beta),
        "error": None,
        "triton_ms": None,
        "triton_first_call_ms": None,
        "prepare_ms": None,
        "pytorch_ms": None,
        "cupy_ms": None,
        "cusparse_ms": None,
        "err_pt": None,
        "err_cu": None,
        "triton_ok_pt": None,
        "triton_ok_cu": None,
        "cu_status": "REF_UNAVAILABLE",
        "cu_reason": None,
        "cusparse_reason": None,
        "triton_started": False,
        "cu_started": False,
        "fallback_used": False,
        "status": "UNKNOWN",
    }

    triton_values = None
    try:
        result["triton_started"] = True
        triton_values, triton_ms, triton_first_ms, meta = _benchmark_triton_sddmm(
            data_in, indices, indptr, shape, x, y, alpha, beta, warmup, iters, acc_mode
        )
        result["triton_ms"] = triton_ms
        result["triton_first_call_ms"] = triton_first_ms
        result["prepare_ms"] = meta.get("prepare_ms")
        result["fallback_used"] = bool(meta.get("fallback_used", False))
    except Exception as exc:
        result["error"] = f"triton: {exc}"

    try:
        ref, result["pytorch_ms"] = _benchmark_reference_sddmm(
            data_in,
            indices,
            indptr,
            x,
            y,
            alpha,
            beta,
            value_dtype,
            warmup,
            iters,
        )
    except Exception as exc:
        result["error"] = str(exc) if result["error"] is None else f"{result['error']}; ref: {exc}"
        result["status"] = "REF_FAIL"
        return result

    if triton_values is not None:
        atol, rtol = _resolve_tolerance(value_dtype, acc_mode)
        result["triton_ok_pt"] = bool(torch.allclose(triton_values, ref, atol=atol, rtol=rtol))
        result["err_pt"] = _scaled_allclose_error(triton_values, ref, atol, rtol)
    else:
        result["triton_ok_pt"] = False

    if run_cusparse:
        if ast_ops.cp is None:
            result["cu_status"] = "PERF_ONLY"
            result["cu_reason"] = "CuPy is not available"
        else:
            try:
                result["cu_started"] = True
                _cu_vals, cupy_ms = _benchmark_cupy_sampled_reference(
                    indices=indices,
                    indptr=indptr,
                    x=x,
                    y=y,
                    data_in=data_in,
                    alpha=alpha,
                    beta=beta,
                    warmup=warmup,
                    iters=iters,
                )
                result["cupy_ms"] = cupy_ms
                result["cusparse_ms"] = cupy_ms
                result["cu_status"] = "PERF_ONLY"
            except Exception as exc:
                result["cu_status"] = "PERF_RESOURCE" if _is_resource_error(exc) else "PERF_UNAVAILABLE"
                result["cu_reason"] = str(exc)
    else:
        result["cu_status"] = "PERF_ONLY"
        result["cu_reason"] = "CuPy reference is disabled by CLI"

    result["cusparse_reason"] = result["cu_reason"]
    result["status"] = "PASS" if result["triton_ok_pt"] else "FAIL"
    return result


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    k_dim=DEFAULT_K,
    alpha=1.0,
    beta=0.0,
    run_cusparse=True,
    on_result=None,
    acc_mode="f32",
):
    results = []
    for path in mtx_paths:
        entry = run_one_mtx(
            path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=warmup,
            iters=iters,
            k_dim=k_dim,
            alpha=alpha,
            beta=beta,
            run_cusparse=run_cusparse,
            acc_mode=acc_mode,
        )
        results.append(entry)
        if on_result is not None:
            on_result(entry)
    return results


def _print_sddmm_mtx_header(value_dtype, index_dtype, k_dim, alpha, beta, acc_mode):
    print(f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}")
    print(
        "Formats: FlagSparse=CSR SDDMM, CuPy sampled-dot performance baseline (not cuSPARSE API), "
        "PyTorch correctness reference."
    )
    print(
        f"Equation: out = alpha*dot(x[row], y[col]) + beta*in  |  K={k_dim}  alpha={alpha}  beta={beta}  acc_mode={acc_mode}"
    )
    print("-" * 196)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'K':>6} "
        f"{'FlagSparse(ms)':>14} {'CuPy(ms)':>11} {'PyTorch(ms)':>11} "
        f"{'FS/CU':>7} {'FS/PT':>7} {'PT':>6} {'CU_Status':>12} {'Err(PT)':>10} {'Err(CU)':>10} {'Prep(ms)':>9}"
    )
    print("-" * 196)


def _print_sddmm_mtx_row(entry):
    name = os.path.basename(entry["path"])[:27]
    n_rows, n_cols = entry["shape"]
    cupy_ms = entry.get("cupy_ms")
    if cupy_ms is None:
        cupy_ms = entry.get("cusparse_ms")
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {entry['nnz_pattern']:>10} {entry['k']:>6} "
        f"{_fmt_ms(entry.get('triton_ms')):>14} {_fmt_ms(cupy_ms):>11} {_fmt_ms(entry.get('pytorch_ms')):>11} "
        f"{_fmt_speedup(cupy_ms, entry.get('triton_ms')):>7} {_fmt_speedup(entry.get('pytorch_ms'), entry.get('triton_ms')):>7} "
        f"{_fmt_check(entry.get('triton_ok_pt')):>6} {_status_label(entry.get('cu_status')):>12} "
        f"{_fmt_err(entry.get('err_pt')):>10} {_fmt_err(entry.get('err_cu')):>10} {_fmt_ms(entry.get('prepare_ms')):>9}"
    )
    err = entry.get("error")
    cu_reason = entry.get("cu_reason")
    if err:
        msg = str(err).replace("\n", " ")
        if len(msg) > 220:
            msg = msg[:217] + "..."
        print(f"  NOTE: {msg}")
    if cu_reason:
        msg = str(cu_reason).replace("\n", " ")
        if len(msg) > 220:
            msg = msg[:217] + "..."
        print(f"  CU_NOTE: {msg}")


def print_mtx_results(results, value_dtype, index_dtype, k_dim, alpha, beta, acc_mode):
    _print_sddmm_mtx_header(value_dtype, index_dtype, k_dim, alpha, beta, acc_mode)
    for entry in results:
        _print_sddmm_mtx_row(entry)
    print("-" * 196)


def run_all_dtypes_export_csv(
    paths,
    csv_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    k_dim=DEFAULT_K,
    alpha=1.0,
    beta=0.0,
    run_cusparse=True,
    acc_mode="f32",
):
    csv_path = _normalize_csv_path(csv_path)
    rows = []
    print("=" * 164)
    _print_sddmm_mtx_header(value_dtype, index_dtype, k_dim, alpha, beta, acc_mode)
    results = run_mtx_batch(
        paths,
        value_dtype=value_dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        k_dim=k_dim,
        alpha=alpha,
        beta=beta,
        run_cusparse=run_cusparse,
        on_result=_print_sddmm_mtx_row,
        acc_mode=acc_mode,
    )
    print("-" * 196)
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
                "cupy_ms": entry.get("cupy_ms"),
                "triton_ms": entry.get("triton_ms"),
                "cusparse_ms": entry.get("cusparse_ms"),
                "pytorch_ms": entry.get("pytorch_ms"),
                "pt_status": _status_label(entry.get("triton_ok_pt")),
                "cu_status": _status_label(entry.get("cu_status")),
                "status": entry.get("status"),
                "err_pt": entry.get("err_pt"),
                "err_cu": entry.get("err_cu"),
                "error": entry.get("error"),
                "cu_reason": entry.get("cu_reason"),
                "triton_started": entry.get("triton_started"),
                "cu_started": entry.get("cu_started"),
                "fallback_used": entry.get("fallback_used"),
                "nnz_pattern": entry.get("nnz_pattern"),
                "k": entry.get("k"),
                "alpha": entry.get("alpha"),
                "beta": entry.get("beta"),
                "prepare_ms": entry.get("prepare_ms"),
            }
        )
    fieldnames = [
        "matrix", "value_dtype", "index_dtype", "n_rows", "n_cols", "nnz",
        "triton_ms", "cupy_ms", "cusparse_ms", "pytorch_ms",
        "pt_status", "cu_status", "status", "err_pt", "err_cu", "error",
        "cu_reason", "triton_started", "cu_started", "fallback_used",
        "nnz_pattern", "k", "alpha", "beta", "prepare_ms",
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
    indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    shape = (2, 2)
    x = torch.randn((2, 8), dtype=torch.float32, device=device)
    y = torch.randn((2, 8), dtype=torch.float32, device=device)
    data = torch.randn(3, dtype=torch.float32, device=device)

    negative_cases = [
        ("indices must int32", lambda: ast.flagsparse_sddmm_csr(indices=indices.to(torch.int64), indptr=indptr, x=x, y=y, shape=shape), TypeError),
        ("x/y K mismatch", lambda: ast.flagsparse_sddmm_csr(indices=indices, indptr=indptr, x=x, y=y[:, :4], shape=shape), ValueError),
        ("data length mismatch", lambda: ast.flagsparse_sddmm_csr(data=torch.randn(2, dtype=torch.float32, device=device), indices=indices, indptr=indptr, x=x, y=y, shape=shape), ValueError),
        ("beta needs data", lambda: ast.flagsparse_sddmm_csr(indices=indices, indptr=indptr, x=x, y=y, shape=shape, beta=0.5), ValueError),
        (
            "K=0 out shape mismatch",
            lambda: ast.flagsparse_sddmm_csr(
                data=data,
                indices=indices,
                indptr=indptr,
                x=x[:, :0],
                y=y[:, :0],
                shape=shape,
                out=torch.empty(2, dtype=torch.float32, device=device),
            ),
            ValueError,
        ),
    ]
    failed = 0
    print("-" * 96)
    print("API validation checks (SDDMM)")
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
        out = ast.flagsparse_sddmm_csr(data=data, indices=indices, indptr=indptr, x=x, y=y, shape=shape, alpha=1.25, beta=0.5)
        if out.shape != (3,):
            raise AssertionError("unexpected output shape")
        print("PASS  positive path returned correct output shape")
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
    parser = argparse.ArgumentParser(description="FlagSparse SDDMM CSR tests")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--index-dtype", type=str, default="int32", choices=["int32"])
    parser.add_argument(
        "--acc_mode",
        type=str,
        default="f32",
        choices=["f32", "f64"],
        help="For float32 runs, choose native f32 accumulation or float64 accumulation.",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Dense feature dimension K")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--no-cupy-ref",
        action="store_true",
        help="Skip CuPy sampled-dot performance baseline",
    )
    parser.add_argument(
        "--no-cusparse",
        action="store_true",
        help="Deprecated alias of --no-cupy-ref",
    )
    parser.add_argument("--csv", type=str, default=None, metavar="FILE")
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
    run_cupy_ref = not (args.no_cupy_ref or args.no_cusparse)

    value_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    index_dtype = torch.int32
    paths = _expand_mtx_paths(args.mtx)
    if not paths and not args.csv:
        print("No .mtx files given. Use: python test_sddmm.py <file.mtx> [file2.mtx ...] or <dir/>")
        print("Or export the current dtype to CSV: python test_sddmm.py <dir/> --csv results.csv")
        return

    if args.csv is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        csv_path = _normalize_csv_path(args.csv)
        print("=" * 110)
        print("FLAGSPARSE SDDMM - export to CSV")
        print("=" * 110)
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  dtype: {args.dtype}  |  acc_mode: {args.acc_mode}  |  K: {args.k}  |  alpha: {args.alpha}  |  beta: {args.beta}  |  CSV: {csv_path}"
        )
        run_all_dtypes_export_csv(
            paths,
            csv_path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=args.warmup,
            iters=args.iters,
            k_dim=args.k,
            alpha=args.alpha,
            beta=args.beta,
            run_cusparse=run_cupy_ref,
            acc_mode=args.acc_mode,
        )
        return

    print("=" * 150)
    print("FLAGSPARSE SDDMM - SuiteSparse .mtx batch (CSR pattern-guided)")
    print("=" * 150)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(
        f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  acc_mode: {args.acc_mode}  K: {args.k}  alpha: {args.alpha}  beta: {args.beta}  warmup: {args.warmup}  iters: {args.iters}"
    )
    print()
    results = run_mtx_batch(
        paths,
        value_dtype=value_dtype,
        index_dtype=index_dtype,
        warmup=args.warmup,
        iters=args.iters,
        k_dim=args.k,
        alpha=args.alpha,
        beta=args.beta,
        run_cusparse=run_cupy_ref,
        acc_mode=args.acc_mode,
    )
    print_mtx_results(results, value_dtype, index_dtype, args.k, args.alpha, args.beta, args.acc_mode)


if __name__ == "__main__":
    main()
