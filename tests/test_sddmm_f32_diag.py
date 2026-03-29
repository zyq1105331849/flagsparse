"""
SDDMM float32 diagnostic runner.

This script isolates three Triton kernel variants for CSR SDDMM:
1. baseline   : current float32 accumulate path
2. acc64      : float32 input/output with float64 accumulation
3. altreduce  : float32 accumulate with explicit step-by-step reduction

It is intentionally separate from the general benchmark script so that the
generated CSV stays diagnostic-only and does not mix with normal benchmark data.
"""

import argparse
import csv
import glob
import os
import sys
import time
from collections import defaultdict
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
from test_sddmm import (
    _benchmark_cupy_sampled_reference,
    _expand_mtx_paths,
    _fmt_check,
    _fmt_err,
    _fmt_ms,
    _is_resource_error,
    _normalize_csv_path,
    _scaled_allclose_error,
    run_api_validation_checks,
)
from test_spmm import load_mtx_to_csr_torch

DEFAULT_VARIANTS = ("baseline", "acc64", "altreduce")
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
DEFAULT_K = 64


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _parse_variants(raw_variants):
    if raw_variants is None:
        return list(DEFAULT_VARIANTS)
    variants = []
    for token in str(raw_variants).split(","):
        name = ast_ops._normalize_sddmm_diagnostic_variant(token)
        if name not in variants:
            variants.append(name)
    if not variants:
        raise ValueError("--variants must contain at least one variant")
    return variants


def _make_cuda_generator(device, seed):
    if seed is None:
        return None
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _build_case_inputs(mtx_path, k_dim, alpha, beta, seed=None):
    del alpha, beta
    device = torch.device("cuda")
    _pattern_values, indices, indptr, shape = load_mtx_to_csr_torch(
        mtx_path,
        dtype=torch.float32,
        device=device,
    )
    indices = indices.to(torch.int32)
    n_rows, n_cols = shape
    nnz = int(indices.numel())
    generator = _make_cuda_generator(device, seed)
    rand_kwargs = {"generator": generator} if generator is not None else {}
    data = torch.randn(nnz, dtype=torch.float32, device=device, **rand_kwargs)
    x = torch.randn((n_rows, k_dim), dtype=torch.float32, device=device, **rand_kwargs)
    y = torch.randn((n_cols, k_dim), dtype=torch.float32, device=device, **rand_kwargs)
    return {
        "indices": indices,
        "indptr": indptr,
        "shape": shape,
        "nnz": nnz,
        "data": data,
        "x": x,
        "y": y,
    }


def _prepare_pattern(indices, indptr, shape, k_dim):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prepared = ast.prepare_sddmm_csr(indices, indptr, shape, k_hint=int(k_dim))
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - t0) * 1000.0
    return prepared, prepare_ms


def _benchmark_triton_variant(prepared, data, x, y, alpha, beta, warmup, iters, variant):
    x_in = x.contiguous()
    y_in = y.contiguous()
    data_in = data.contiguous() if data is not None else None
    torch.cuda.synchronize()
    t_first0 = time.perf_counter()
    out_first, meta = ast_ops._run_sddmm_prepared(
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
    triton_values, triton_ms = ast_ops._benchmark_cuda_op(op, warmup=warmup, iters=iters)
    return triton_values, triton_ms, first_call_ms, meta, out_first


def _benchmark_reference_bundle(indices, indptr, x, y, data, alpha, beta, warmup, iters, run_cupy):
    result = {
        "ref": None,
        "pytorch_ms": None,
        "cupy_values": None,
        "cupy_ms": None,
        "cu_status": "REF_UNAVAILABLE",
        "cu_reason": None,
    }
    ref = ast_ops._sddmm_reference(indices, indptr.to(torch.int64), x, y, data, alpha, beta)
    _, pytorch_ms = ast_ops._benchmark_cuda_op(
        lambda: ast_ops._sddmm_reference(indices, indptr.to(torch.int64), x, y, data, alpha, beta),
        warmup=warmup,
        iters=iters,
    )
    result["ref"] = ref
    result["pytorch_ms"] = pytorch_ms

    if not run_cupy:
        result["cu_reason"] = "CuPy reference is disabled by CLI"
        return result
    if ast_ops.cp is None:
        result["cu_reason"] = "CuPy is not available"
        return result
    try:
        cu_vals, cupy_ms = _benchmark_cupy_sampled_reference(
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
        result["cupy_values"] = cu_vals
        result["cupy_ms"] = cupy_ms
        result["cu_status"] = "READY"
    except Exception as exc:
        result["cu_status"] = "REF_RESOURCE" if _is_resource_error(exc) else "REF_UNAVAILABLE"
        result["cu_reason"] = str(exc)
    return result


def _make_result_row(mtx_path, shape, nnz, k_dim, alpha, beta, dtype, variant, seed):
    return {
        "matrix": os.path.basename(mtx_path),
        "path": os.path.abspath(mtx_path),
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "shape": f"{int(shape[0])}x{int(shape[1])}",
        "nnz": int(nnz),
        "k": int(k_dim),
        "dtype": _dtype_name(dtype),
        "variant": variant,
        "alpha": float(alpha),
        "beta": float(beta),
        "triton_ms": None,
        "triton_first_call_ms": None,
        "prepare_ms": None,
        "pytorch_ms": None,
        "cupy_ms": None,
        "err_pt": None,
        "err_cu": None,
        "triton_ok_pt": None,
        "triton_ok_cu": None,
        "status": "UNKNOWN",
        "cu_status": "REF_UNAVAILABLE",
        "fallback_used": False,
        "block_k": None,
        "num_warps": None,
        "error": None,
        "cu_reason": None,
        "seed": "" if seed is None else int(seed),
        "acc_dtype": None,
    }


def _populate_match_fields(row, triton_values, ref_bundle, value_dtype):
    atol, rtol = ast_ops._tolerance_for_dtype(value_dtype)
    ref = ref_bundle["ref"]
    row["triton_ok_pt"] = bool(torch.allclose(triton_values, ref, atol=atol, rtol=rtol))
    row["err_pt"] = _scaled_allclose_error(triton_values, ref, value_dtype)
    row["pytorch_ms"] = ref_bundle["pytorch_ms"]
    row["status"] = "PASS" if row["triton_ok_pt"] else "FAIL"

    cu_vals = ref_bundle["cupy_values"]
    row["cupy_ms"] = ref_bundle["cupy_ms"]
    row["cu_reason"] = ref_bundle["cu_reason"]
    if cu_vals is None:
        row["cu_status"] = ref_bundle["cu_status"]
        row["triton_ok_cu"] = None
        row["err_cu"] = None
        return
    row["triton_ok_cu"] = bool(torch.allclose(triton_values, cu_vals, atol=atol, rtol=rtol))
    row["err_cu"] = _scaled_allclose_error(triton_values, cu_vals, value_dtype)
    row["cu_status"] = "PASS" if row["triton_ok_cu"] else "FAIL"


def run_one_mtx_diag(
    mtx_path,
    variants,
    warmup=DEFAULT_WARMUP,
    iters=DEFAULT_ITERS,
    k_dim=DEFAULT_K,
    alpha=1.0,
    beta=0.0,
    run_cupy=True,
    seed=None,
    include_f64_ref=False,
):
    base = _build_case_inputs(mtx_path, k_dim=k_dim, alpha=alpha, beta=beta, seed=seed)
    prepared, prepare_ms = _prepare_pattern(base["indices"], base["indptr"], base["shape"], k_dim)
    ref_bundle_f32 = _benchmark_reference_bundle(
        indices=base["indices"],
        indptr=base["indptr"],
        x=base["x"],
        y=base["y"],
        data=base["data"],
        alpha=alpha,
        beta=beta,
        warmup=warmup,
        iters=iters,
        run_cupy=run_cupy,
    )

    rows = []
    for variant in variants:
        row = _make_result_row(
            mtx_path=mtx_path,
            shape=base["shape"],
            nnz=base["nnz"],
            k_dim=k_dim,
            alpha=alpha,
            beta=beta,
            dtype=torch.float32,
            variant=variant,
            seed=seed,
        )
        row["prepare_ms"] = prepare_ms
        try:
            triton_values, triton_ms, first_call_ms, meta, _out_first = _benchmark_triton_variant(
                prepared=prepared,
                data=base["data"],
                x=base["x"],
                y=base["y"],
                alpha=alpha,
                beta=beta,
                warmup=warmup,
                iters=iters,
                variant=variant,
            )
            row["triton_ms"] = triton_ms
            row["triton_first_call_ms"] = first_call_ms
            row["fallback_used"] = bool(meta.get("fallback_used", False))
            row["block_k"] = meta.get("block_k")
            row["num_warps"] = meta.get("num_warps")
            row["acc_dtype"] = meta.get("acc_dtype")
            _populate_match_fields(row, triton_values, ref_bundle_f32, torch.float32)
        except Exception as exc:
            row["error"] = f"triton: {exc}"
            row["status"] = "TRITON_FAIL"
            row["cu_status"] = ref_bundle_f32.get("cu_status", "REF_UNAVAILABLE")
            row["pytorch_ms"] = ref_bundle_f32.get("pytorch_ms")
            row["cupy_ms"] = ref_bundle_f32.get("cupy_ms")
            row["cu_reason"] = ref_bundle_f32.get("cu_reason")
        rows.append(row)

    if include_f64_ref:
        data64 = base["data"].to(torch.float64)
        x64 = base["x"].to(torch.float64)
        y64 = base["y"].to(torch.float64)
        ref_bundle_f64 = _benchmark_reference_bundle(
            indices=base["indices"],
            indptr=base["indptr"],
            x=x64,
            y=y64,
            data=data64,
            alpha=alpha,
            beta=beta,
            warmup=warmup,
            iters=iters,
            run_cupy=run_cupy,
        )
        row = _make_result_row(
            mtx_path=mtx_path,
            shape=base["shape"],
            nnz=base["nnz"],
            k_dim=k_dim,
            alpha=alpha,
            beta=beta,
            dtype=torch.float64,
            variant="f64_ref",
            seed=seed,
        )
        row["prepare_ms"] = prepare_ms
        try:
            triton_values, triton_ms, first_call_ms, meta, _out_first = _benchmark_triton_variant(
                prepared=prepared,
                data=data64,
                x=x64,
                y=y64,
                alpha=alpha,
                beta=beta,
                warmup=warmup,
                iters=iters,
                variant="baseline",
            )
            row["triton_ms"] = triton_ms
            row["triton_first_call_ms"] = first_call_ms
            row["fallback_used"] = bool(meta.get("fallback_used", False))
            row["block_k"] = meta.get("block_k")
            row["num_warps"] = meta.get("num_warps")
            row["acc_dtype"] = meta.get("acc_dtype")
            _populate_match_fields(row, triton_values, ref_bundle_f64, torch.float64)
        except Exception as exc:
            row["error"] = f"triton: {exc}"
            row["status"] = "TRITON_FAIL"
            row["cu_status"] = ref_bundle_f64.get("cu_status", "REF_UNAVAILABLE")
            row["pytorch_ms"] = ref_bundle_f64.get("pytorch_ms")
            row["cupy_ms"] = ref_bundle_f64.get("cupy_ms")
            row["cu_reason"] = ref_bundle_f64.get("cu_reason")
        rows.append(row)

    return rows


def run_mtx_batch_diag(
    mtx_paths,
    variants,
    warmup=DEFAULT_WARMUP,
    iters=DEFAULT_ITERS,
    k_dim=DEFAULT_K,
    alpha=1.0,
    beta=0.0,
    run_cupy=True,
    seed=None,
    include_f64_ref=False,
    on_rows=None,
):
    rows = []
    for idx, path in enumerate(mtx_paths):
        row_seed = None if seed is None else int(seed) + idx
        case_rows = run_one_mtx_diag(
            mtx_path=path,
            variants=variants,
            warmup=warmup,
            iters=iters,
            k_dim=k_dim,
            alpha=alpha,
            beta=beta,
            run_cupy=run_cupy,
            seed=row_seed,
            include_f64_ref=include_f64_ref,
        )
        rows.extend(case_rows)
        if on_rows is not None:
            on_rows(case_rows)
    return rows


def _print_case_rows(rows):
    for row in rows:
        print(
            f"{row['matrix'][:26]:<26} {row['variant']:<10} {row['dtype']:<8} "
            f"{row['k']:>5} {_fmt_ms(row.get('triton_ms')):>12} {_fmt_ms(row.get('pytorch_ms')):>12} "
            f"{_fmt_ms(row.get('cupy_ms')):>12} {_fmt_check(row.get('triton_ok_pt')):>8} "
            f"{str(row.get('cu_status')):>12} {_fmt_err(row.get('err_pt')):>10} {_fmt_err(row.get('err_cu')):>10}"
        )
        if row.get("error"):
            print(f"  NOTE: {str(row['error']).replace(chr(10), ' ')}")
        if row.get("cu_reason") and row.get("cu_status") not in ("PASS", "FAIL"):
            print(f"  CU_NOTE: {str(row['cu_reason']).replace(chr(10), ' ')}")


def _print_summary(rows):
    print("-" * 132)
    print("按变体汇总")
    print("-" * 132)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    for variant in sorted(grouped):
        items = grouped[variant]
        pass_count = sum(1 for row in items if row.get("status") == "PASS")
        fail_count = sum(1 for row in items if row.get("status") not in ("PASS",))
        valid_ms = [row["triton_ms"] for row in items if row.get("triton_ms") is not None]
        valid_err = [row["err_pt"] for row in items if row.get("err_pt") is not None]
        avg_ms = sum(valid_ms) / len(valid_ms) if valid_ms else None
        max_err = max(valid_err) if valid_err else None
        print(
            f"{variant:<10} total={len(items):>4}  pass={pass_count:>4}  fail={fail_count:>4}  "
            f"avg_triton_ms={_fmt_ms(avg_ms):>10}  max_err_pt={_fmt_err(max_err):>10}"
        )
    print("-" * 132)


def _write_csv(rows, csv_path):
    csv_path = _normalize_csv_path(csv_path)
    fieldnames = [
        "matrix",
        "path",
        "n_rows",
        "n_cols",
        "shape",
        "nnz",
        "k",
        "dtype",
        "variant",
        "alpha",
        "beta",
        "triton_ms",
        "triton_first_call_ms",
        "prepare_ms",
        "pytorch_ms",
        "cupy_ms",
        "err_pt",
        "err_cu",
        "triton_ok_pt",
        "triton_ok_cu",
        "status",
        "cu_status",
        "fallback_used",
        "block_k",
        "num_warps",
        "acc_dtype",
        "seed",
        "error",
        "cu_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    print(f"Wrote {len(rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="FlagSparse SDDMM float32 diagnostic runner")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Dense feature dimension K")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--variants", type=str, default="baseline,acc64,altreduce")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv", type=str, default=None, metavar="FILE")
    parser.add_argument("--include-f64-ref", action="store_true")
    parser.add_argument("--no-cupy-ref", action="store_true", help="Skip CuPy sampled-dot reference baseline")
    parser.add_argument(
        "--no-cusparse",
        action="store_true",
        help="Deprecated alias of --no-cupy-ref",
    )
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

    variants = _parse_variants(args.variants)
    run_cupy_ref = not (args.no_cupy_ref or args.no_cusparse)
    paths = _expand_mtx_paths(args.mtx)
    if not paths and not args.csv:
        print("No .mtx files given. Use: python test_sddmm_f32_diag.py <file.mtx> [file2.mtx ...] or <dir/>")
        print("Or export CSV: python test_sddmm_f32_diag.py <dir/> --k 64 --csv results_f32_diag.csv")
        return
    if args.csv is not None and not paths:
        paths = sorted(glob.glob("*.mtx"))
    if not paths:
        print("No .mtx files found. Specify files or a directory.")
        return

    print("=" * 132)
    print("FLAGSPARSE SDDMM - float32 diagnostic variants")
    print("=" * 132)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  "
        f"K: {args.k}  |  alpha: {args.alpha}  |  beta: {args.beta}  |  variants: {','.join(variants)}"
    )
    if args.seed is not None:
        print(f"Seed base: {args.seed}")
    if args.include_f64_ref:
        print("Extra row: enabled float64 reference variant (f64_ref)")
    print("-" * 132)
    print(
        f"{'Matrix':<26} {'Variant':<10} {'DType':<8} {'K':>5} "
        f"{'Triton(ms)':>12} {'PyTorch(ms)':>12} {'CuPy(ms)':>12} {'PT':>8} {'CU_Status':>12} "
        f"{'Err(PT)':>10} {'Err(CU)':>10}"
    )
    print("-" * 132)

    rows = run_mtx_batch_diag(
        mtx_paths=paths,
        variants=variants,
        warmup=args.warmup,
        iters=args.iters,
        k_dim=args.k,
        alpha=args.alpha,
        beta=args.beta,
        run_cupy=run_cupy_ref,
        seed=args.seed,
        include_f64_ref=args.include_f64_ref,
        on_rows=_print_case_rows,
    )
    _print_summary(rows)

    if args.csv is not None:
        _write_csv(rows, args.csv)


if __name__ == "__main__":
    main()

