"""
SDDMM float32 diagnostic runner.

This script is intentionally separate from the general benchmark script. It is
used to diagnose float32 SDDMM behavior with several Triton kernel variants and
with an explicit float64 oracle path.
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
    run_api_validation_checks,
)
from test_spmm import load_mtx_to_csr_torch

DEFAULT_VARIANTS = ("baseline", "acc64", "acc64_out64", "altreduce")
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


def _build_case_inputs(mtx_path, k_dim, seed=None):
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


def _empty_metrics():
    return {
        "ok": None,
        "scaled_max": None,
        "max_abs": None,
        "mean_abs": None,
        "p99_scaled": None,
        "mismatch_rate": None,
    }


def _quantile99(values):
    if values.numel() == 0:
        return 0.0
    if values.numel() == 1:
        return float(values.item())
    return float(torch.quantile(values, 0.99).item())


def _compute_metrics(candidate, reference, tolerance_dtype, compare_dtype=None):
    metrics = _empty_metrics()
    if reference is None:
        return metrics
    if compare_dtype is None:
        compare_dtype = reference.dtype
    cand = candidate.to(compare_dtype)
    ref = reference.to(compare_dtype)
    if cand.numel() == 0:
        metrics.update(
            {
                "ok": True,
                "scaled_max": 0.0,
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "p99_scaled": 0.0,
                "mismatch_rate": 0.0,
            }
        )
        return metrics
    atol, rtol = ast_ops._tolerance_for_dtype(tolerance_dtype)
    diff = torch.abs(cand - ref)
    denom = atol + rtol * torch.abs(ref)
    scaled = diff / denom
    metrics.update(
        {
            "ok": bool(torch.allclose(cand, ref, atol=atol, rtol=rtol)),
            "scaled_max": float(torch.max(scaled).item()),
            "max_abs": float(torch.max(diff).item()),
            "mean_abs": float(torch.mean(diff).item()),
            "p99_scaled": _quantile99(scaled),
            "mismatch_rate": float(torch.mean((scaled > 1.0).to(torch.float32)).item()),
        }
    )
    return metrics


def _benchmark_triton_variant(prepared, data, x, y, alpha, beta, warmup, iters, variant, out_dtype=None):
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
        out_dtype=out_dtype,
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
        out_dtype=out_dtype,
    )[0]
    triton_values, triton_ms = ast_ops._benchmark_cuda_op(op, warmup=warmup, iters=iters)
    return triton_values, triton_ms, first_call_ms, meta


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


def _build_oracle_bundle(prepared, indices, indptr, data, x, y, alpha, beta, warmup, iters, run_cupy):
    data64 = data.to(torch.float64)
    x64 = x.to(torch.float64)
    y64 = y.to(torch.float64)
    ref_bundle_f64 = _benchmark_reference_bundle(
        indices=indices,
        indptr=indptr,
        x=x64,
        y=y64,
        data=data64,
        alpha=alpha,
        beta=beta,
        warmup=warmup,
        iters=iters,
        run_cupy=run_cupy,
    )
    oracle = {
        "source": "triton_f64",
        "values_f64": None,
        "values_f32": None,
        "triton_ms": None,
        "triton_first_call_ms": None,
        "meta": None,
        "pytorch_ms": ref_bundle_f64["pytorch_ms"],
        "cupy_ms": ref_bundle_f64["cupy_ms"],
        "cu_status": ref_bundle_f64["cu_status"],
        "cu_reason": ref_bundle_f64["cu_reason"],
        "ref_bundle_f64": ref_bundle_f64,
    }
    try:
        values_f64, triton_ms, first_call_ms, meta = _benchmark_triton_variant(
            prepared=prepared,
            data=data64,
            x=x64,
            y=y64,
            alpha=alpha,
            beta=beta,
            warmup=warmup,
            iters=iters,
            variant="baseline",
            out_dtype=torch.float64,
        )
        oracle["values_f64"] = values_f64
        oracle["values_f32"] = values_f64.to(torch.float32)
        oracle["triton_ms"] = triton_ms
        oracle["triton_first_call_ms"] = first_call_ms
        oracle["meta"] = meta
    except Exception:
        oracle["source"] = "pytorch_f64_ref"
        oracle["values_f64"] = ref_bundle_f64["ref"]
        oracle["values_f32"] = ref_bundle_f64["ref"].to(torch.float32)
    return oracle


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
        "err_oracle_f32": None,
        "err_oracle_f64": None,
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
        "out_dtype": None,
        "oracle_source": None,
        "reference_mode": "pt+oracle",
        "max_abs_err_pt": None,
        "mean_abs_err_pt": None,
        "p99_scaled_err_pt": None,
        "mismatch_rate_pt": None,
        "max_abs_err_oracle": None,
        "mean_abs_err_oracle": None,
        "p99_scaled_err_oracle": None,
        "mismatch_rate_oracle": None,
        "max_abs_err_oracle_f64": None,
        "mean_abs_err_oracle_f64": None,
        "p99_scaled_err_oracle_f64": None,
        "mismatch_rate_oracle_f64": None,
    }


def _attach_metrics(row, prefix, metrics, err_field, ok_field=None, status_field=None):
    row[err_field] = metrics["scaled_max"]
    row[f"max_abs_err_{prefix}"] = metrics["max_abs"]
    row[f"mean_abs_err_{prefix}"] = metrics["mean_abs"]
    row[f"p99_scaled_err_{prefix}"] = metrics["p99_scaled"]
    row[f"mismatch_rate_{prefix}"] = metrics["mismatch_rate"]
    if ok_field is not None:
        row[ok_field] = metrics["ok"]
    if status_field is not None:
        row[status_field] = "PASS" if metrics["ok"] else "FAIL"


def _populate_row_metrics(row, triton_values, ref_bundle_f32, oracle_bundle):
    row["pytorch_ms"] = ref_bundle_f32["pytorch_ms"]
    row["cupy_ms"] = ref_bundle_f32["cupy_ms"]
    row["cu_reason"] = ref_bundle_f32["cu_reason"]
    row["oracle_source"] = oracle_bundle["source"]

    pt_metrics = _compute_metrics(
        triton_values,
        ref_bundle_f32["ref"],
        tolerance_dtype=torch.float32,
        compare_dtype=torch.float32,
    )
    _attach_metrics(row, "pt", pt_metrics, "err_pt", ok_field="triton_ok_pt", status_field="status")

    cu_vals = ref_bundle_f32["cupy_values"]
    if cu_vals is None:
        row["cu_status"] = ref_bundle_f32["cu_status"]
        row["triton_ok_cu"] = None
        row["err_cu"] = None
    else:
        cu_metrics = _compute_metrics(
            triton_values,
            cu_vals,
            tolerance_dtype=torch.float32,
            compare_dtype=torch.float32,
        )
        row["triton_ok_cu"] = cu_metrics["ok"]
        row["err_cu"] = cu_metrics["scaled_max"]
        row["cu_status"] = "PASS" if cu_metrics["ok"] else "FAIL"

    oracle_f32_metrics = _compute_metrics(
        triton_values,
        oracle_bundle["values_f32"],
        tolerance_dtype=torch.float32,
        compare_dtype=torch.float32,
    )
    _attach_metrics(row, "oracle", oracle_f32_metrics, "err_oracle_f32")

    oracle_f64_metrics = _compute_metrics(
        triton_values,
        oracle_bundle["values_f64"],
        tolerance_dtype=torch.float64,
        compare_dtype=torch.float64,
    )
    row["err_oracle_f64"] = oracle_f64_metrics["scaled_max"]
    row["max_abs_err_oracle_f64"] = oracle_f64_metrics["max_abs"]
    row["mean_abs_err_oracle_f64"] = oracle_f64_metrics["mean_abs"]
    row["p99_scaled_err_oracle_f64"] = oracle_f64_metrics["p99_scaled"]
    row["mismatch_rate_oracle_f64"] = oracle_f64_metrics["mismatch_rate"]


def _make_oracle_row(mtx_path, base, k_dim, alpha, beta, prepare_ms, seed, oracle_bundle):
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
    row["oracle_source"] = oracle_bundle["source"]
    row["reference_mode"] = "oracle_f64_self"
    row["pytorch_ms"] = oracle_bundle["pytorch_ms"]
    row["cupy_ms"] = oracle_bundle["cupy_ms"]
    row["cu_reason"] = oracle_bundle["cu_reason"]
    row["triton_ms"] = oracle_bundle["triton_ms"]
    row["triton_first_call_ms"] = oracle_bundle["triton_first_call_ms"]
    row["status"] = "PASS"
    row["triton_ok_pt"] = True
    row["triton_ok_cu"] = True if oracle_bundle["cu_status"] == "READY" else None
    row["cu_status"] = "PASS" if oracle_bundle["cu_status"] == "READY" else oracle_bundle["cu_status"]
    row["err_pt"] = 0.0
    row["err_cu"] = 0.0 if oracle_bundle["cu_status"] == "READY" else None
    row["err_oracle_f32"] = 0.0
    row["err_oracle_f64"] = 0.0
    row["max_abs_err_pt"] = 0.0
    row["mean_abs_err_pt"] = 0.0
    row["p99_scaled_err_pt"] = 0.0
    row["mismatch_rate_pt"] = 0.0
    row["max_abs_err_oracle"] = 0.0
    row["mean_abs_err_oracle"] = 0.0
    row["p99_scaled_err_oracle"] = 0.0
    row["mismatch_rate_oracle"] = 0.0
    row["max_abs_err_oracle_f64"] = 0.0
    row["mean_abs_err_oracle_f64"] = 0.0
    row["p99_scaled_err_oracle_f64"] = 0.0
    row["mismatch_rate_oracle_f64"] = 0.0
    row["fallback_used"] = False
    meta = oracle_bundle["meta"] or {}
    row["block_k"] = meta.get("block_k")
    row["num_warps"] = meta.get("num_warps")
    row["acc_dtype"] = meta.get("acc_dtype", "float64")
    row["out_dtype"] = meta.get("out_dtype", "float64")
    return row


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
    base = _build_case_inputs(mtx_path, k_dim=k_dim, seed=seed)
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
    oracle_bundle = _build_oracle_bundle(
        prepared=prepared,
        indices=base["indices"],
        indptr=base["indptr"],
        data=base["data"],
        x=base["x"],
        y=base["y"],
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
            triton_values, triton_ms, first_call_ms, meta = _benchmark_triton_variant(
                prepared=prepared,
                data=base["data"],
                x=base["x"],
                y=base["y"],
                alpha=alpha,
                beta=beta,
                warmup=warmup,
                iters=iters,
                variant=variant,
                out_dtype=None,
            )
            row["triton_ms"] = triton_ms
            row["triton_first_call_ms"] = first_call_ms
            row["fallback_used"] = bool(meta.get("fallback_used", False))
            row["block_k"] = meta.get("block_k")
            row["num_warps"] = meta.get("num_warps")
            row["acc_dtype"] = meta.get("acc_dtype")
            row["out_dtype"] = meta.get("out_dtype", _dtype_name(triton_values.dtype))
            _populate_row_metrics(row, triton_values, ref_bundle_f32, oracle_bundle)
        except Exception as exc:
            row["error"] = f"triton: {exc}"
            row["status"] = "TRITON_FAIL"
            row["pytorch_ms"] = ref_bundle_f32.get("pytorch_ms")
            row["cupy_ms"] = ref_bundle_f32.get("cupy_ms")
            row["cu_status"] = ref_bundle_f32.get("cu_status", "REF_UNAVAILABLE")
            row["cu_reason"] = ref_bundle_f32.get("cu_reason")
            row["oracle_source"] = oracle_bundle["source"]
        rows.append(row)

    if include_f64_ref:
        rows.append(
            _make_oracle_row(
                mtx_path=mtx_path,
                base=base,
                k_dim=k_dim,
                alpha=alpha,
                beta=beta,
                prepare_ms=prepare_ms,
                seed=seed,
                oracle_bundle=oracle_bundle,
            )
        )

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
            f"{row['matrix'][:22]:<22} {row['variant']:<12} {row['acc_dtype'] or 'N/A':<8} {row['out_dtype'] or 'N/A':<8} "
            f"{row['k']:>5} {_fmt_ms(row.get('triton_ms')):>11} {_fmt_err(row.get('err_pt')):>10} "
            f"{_fmt_err(row.get('err_oracle_f32')):>12} {_fmt_err(row.get('err_oracle_f64')):>12} "
            f"{_fmt_check(row.get('triton_ok_pt')):>6} {str(row.get('cu_status')):>12}"
        )
        if row.get("error"):
            print(f"  NOTE: {str(row['error']).replace(chr(10), ' ')}")
        if row.get("cu_reason") and row.get("cu_status") not in ("PASS", "FAIL"):
            print(f"  CU_NOTE: {str(row['cu_reason']).replace(chr(10), ' ')}")


def _summary_oracle_metrics(row):
    if row.get("out_dtype") == "float64":
        return {
            "err": row.get("err_oracle_f64"),
            "mismatch": row.get("mismatch_rate_oracle_f64"),
            "p99": row.get("p99_scaled_err_oracle_f64"),
        }
    return {
        "err": row.get("err_oracle_f32"),
        "mismatch": row.get("mismatch_rate_oracle"),
        "p99": row.get("p99_scaled_err_oracle"),
    }


def _print_summary(rows, oracle_only_summary=False):
    print("-" * 148)
    print("按变体汇总")
    print("-" * 148)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    for variant in sorted(grouped):
        items = grouped[variant]
        pass_count = sum(1 for row in items if row.get("status") == "PASS")
        valid_ms = [row["triton_ms"] for row in items if row.get("triton_ms") is not None]
        oracle_errs = []
        oracle_mismatch = []
        oracle_p99 = []
        for row in items:
            metrics = _summary_oracle_metrics(row)
            if metrics["err"] is not None:
                oracle_errs.append(metrics["err"])
            if metrics["mismatch"] is not None:
                oracle_mismatch.append(metrics["mismatch"])
            if metrics["p99"] is not None:
                oracle_p99.append(metrics["p99"])
        avg_ms = sum(valid_ms) / len(valid_ms) if valid_ms else None
        avg_err = sum(oracle_errs) / len(oracle_errs) if oracle_errs else None
        avg_mismatch = sum(oracle_mismatch) / len(oracle_mismatch) if oracle_mismatch else None
        max_p99 = max(oracle_p99) if oracle_p99 else None
        if oracle_only_summary:
            print(
                f"{variant:<12} avg_triton_ms={_fmt_ms(avg_ms):>10}  avg_err_oracle={_fmt_err(avg_err):>10}  "
                f"avg_mismatch_rate={avg_mismatch if avg_mismatch is not None else float('nan'):.4f}  "
                f"max_p99_scaled={_fmt_err(max_p99):>10}"
            )
        else:
            print(
                f"{variant:<12} pass={pass_count:>4}/{len(items):<4}  avg_triton_ms={_fmt_ms(avg_ms):>10}  "
                f"avg_err_oracle_f32={_fmt_err(avg_err):>10}  avg_mismatch_rate_oracle={avg_mismatch if avg_mismatch is not None else float('nan'):.4f}  "
                f"max_p99_scaled_err_oracle={_fmt_err(max_p99):>10}"
            )
    print("-" * 148)


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
        "reference_mode",
        "alpha",
        "beta",
        "triton_ms",
        "triton_first_call_ms",
        "prepare_ms",
        "pytorch_ms",
        "cupy_ms",
        "err_pt",
        "err_cu",
        "err_oracle_f32",
        "err_oracle_f64",
        "triton_ok_pt",
        "triton_ok_cu",
        "status",
        "cu_status",
        "fallback_used",
        "block_k",
        "num_warps",
        "acc_dtype",
        "out_dtype",
        "oracle_source",
        "max_abs_err_pt",
        "mean_abs_err_pt",
        "p99_scaled_err_pt",
        "mismatch_rate_pt",
        "max_abs_err_oracle",
        "mean_abs_err_oracle",
        "p99_scaled_err_oracle",
        "mismatch_rate_oracle",
        "max_abs_err_oracle_f64",
        "mean_abs_err_oracle_f64",
        "p99_scaled_err_oracle_f64",
        "mismatch_rate_oracle_f64",
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
    parser.add_argument("--variants", type=str, default="baseline,acc64,acc64_out64,altreduce")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv", type=str, default=None, metavar="FILE")
    parser.add_argument("--include-f64-ref", action="store_true")
    parser.add_argument("--oracle-only-summary", action="store_true")
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

    print("=" * 148)
    print("FLAGSPARSE SDDMM - float32 diagnostic variants with float64 oracle")
    print("=" * 148)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  "
        f"K: {args.k}  |  alpha: {args.alpha}  |  beta: {args.beta}  |  variants: {','.join(variants)}"
    )
    if args.seed is not None:
        print(f"Seed base: {args.seed}")
    if args.include_f64_ref:
        print("Extra row: enabled float64 oracle row (f64_ref)")
    print("-" * 148)
    print(
        f"{'Matrix':<22} {'Variant':<12} {'Acc':<8} {'Out':<8} {'K':>5} "
        f"{'Triton(ms)':>11} {'Err(PT)':>10} {'Err(OrF32)':>12} {'Err(OrF64)':>12} {'PT':>6} {'CU_Status':>12}"
    )
    print("-" * 148)

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
    _print_summary(rows, oracle_only_summary=args.oracle_only_summary)

    if args.csv is not None:
        _write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
