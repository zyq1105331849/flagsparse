"""
SpMV tests (CSR): load SuiteSparse .mtx, batch run, output error and performance.
Supports: multi .mtx files, value_dtype / index_dtype, --csv-csr to run all dtypes and export CSV.
"""
import argparse
import csv
import glob
import math
import os

import torch
import flagsparse as ast

VALUE_DTYPES = [
    torch.float32,
    torch.float64,
]
INDEX_DTYPES = [torch.int32]
TEST_CASES = [
    (512, 512, 4096),
    (1024, 1024, 16384),
    (2048, 2048, 65536),
    (4096, 4096, 131072),
]
WARMUP = 10
ITERS = 50


def load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    """
    Load SuiteSparse / Matrix Market .mtx file into CSR as torch tensors.
    Correctly handles *pattern* matrices and *symmetric/skew-symmetric* expansions.
    Returns (data, indices, indptr, shape) on device.
    """
    import math as _math
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    mm_field = "real"
    mm_symmetry = "general"
    data_lines = []
    header_info = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%%MatrixMarket"):
            tokens = stripped.split()
            if len(tokens) >= 5:
                mm_field = tokens[3].lower()
                mm_symmetry = tokens[4].lower()
            continue
        if stripped.startswith("%"):
            continue
        if not header_info and stripped:
            parts = stripped.split()
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            header_info = (n_rows, n_cols, nnz)
            continue
        if stripped:
            data_lines.append(stripped)
    if header_info is None:
        raise ValueError(f"Cannot parse .mtx header: {file_path}")
    n_rows, n_cols, nnz = header_info
    if nnz == 0:
        data = torch.tensor([], dtype=dtype, device=device)
        indices = torch.tensor([], dtype=torch.int64, device=device)
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
        return data, indices, indptr, (n_rows, n_cols)

    is_pattern = (mm_field == "pattern")
    is_symmetric = mm_symmetry in ("symmetric", "hermitian")
    is_skew = (mm_symmetry == "skew-symmetric")

    row_maps = [dict() for _ in range(n_rows)]
    for line in data_lines[:nnz]:
        parts = line.split()
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        v = 1.0 if is_pattern else float(parts[2])
        if 0 <= r < n_rows and 0 <= c < n_cols:
            row_maps[r][c] = row_maps[r].get(c, 0.0) + v
            if r != c:
                if is_symmetric and 0 <= c < n_rows and 0 <= r < n_cols:
                    row_maps[c][r] = row_maps[c].get(r, 0.0) + v
                elif is_skew and 0 <= c < n_rows and 0 <= r < n_cols:
                    row_maps[c][r] = row_maps[c].get(r, 0.0) - v

    cols_s = []
    vals_s = []
    indptr_list = [0]
    for r in range(n_rows):
        row = row_maps[r]
        for c in sorted(row.keys()):
            cols_s.append(c)
            vals_s.append(row[c])
        indptr_list.append(len(cols_s))
    data = torch.tensor(vals_s, dtype=dtype, device=device)
    indices = torch.tensor(cols_s, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def _allclose_error_ratio(actual, reference, atol, rtol):
    if actual.numel() == 0:
        return 0.0
    diff = torch.abs(actual - reference).to(torch.float64)
    tol = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / tol).item())


def _benchmark_flagsparse_spmv(data, indices, indptr, x, shape, warmup, iters, block_nnz, max_segments):
    prepared = ast.prepare_spmv_csr(
        data,
        indices,
        indptr,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )
    op = lambda: ast.flagsparse_spmv_csr(
        x=x,
        prepared=prepared,
        return_time=False,
    )
    y = op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = op()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        y = op()
    end_ev.record()
    torch.cuda.synchronize()
    return y, start_ev.elapsed_time(end_ev) / iters


def _reference_dtype(dtype):
    return torch.float64 if dtype == torch.float32 else dtype


def _pytorch_spmv_reference(data, indices, indptr, x, shape, out_dtype):
    device = data.device
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    try:
        csr_ref = torch.sparse_csr_tensor(
            indptr.to(torch.int64),
            indices.to(torch.int64),
            data_ref,
            size=shape,
            device=device,
        )
        y_ref = torch.sparse.mm(csr_ref, x_ref.unsqueeze(1)).squeeze(1)
    except Exception:
        n_rows = int(shape[0])
        row_ind = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        coo_ref = torch.sparse_coo_tensor(
            torch.stack([row_ind, indices.to(torch.int64)]),
            data_ref,
            shape,
            device=device,
        ).coalesce()
        y_ref = torch.sparse.mm(coo_ref, x_ref.unsqueeze(1)).squeeze(1)
    return y_ref.to(out_dtype) if ref_dtype != out_dtype else y_ref


def _cupy_csr_reference(data, indices, indptr, x, shape, out_dtype):
    import cupy as cp
    import cupyx.scipy.sparse as cpx

    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_ref))
    A_csr_ref = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    y_ref = A_csr_ref @ x_cp
    y_ref_t = torch.utils.dlpack.from_dlpack(y_ref.toDlpack())
    return y_ref_t.to(out_dtype) if ref_dtype != out_dtype else y_ref_t


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    block_nnz=256,
    max_segments=None,
):
    """Run SpMV on one .mtx: load, compute ref, run Triton and optional cuSPARSE, return errors and timings."""
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(mtx_path, dtype=value_dtype, device=device)
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz = data.numel()
    x = torch.randn(n_cols, dtype=value_dtype, device=device)
    atol, rtol = 1e-6, 1e-5
    if value_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 2e-3, 2e-3
    elif value_dtype == torch.float32 or value_dtype == torch.complex64:
        # Relaxed for float32: reduction order differs from PyTorch/cuSPARSE on some irregular matrices.
        atol, rtol = 1.25e-4, 1.25e-2
    elif value_dtype == torch.float64 or value_dtype == torch.complex128:
        atol, rtol = 1e-12, 1e-10
    triton_y, triton_ms = _benchmark_flagsparse_spmv(
        data,
        indices,
        indptr,
        x,
        shape,
        warmup=warmup,
        iters=iters,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )
    pt_y = None
    pt_ref_y = None
    pytorch_ms = None
    err_pt = None
    triton_ok_pt = False
    try:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        try:
            csr_pt = torch.sparse_csr_tensor(
                indptr.to(torch.int64),
                indices.to(torch.int64),
                data,
                size=shape,
                device=device,
            )
            pt_y = torch.sparse.mm(csr_pt, x.unsqueeze(1)).squeeze(1)
            torch.cuda.synchronize()
            for _ in range(warmup):
                _ = torch.sparse.mm(csr_pt, x.unsqueeze(1))
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(iters):
                _ = torch.sparse.mm(csr_pt, x.unsqueeze(1))
            end_ev.record()
        except Exception:
            row_ind = torch.repeat_interleave(
                torch.arange(n_rows, device=device, dtype=torch.int64),
                indptr[1:] - indptr[:-1],
            )
            coo = torch.sparse_coo_tensor(
                torch.stack([row_ind, indices.to(torch.int64)]),
                data,
                shape,
                device=device,
            ).coalesce()
            pt_y = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)
            torch.cuda.synchronize()
            for _ in range(warmup):
                _ = torch.sparse.mm(coo, x.unsqueeze(1))
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(iters):
                _ = torch.sparse.mm(coo, x.unsqueeze(1))
            end_ev.record()
        if pt_y is not None:
            try:
                pt_ref_y = _pytorch_spmv_reference(
                    data, indices, indptr, x, shape, value_dtype
                )
            except Exception:
                pt_ref_y = pt_y
        if pt_ref_y is not None and n_rows:
            err_pt = _allclose_error_ratio(triton_y, pt_ref_y, atol, rtol)
            triton_ok_pt = (not math.isnan(err_pt)) and err_pt <= 1.0
        torch.cuda.synchronize()
        pytorch_ms = start_ev.elapsed_time(end_ev) / iters
    except Exception:
        pytorch_ms = None
    cs_y_t = None
    cs_ref_t = None
    cusparse_ms = None
    err_cu = None
    triton_ok_cu = False
    csc_ms = None
    if run_cusparse and value_dtype not in (torch.bfloat16,):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx
            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
            ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
            ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
            x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
            A_csr = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
            torch.cuda.synchronize()
            for _ in range(warmup):
                _ = A_csr @ x_cp
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = A_csr @ x_cp
            end.record()
            torch.cuda.synchronize()
            cusparse_ms = start.elapsed_time(end) / iters
            cs_y = A_csr @ x_cp
            cs_y_t = torch.utils.dlpack.from_dlpack(cs_y.toDlpack())
            try:
                cs_ref_t = _cupy_csr_reference(
                    data, indices, indptr, x, shape, value_dtype
                )
            except Exception:
                cs_ref_t = cs_y_t
            if cs_ref_t is not None and n_rows:
                err_cu = _allclose_error_ratio(triton_y, cs_ref_t, atol, rtol)
                triton_ok_cu = (not math.isnan(err_cu)) and err_cu <= 1.0
            A_csc = A_csr.tocsc()
            torch.cuda.synchronize()
            for _ in range(warmup):
                _ = A_csc @ x_cp
            torch.cuda.synchronize()
            start.record()
            for _ in range(iters):
                _ = A_csc @ x_cp
            end.record()
            torch.cuda.synchronize()
            csc_ms = start.elapsed_time(end) / iters
        except Exception:
            cusparse_ms = None
            err_cu = None
            csc_ms = None
    if pt_y is None and err_cu is None:
        return {
            "path": mtx_path,
            "shape": shape,
            "nnz": nnz,
            "error": "ref: no PyTorch or cuSPARSE result",
            "triton_ms": triton_ms,
            "cusparse_ms": None,
            "pytorch_ms": None,
            "csc_ms": None,
            "err_pt": None,
            "err_cu": None,
            "triton_ok_pt": False,
            "triton_ok_cu": False,
            "status": "REF_FAIL",
        }
    status = "PASS" if (triton_ok_pt or triton_ok_cu) else "FAIL"
    return {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz,
        "error": None,
        "triton_ms": triton_ms,
        "cusparse_ms": cusparse_ms,
        "pytorch_ms": pytorch_ms,
        "csc_ms": csc_ms,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "triton_ok_pt": triton_ok_pt,
        "triton_ok_cu": triton_ok_cu,
        "status": status,
    }


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    on_result=None,
):
    """Batch run SpMV on multiple .mtx files; return list of result dicts."""
    results = []
    for path in mtx_paths:
        r = run_one_mtx(
            path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=warmup,
            iters=iters,
            run_cusparse=run_cusparse,
        )
        results.append(r)
        if on_result is not None:
            on_result(r)
    return results


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_speedup(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return "N/A"
    return f"{other_ms / triton_ms:.2f}x"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _status_str(ok, available):
    if not available:
        return "N/A"
    return "PASS" if ok else "FAIL"


def _print_mtx_header(value_dtype, index_dtype):
    print(
        f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}"
    )
    print("Formats: FlagSparse=CSR, cuSPARSE=CSR/CSC, PyTorch=CSR or COO.")
    print("Timing stays in native dtype. For float32, correctness references use float64 compute then cast.")
    print("PT/CU show per-reference correctness. Err(PT)/Err(CU)=max(|diff| / (atol + rtol*|ref|)).")
    print("-" * 150)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
        f"{'FlagSparse(ms)':>10} {'CSR(ms)':>10} {'CSC(ms)':>10} {'PyTorch(ms)':>11} "
        f"{'FS/CSR':>7} {'FS/PT':>7} {'PT':>6} {'CU':>6} {'Err(PT)':>10} {'Err(CU)':>10}"
    )
    print("-" * 150)


def _print_mtx_row(r):
    name = os.path.basename(r["path"])[:27]
    if len(os.path.basename(r["path"])) > 27:
        name = name + "…"
    n_rows, n_cols = r["shape"]
    triton_ms = r.get("triton_ms")
    csr_ms = r.get("cusparse_ms")
    csc_ms = r.get("csc_ms")
    pt_ms = r.get("pytorch_ms")
    err_pt_str = _fmt_err(r.get("err_pt"))
    err_cu_str = _fmt_err(r.get("err_cu"))
    pt_status = _status_str(r.get("triton_ok_pt", False), r.get("err_pt") is not None)
    cu_status = _status_str(r.get("triton_ok_cu", False), r.get("err_cu") is not None)
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {r['nnz']:>10} "
        f"{_fmt_ms(triton_ms):>10} {_fmt_ms(csr_ms):>10} {_fmt_ms(csc_ms):>10} {_fmt_ms(pt_ms):>11} "
        f"{_fmt_speedup(csr_ms, triton_ms):>7} {_fmt_speedup(pt_ms, triton_ms):>7} "
        f"{pt_status:>6} {cu_status:>6} {err_pt_str:>10} {err_cu_str:>10}"
    )


def print_mtx_results(results, value_dtype, index_dtype):
    _print_mtx_header(value_dtype, index_dtype)
    for r in results:
        _print_mtx_row(r)
    print("-" * 150)


def _dtype_str(d):
    return str(d).replace("torch.", "")


def run_all_dtypes_export_csv(paths, csv_path, warmup=10, iters=50, run_cusparse=True):
    """Run SpMV for all VALUE_DTYPES x INDEX_DTYPES on each .mtx and write results to CSV."""
    rows = []
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("=" * 150)
            _print_mtx_header(value_dtype, index_dtype)
            results = run_mtx_batch(
                paths,
                value_dtype=value_dtype,
                index_dtype=index_dtype,
                warmup=warmup,
                iters=iters,
                run_cusparse=run_cusparse,
                on_result=_print_mtx_row,
            )
            print("-" * 150)
            for r in results:
                n_rows, n_cols = r["shape"]
                rows.append({
                    "matrix": os.path.basename(r["path"]),
                    "value_dtype": _dtype_str(value_dtype),
                    "index_dtype": _dtype_str(index_dtype),
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "nnz": r["nnz"],
                    "triton_ms": r.get("triton_ms"),
                    "cusparse_ms": r.get("cusparse_ms"),
                    "pytorch_ms": r.get("pytorch_ms"),
                    "csc_ms": r.get("csc_ms"),
                    "pt_status": _status_str(r.get("triton_ok_pt", False), r.get("err_pt") is not None),
                    "cu_status": _status_str(r.get("triton_ok_cu", False), r.get("err_cu") is not None),
                    "status": r.get("status", r.get("error", "")),
                    "err_pt": r.get("err_pt"),
                    "err_cu": r.get("err_cu"),
                })
    fieldnames = [
        "matrix", "value_dtype", "index_dtype", "n_rows", "n_cols", "nnz",
        "triton_ms", "cusparse_ms", "pytorch_ms", "csc_ms",
        "pt_status", "cu_status", "status", "err_pt", "err_cu",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    print(f"Wrote {len(rows)} rows to {csv_path}")


def run_comprehensive_synthetic():
    """Synthetic benchmark with per-case table (like test_gather)."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    print("=" * 110)
    print("FLAGSPARSE SpMV BENCHMARK (synthetic CSR)")
    print("=" * 110)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Warmup: {WARMUP}  Iters: {ITERS}")
    print("Formats: FlagSparse=CSR, cuSPARSE=CSR (when supported), Reference=CuPy CSR or PyTorch COO")
    print("When CuPy does not support dtype (e.g. bfloat16/float16), reference = PyTorch (float32 then cast).")
    print()
    total = 0
    failed = 0
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("-" * 110)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<12}  |  Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * 110)
            print(
                f"{'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                f"{'FlagSparse(ms)':>11} {'cuSPARSE(ms)':>12} {'FS/CS':>8} "
                f"{'Status':>6} {'Err(FS)':>10} {'Err(CS)':>10}"
            )
            print("-" * 110)
            for n_rows, n_cols, nnz in TEST_CASES:
                result = ast.benchmark_spmv_case(
                    n_rows=n_rows,
                    n_cols=n_cols,
                    nnz=nnz,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=WARMUP,
                    iters=ITERS,
                    run_cusparse=True,
                )
                total += 1
                perf = result["performance"]
                verify = result["verification"]
                backend = result["backend_status"]
                ok = verify["triton_match_reference"]
                cs_ok = verify.get("cusparse_match_reference")
                status = "PASS" if (ok and (cs_ok is None or cs_ok)) else "FAIL"
                if not ok or (cs_ok is False):
                    failed += 1
                triton_ms = perf["triton_ms"]
                cusparse_ms = perf["cusparse_ms"]
                speedup = perf.get("triton_speedup_vs_cusparse")
                if speedup is not None and speedup != "N/A":
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"
                print(
                    f"{n_rows:>7} {n_cols:>7} {nnz:>10} "
                    f"{_fmt_ms(triton_ms):>11} {_fmt_ms(cusparse_ms):>12} {speedup_str:>8} "
                    f"{status:>6} {_fmt_err(verify.get('triton_max_error')):>10} "
                    f"{_fmt_err(verify.get('cusparse_max_error')):>10}"
                )
            print("-" * 110)
            if backend.get("cusparse_unavailable_reason"):
                print(f"  cuSPARSE: {backend['cusparse_unavailable_reason']}")
                print("  Reference: PyTorch (float32 compute then cast to value dtype).")
            print()
    print("=" * 110)
    print(f"Total: {total}  Failed: {failed}")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="SpMV test: SuiteSparse .mtx batch run, error and performance."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic benchmark instead of .mtx",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64", "complex64", "complex128"],
        help="Value dtype (default: float32)",
    )
    parser.add_argument(
        "--index-dtype",
        default="int32",
        choices=["int32", "int64"],
        help="Index dtype (default: int32)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--no-cusparse", action="store_true", help="Skip cuSPARSE baseline")
    parser.add_argument(
        "--csv-csr",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all value_dtype x index_dtype on all .mtx (CSR) and write results to CSV",
    )
    args = parser.parse_args()
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
        run_comprehensive_synthetic()
        return
    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if not paths and not args.csv_csr:
        print("No .mtx files given. Use: python test_spmv.py <file.mtx> [file2.mtx ...] or <dir/>")
        print("Or run synthetic: python test_spmv.py --synthetic")
        print("Or run all dtypes and export CSR CSV: python test_spmv.py <dir/> --csv-csr results.csv")
        return
    if args.csv_csr is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        print("=" * 80)
        print("FLAGSPARSE SpMV (CSR) — all dtypes, export to CSV")
        print("=" * 80)
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  CSV: {args.csv_csr}")
        run_all_dtypes_export_csv(
            paths,
            args.csv_csr,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
        )
        return
    print("=" * 120)
    print("FLAGSPARSE SpMV — SuiteSparse .mtx batch (error + performance)")
    print("=" * 120)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  warmup: {args.warmup}  iters: {args.iters}")
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
    passed = sum(1 for r in results if r.get("status") == "PASS")
    print(f"Passed: {passed} / {len(results)}")


if __name__ == "__main__":
    main()
