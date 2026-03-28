"""SpMV COO tests: float32/float64 + int32, synthetic + optional .mtx, compare FlagSparse vs PyTorch/CuPy."""
import argparse
import glob
import csv
import math
import os

import torch
import flagsparse as fs
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    cp = None
    cpx_sparse = None

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPE = torch.int32
TEST_SIZES = [(512, 512), (1024, 1024), (2048, 2048)]
WARMUP = 10
ITERS = 50


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


def _allclose_error_ratio(actual, reference, atol, rtol):
    if actual.numel() == 0:
        return 0.0
    diff = torch.abs(actual - reference).to(torch.float64)
    tol = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / tol).item())


def _reference_dtype(dtype):
    return torch.float64 if dtype == torch.float32 else dtype


def _pytorch_coo_reference(data, row, col, x, shape, out_dtype):
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    coo_ref = torch.sparse_coo_tensor(
        torch.stack([row.to(torch.int64), col.to(torch.int64)]),
        data_ref,
        shape,
        device=data.device,
    ).coalesce()
    y_ref = torch.sparse.mm(coo_ref, x_ref.unsqueeze(1)).squeeze(1)
    return y_ref.to(out_dtype) if ref_dtype != out_dtype else y_ref


def _cupy_coo_reference(data, row, col, x, shape, out_dtype):
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref))
    row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.to(torch.int64)))
    col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.to(torch.int64)))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_ref))
    A_cp_ref = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    y_ref = A_cp_ref @ x_cp
    y_ref_t = torch.utils.dlpack.from_dlpack(y_ref.toDlpack())
    return y_ref_t.to(out_dtype) if ref_dtype != out_dtype else y_ref_t


def _dense_to_coo(A):
    rows, cols = A.nonzero(as_tuple=True)
    data = A[rows, cols]
    return data, rows, cols


COO_SEP = "-" * 172
COO_HEADER = (
    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10}  "
    f"{'Base(ms)':>9} {'Opt(ms)':>9} {'PT(ms)':>9} {'CU(ms)':>9}  "
    f"{'Opt/Base':>8} {'Opt/PT':>8} {'Opt/CU':>8}  "
    f"{'Err(Base)':>10} {'Err(Opt)':>10} {'Status':>6}"
)


def _spd(num, den):
    if num is None or den is None or den <= 0:
        return "N/A"
    return f"{num / den:.2f}x"


# FlagSparse native COO SpMV: see sparse_operations.spmv_coo
COO_ATOMIC_BLOCK = 256
COO_ATOMIC_WARPS = 4
COO_SEG_BLOCK_INNER = 128

def _timed_flagsparse_coo(prepared, x, warmup, iters):
    op = lambda: fs.flagsparse_spmv_coo(
        x=x,
        prepared=prepared,
        return_time=False,
        block_inner=COO_SEG_BLOCK_INNER,
        block_size=COO_ATOMIC_BLOCK,
        num_warps=COO_ATOMIC_WARPS,
    )
    y = op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        y = op()
    e1.record()
    torch.cuda.synchronize()
    return y, e0.elapsed_time(e1) / iters


def run_synthetic():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device("cuda")
    print("=" * 172)
    print("FLAGSPARSE SpMV COO BENCHMARK (synthetic dense -> COO). All backends stay COO.")
    print("=" * 172)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP} | Iters: {ITERS}")
    print()

    for dtype in VALUE_DTYPES:
        atol, rtol = _tol_for_dtype(dtype)
        print(COO_SEP)
        print(f"dtype: {_dtype_name(dtype)}  index_dtype: int32")
        print(COO_SEP)
        print(
            "FlagSparse: prepare_spmv_coo + Triton COO SpMV (no CSR). "
            "Base(ms) = row-run (seg) kernel; Opt(ms) = NNZ atomic kernel."
        )
        print(
            "PyTorch/CuPy: COO sparse.mm / cupyx coo_matrix @ x (no tocsr). "
            "Err vs PyTorch COO reference in fp64 (fp32 casts back)."
        )
        print(COO_SEP)
        print(
            f"{'M':>6} {'N':>6} {'NNZ':>10}  "
            f"{'Base(ms)':>9} {'Opt(ms)':>9} {'PT(ms)':>9} {'CU(ms)':>9}  "
            f"{'Opt/Base':>8} {'Opt/PT':>8} {'Opt/CU':>8}  "
            f"{'Err(Base)':>10} {'Err(Opt)':>10} {'Status':>6}"
        )
        print(COO_SEP)
        for m, n in TEST_SIZES:
            A = torch.randn(m, n, dtype=dtype, device=device)
            A *= (torch.rand_like(A) < 0.1)
            data, row, col = _dense_to_coo(A)
            nnz = int(data.numel())
            x = torch.randn(n, dtype=dtype, device=device)
            prepared_seg = fs.prepare_spmv_coo(
                data, row, col, (m, n), sort_by_row=True
            )
            prepared_at = fs.prepare_spmv_coo(
                data, row, col, (m, n), sort_by_row=False
            )
            y_base, base_ms = _timed_flagsparse_coo(
                prepared_seg, x, WARMUP, ITERS
            )
            y_opt, opt_ms = _timed_flagsparse_coo(
                prepared_at, x, WARMUP, ITERS
            )
            y_ref = _pytorch_coo_reference(data, row, col, x, (m, n), dtype)
            err_base = _allclose_error_ratio(y_base, y_ref, atol, rtol)
            err_opt = _allclose_error_ratio(y_opt, y_ref, atol, rtol)
            st = (
                "PASS"
                if (
                    (not math.isnan(err_base))
                    and (not math.isnan(err_opt))
                    and err_base <= 1.0
                    and err_opt <= 1.0
                )
                else "FAIL"
            )
            if nnz > 0:
                coo = torch.sparse_coo_tensor(
                    torch.stack([row.to(torch.int64), col.to(torch.int64)]),
                    data,
                    (m, n),
                    device=device,
                ).coalesce()
                torch.cuda.synchronize()
                for _ in range(WARMUP):
                    _ = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(True)
                e1 = torch.cuda.Event(True)
                e0.record()
                for _ in range(ITERS):
                    y_pt = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)
                e1.record()
                torch.cuda.synchronize()
                pt_ms = e0.elapsed_time(e1) / ITERS
            else:
                y_pt = torch.zeros(m, dtype=dtype, device=device)
                pt_ms = 0.0
            cu_ms = None
            if cp is not None and cpx_sparse is not None:
                try:
                    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
                    row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.to(torch.int64)))
                    col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.to(torch.int64)))
                    A_cp = cpx_sparse.coo_matrix(
                        (data_cp, (row_cp, col_cp)), shape=(m, n)
                    )
                    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
                    for _ in range(WARMUP):
                        _ = A_cp @ x_cp
                    cp.cuda.runtime.deviceSynchronize()
                    c0 = cp.cuda.Event()
                    c1 = cp.cuda.Event()
                    c0.record()
                    for _ in range(ITERS):
                        y_cu = A_cp @ x_cp
                    c1.record()
                    c1.synchronize()
                    cu_ms = cp.cuda.get_elapsed_time(c0, c1) / ITERS
                except Exception:
                    cu_ms = None
            print(
                f"{m:>6} {n:>6} {nnz:>10}  "
                f"{_fmt_ms(base_ms):>9} {_fmt_ms(opt_ms):>9} {_fmt_ms(pt_ms):>9} {_fmt_ms(cu_ms):>9}  "
                f"{_spd(base_ms, opt_ms):>8} {_spd(pt_ms, opt_ms):>8} {_spd(cu_ms, opt_ms):>8}  "
                f"{_fmt_err(err_base):>10} {_fmt_err(err_opt):>10} {st:>6}"
            )
        print(COO_SEP)
        print()


def _load_mtx_to_coo_torch(file_path, dtype=torch.float32, device=None):
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

    is_pattern = (mm_field == "pattern")
    is_symmetric = mm_symmetry in ("symmetric", "hermitian")
    is_skew = (mm_symmetry == "skew-symmetric")

    rows_host = []
    cols_host = []
    vals_host = []
    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        v = 1.0 if is_pattern else (float(parts[2]) if len(parts) >= 3 else 0.0)
        if 0 <= r < n_rows and 0 <= c < n_cols:
            rows_host.append(r)
            cols_host.append(c)
            vals_host.append(v)
            if r != c:
                if is_symmetric and 0 <= c < n_rows and 0 <= r < n_cols:
                    rows_host.append(c)
                    cols_host.append(r)
                    vals_host.append(v)
                elif is_skew and 0 <= c < n_rows and 0 <= r < n_cols:
                    rows_host.append(c)
                    cols_host.append(r)
                    vals_host.append(-v)
    rows = torch.tensor(rows_host, dtype=torch.int64, device=device)
    cols = torch.tensor(cols_host, dtype=torch.int64, device=device)
    vals = torch.tensor(vals_host, dtype=dtype, device=device)
    return vals, rows, cols, (n_rows, n_cols)


def _tol_for_dtype(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-2
    return 1e-12, 1e-10


# Dense PyTorch reference for SpSV can OOM on large matrices.
DENSE_REF_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _allow_dense_pytorch_ref(shape, dtype):
    n_rows, n_cols = shape
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    dense_bytes = int(n_rows) * int(n_cols) * int(elem_bytes)
    return dense_bytes <= DENSE_REF_MAX_BYTES


def run_all_dtypes_coo_csv(mtx_paths, csv_path):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    for dtype in VALUE_DTYPES:
        atol, rtol = _tol_for_dtype(dtype)
        print("=" * 172)
        print(f"Value dtype: {_dtype_name(dtype)}  |  Index dtype: int32")
        print("Input: MatrixMarket → COO. FlagSparse: COO Triton only (seg + atomic), no CSR.")
        print("PyTorch = COO sparse.mm; CuPy = COO matvec (coo_matrix @ x, no tocsr).")
        print(
            "Base(ms) = FlagSparse COO row-run (seg); Opt(ms) = FlagSparse COO NNZ atomic; "
            "PT/CU = COO baselines."
        )
        print(
            f"prepare_spmv_coo once per variant + {WARMUP} warmup + "
            f"{ITERS} CUDA-event-averaged SpMV per backend."
        )
        print(COO_SEP)
        print(COO_HEADER)
        print(COO_SEP)
        for path in mtx_paths:
            try:
                data, row, col, shape = _load_mtx_to_coo_torch(
                    path, dtype=dtype, device=device
                )
                m, n = shape
                x = torch.randn(n, dtype=dtype, device=device)
                prepared_seg = fs.prepare_spmv_coo(
                    data, row, col, shape, sort_by_row=True
                )
                prepared_at = fs.prepare_spmv_coo(
                    data, row, col, shape, sort_by_row=False
                )
                y_base, base_ms = _timed_flagsparse_coo(
                    prepared_seg, x, WARMUP, ITERS
                )
                y_opt, opt_ms = _timed_flagsparse_coo(
                    prepared_at, x, WARMUP, ITERS
                )
                y_ref = _pytorch_coo_reference(data, row, col, x, shape, dtype)
                err_base = _allclose_error_ratio(y_base, y_ref, atol, rtol)
                err_opt = _allclose_error_ratio(y_opt, y_ref, atol, rtol)
                err_pt = None
                err_cu = None
                opt_ok = (
                    (not math.isnan(err_base))
                    and (not math.isnan(err_opt))
                    and err_base <= 1.0
                    and err_opt <= 1.0
                )
                if data.numel() > 0:
                    coo = torch.sparse_coo_tensor(
                        torch.stack([row, col]),
                        data,
                        shape,
                        device=device,
                    ).coalesce()
                    torch.cuda.synchronize()
                    for _ in range(WARMUP):
                        _ = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(True)
                    e1 = torch.cuda.Event(True)
                    e0.record()
                    for _ in range(ITERS):
                        y_pt = torch.sparse.mm(coo, x.unsqueeze(1)).squeeze(1)
                    e1.record()
                    torch.cuda.synchronize()
                    pt_ms = e0.elapsed_time(e1) / ITERS
                    err_pt = _allclose_error_ratio(y_opt, y_pt, atol, rtol)
                else:
                    pt_ms = 0.0
                cu_ms = None
                triton_ok_pt = False
                triton_ok_cu = False
                if cp is not None and cpx_sparse is not None:
                    try:
                        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
                        row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row))
                        col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col))
                        A_cp = cpx_sparse.coo_matrix(
                            (data_cp, (row_cp, col_cp)), shape=shape
                        )
                        x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
                        for _ in range(WARMUP):
                            _ = A_cp @ x_cp
                        cp.cuda.runtime.deviceSynchronize()
                        c0 = cp.cuda.Event()
                        c1 = cp.cuda.Event()
                        c0.record()
                        for _ in range(ITERS):
                            y_cu = A_cp @ x_cp
                        c1.record()
                        c1.synchronize()
                        cu_ms = cp.cuda.get_elapsed_time(c0, c1) / ITERS
                        y_cu_t = torch.utils.dlpack.from_dlpack(y_cu.toDlpack())
                        err_cu = _allclose_error_ratio(y_opt, y_cu_t, atol, rtol)
                        triton_ok_cu = (not math.isnan(err_cu)) and err_cu <= 1.0
                    except Exception:
                        cu_ms = None
                        err_cu = None
                if err_pt is not None:
                    triton_ok_pt = (not math.isnan(err_pt)) and err_pt <= 1.0
                status = "PASS" if opt_ok else "FAIL"
                pt_status = _status_str(triton_ok_pt, err_pt is not None)
                cu_status = _status_str(triton_ok_cu, err_cu is not None)
                rows_out.append(
                    {
                        "matrix": os.path.basename(path),
                        "value_dtype": _dtype_name(dtype),
                        "index_dtype": "torch.int32",
                        "n_rows": m,
                        "n_cols": n,
                        "nnz": int(data.numel()),
                        "base_ms": base_ms,
                        "opt_ms": opt_ms,
                        "triton_ms": opt_ms,
                        "cusparse_ms": cu_ms,
                        "pytorch_ms": pt_ms,
                        "csc_ms": None,
                        "pt_status": pt_status,
                        "cu_status": cu_status,
                        "status": status,
                        "err_base": err_base,
                        "err_opt": err_opt,
                        "err_pt": err_pt,
                        "err_cu": err_cu,
                    }
                )
                name = os.path.basename(path)[:27]
                if len(os.path.basename(path)) > 27:
                    name = name + "…"
                print(
                    f"{name:<28} {m:>7} {n:>7} {int(data.numel()):>10}  "
                    f"{_fmt_ms(base_ms):>9} {_fmt_ms(opt_ms):>9} {_fmt_ms(pt_ms):>9} {_fmt_ms(cu_ms):>9}  "
                    f"{_spd(base_ms, opt_ms):>8} {_spd(pt_ms, opt_ms):>8} {_spd(cu_ms, opt_ms):>8}  "
                    f"{_fmt_err(err_base):>10} {_fmt_err(err_opt):>10} {status:>6}"
                )
            except Exception as e:
                rows_out.append(
                    {
                        "matrix": os.path.basename(path),
                        "value_dtype": _dtype_name(dtype),
                        "index_dtype": "torch.int32",
                        "n_rows": "ERR",
                        "n_cols": "ERR",
                        "nnz": "ERR",
                        "base_ms": None,
                        "opt_ms": None,
                        "triton_ms": None,
                        "cusparse_ms": None,
                        "pytorch_ms": None,
                        "csc_ms": None,
                        "status": "ERROR",
                        "err_base": None,
                        "err_opt": None,
                        "err_pt": None,
                        "err_cu": None,
                        "pt_status": "N/A",
                        "cu_status": "N/A",
                    }
                )
                name = os.path.basename(path)[:27]
                if len(os.path.basename(path)) > 27:
                    name = name + "…"
                print(
                    f"{name:<28} {'ERR':>7} {'ERR':>7} {'ERR':>10}  "
                    f"{_fmt_ms(None):>9} {_fmt_ms(None):>9} {_fmt_ms(None):>9} {_fmt_ms(None):>9}  "
                    f"{'N/A':>8} {'N/A':>8} {'N/A':>8}  "
                    f"{_fmt_err(None):>10} {_fmt_err(None):>10} {'ERROR':>6}"
                )
                print(f"  ERROR: {e}")
        print(COO_SEP)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "n_rows",
        "n_cols",
        "nnz",
        "base_ms",
        "opt_ms",
        "triton_ms",
        "cusparse_ms",
        "pytorch_ms",
        "csc_ms",
        "pt_status",
        "cu_status",
        "status",
        "err_base",
        "err_opt",
        "err_pt",
        "err_cu",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpMV COO test: synthetic dense->COO and optional .mtx, export CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic dense->COO tests"
    )
    parser.add_argument(
        "--csv-coo",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all dtypes on given .mtx and export COO SpMV results to CSV",
    )
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic()
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_coo_csv(paths, args.csv_coo)
        return

    print("Use --synthetic or --csv-coo to run COO SpMV tests.")


if __name__ == "__main__":
    main()
