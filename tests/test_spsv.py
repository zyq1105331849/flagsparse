"""SpSV tests: synthetic triangular systems and optional .mtx (CSR), multi dtypes/index dtypes."""
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
    from cupyx.scipy.sparse.linalg import spsolve_triangular as cpx_spsolve_triangular
except Exception:
    cp = None
    cpx_sparse = None
    cpx_spsolve_triangular = None

VALUE_DTYPES = [
    torch.float32,
    torch.float64,
]
INDEX_DTYPES = [torch.int32]
TEST_SIZES = [256, 512, 1024, 2048]
WARMUP = 5
ITERS = 20


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


def _tol_for_dtype(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-2
    return 1e-12, 1e-10


DENSE_REF_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _allow_dense_pytorch_ref(shape, dtype):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    dense_bytes = n_rows * n_cols * elem_bytes
    return dense_bytes <= DENSE_REF_MAX_BYTES


def _build_random_triangular_csr(n, value_dtype, index_dtype, device, lower=True):
    """Build a well-conditioned triangular CSR matrix matching CuPy spsolve_triangular semantics.

    - Strictly triangular (lower or upper)
    - Diagonally dominant to avoid singular / ill-conditioned systems
    - Indices sorted by row, then col
    """
    max_bandwidth = max(4, min(n, 16))
    rows_host = []
    cols_host = []
    vals_host = []

    # Use float32/float64 as base for random generation; cast later to target dtype.
    is_complex = value_dtype in (torch.complex64, torch.complex128)
    base_real_dtype = torch.float32 if value_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64) else torch.float64

    for i in range(n):
        if lower:
            cand_cols = list(range(0, i + 1))
        else:
            cand_cols = list(range(i, n))
        if not cand_cols:
            cand_cols = [i]
        k = min(len(cand_cols), max_bandwidth)
        # Always include diagonal index.
        diag_col = i
        off_cand = [c for c in cand_cols if c != diag_col]
        k_off = max(0, k - 1)
        if k_off > len(off_cand):
            k_off = len(off_cand)
        if k_off > 0:
            perm = torch.randperm(len(off_cand))[:k_off].tolist()
            off_cols = [off_cand[j] for j in perm]
        else:
            off_cols = []

        # Generate small off-diagonal values.
        off_vals_real = torch.randn(len(off_cols), dtype=base_real_dtype)
        off_vals_real.mul_(0.01)
        if is_complex:
            off_vals_imag = torch.randn(len(off_cols), dtype=base_real_dtype).mul_(0.01)
            off_vals = off_vals_real + 1j * off_vals_imag
        else:
            off_vals = off_vals_real

        sum_abs = float(torch.sum(torch.abs(off_vals)).item()) if off_vals.numel() > 0 else 0.0
        diag_val = sum_abs + 1.0

        row_idx = [i]
        col_idx = [diag_col]
        val_idx = [diag_val]
        for c, v in zip(off_cols, off_vals.tolist()):
            row_idx.append(i)
            col_idx.append(int(c))
            val_idx.append(v)

        rows_host.extend(row_idx)
        cols_host.extend(col_idx)
        vals_host.extend(val_idx)

    # Convert to tensors and sort by (row, col).
    rows_t = torch.tensor(rows_host, dtype=torch.int64, device=device)
    cols_t = torch.tensor(cols_host, dtype=torch.int64, device=device)
    if is_complex:
        target_complex_dtype = (
            torch.complex64 if value_dtype == torch.complex64 else torch.complex128
        )
        vals_t = torch.tensor(vals_host, dtype=target_complex_dtype, device=device)
    else:
        vals_t = torch.tensor(vals_host, dtype=base_real_dtype, device=device)

    order = torch.argsort(rows_t * max(1, n) + cols_t)
    rows_t = rows_t[order]
    cols_t = cols_t[order]
    vals_t = vals_t[order].to(value_dtype)

    n_rows = n
    nnz_per_row = torch.bincount(rows_t, minlength=n_rows)
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = cols_t.to(index_dtype)
    return vals_t, indices, indptr, (n_rows, n_rows)


def run_spsv_synthetic_all():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device("cuda")
    print("=" * 110)
    print("FLAGSPARSE SpSV BENCHMARK (synthetic triangular systems)")
    print("=" * 110)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP} | Iters: {ITERS}")
    print()

    total = 0
    failed = 0

    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("-" * 110)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<12} | "
                f"Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * 110)
            print(
                f"{'N':>6} {'FlagSparse(ms)':>14} {'PyTorch(ms)':>12} {'CuPy(ms)':>10} "
                f"{'FS/PT':>8} {'FS/CU':>8} {'Status':>8} {'Err(PT)':>12} {'Err(CU)':>12}"
            )
            print("-" * 110)
            for n in TEST_SIZES:
                data, indices, indptr, shape = _build_random_triangular_csr(
                    n, value_dtype, index_dtype, device, lower=True
                )
                A_dense = _csr_to_dense(
                    data, indices.to(torch.int64), indptr, shape
                )
                x_true = torch.randn(n, dtype=value_dtype, device=device)
                if value_dtype in (torch.float16, torch.bfloat16):
                    b = (A_dense.to(torch.float32) @ x_true.to(torch.float32)).to(value_dtype)
                else:
                    b = A_dense @ x_true

                torch.cuda.synchronize()
                x, t_ms = fs.flagsparse_spsv_csr(
                    data, indices, indptr, b, shape, lower=True, return_time=True
                )
                torch.cuda.synchronize()

                # PyTorch baseline (dense solve)
                if value_dtype in (torch.float16, torch.bfloat16):
                    A_ref = A_dense.to(torch.float32)
                    b_ref = b.to(torch.float32)
                else:
                    A_ref = A_dense
                    b_ref = b
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(True)
                e1 = torch.cuda.Event(True)
                e0.record()
                x_pt = torch.linalg.solve(A_ref, b_ref.unsqueeze(1)).squeeze(1)
                e1.record()
                torch.cuda.synchronize()
                pytorch_ms = e0.elapsed_time(e1)
                if value_dtype in (torch.float16, torch.bfloat16):
                    x_pt = x_pt.to(value_dtype)

                err_pt = float(torch.max(torch.abs(x - x_pt)).item()) if n > 0 else 0.0

                # CuPy baseline (triangular solve) for CuPy-supported dtypes.
                cupy_ms = None
                err_cu = None
                x_cu_t = None
                if (
                    cp is not None
                    and cpx_sparse is not None
                    and cpx_spsolve_triangular is not None
                    and value_dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
                ):
                    try:
                        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
                        idx_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
                        ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
                        b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b))
                        A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
                        for _ in range(WARMUP):
                            _ = cpx_spsolve_triangular(
                                A_cp, b_cp, lower=True, unit_diagonal=False
                            )
                        cp.cuda.runtime.deviceSynchronize()
                        t0 = cp.cuda.Event()
                        t1 = cp.cuda.Event()
                        t0.record()
                        for _ in range(ITERS):
                            x_cu = cpx_spsolve_triangular(
                                A_cp, b_cp, lower=True, unit_diagonal=False
                            )
                        t1.record()
                        t1.synchronize()
                        cupy_ms = cp.cuda.get_elapsed_time(t0, t1) / ITERS
                        x_cu_t = torch.utils.dlpack.from_dlpack(x_cu.toDlpack()).to(x.dtype)
                        err_cu = float(torch.max(torch.abs(x - x_cu_t)).item()) if n > 0 else 0.0
                    except Exception:
                        cupy_ms = None
                        err_cu = None

                atol, rtol = 1e-5, 1e-4
                if value_dtype in (torch.float16, torch.bfloat16):
                    atol, rtol = 1e-1, 1e-1
                ok_pt = torch.allclose(x, x_pt, atol=atol, rtol=rtol)
                ok_cu = (
                    True
                    if x_cu_t is None
                    else torch.allclose(x, x_cu_t, atol=atol, rtol=rtol)
                )
                ok = ok_pt and ok_cu
                status = "PASS" if ok else "FAIL"
                if not ok:
                    failed += 1
                total += 1

                fs_vs_pt = (pytorch_ms / t_ms) if (t_ms and t_ms > 0) else None
                fs_vs_cu = (
                    (cupy_ms / t_ms) if (cupy_ms is not None and t_ms and t_ms > 0) else None
                )
                fs_vs_pt_s = f"{fs_vs_pt:.2f}x" if fs_vs_pt is not None else "N/A"
                fs_vs_cu_s = f"{fs_vs_cu:.2f}x" if fs_vs_cu is not None else "N/A"

                print(
                    f"{n:>6} {_fmt_ms(t_ms):>14} {_fmt_ms(pytorch_ms):>12} {_fmt_ms(cupy_ms):>10} "
                    f"{fs_vs_pt_s:>8} {fs_vs_cu_s:>8} {status:>8} {_fmt_err(err_pt):>12} {_fmt_err(err_cu):>12}"
                )
            print("-" * 110)
            print()

    print("=" * 110)
    print(f"Total cases: {total}  Failed: {failed}")
    print("=" * 110)


def _load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = []
    header_info = None
    mm_field = "real"
    mm_symmetry = "general"
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
    if n_rows != n_cols:
        raise ValueError("SpSV requires square matrices")
    row_maps = [dict() for _ in range(n_rows)]

    def _accum(r, c, v):
        row = row_maps[r]
        row[c] = row.get(c, 0.0) + v

    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        if len(parts) >= 3:
            v = float(parts[2])
        elif mm_field == "pattern":
            v = 1.0
        else:
            continue
        _accum(r, c, v)
        if mm_symmetry in ("symmetric", "hermitian") and r != c:
            _accum(c, r, v)
        elif mm_symmetry == "skew-symmetric" and r != c:
            _accum(c, r, -v)

    # SpSV test uses lower triangular solve semantics.
    # Keep only lower-triangular entries (c <= r), then apply requested
    # diagonal update: diag <- diag + sum(row off-diagonal), and enforce
    # non-zero diagonal to avoid singular/nan solves.
    for r in range(n_rows):
        row = row_maps[r]
        lower_row = {}
        off_abs_sum = 0.0
        for c, v in row.items():
            if c < r:
                lower_row[c] = lower_row.get(c, 0.0) + v
                off_abs_sum += abs(v)
        lower_row[r] = off_abs_sum + 1.0
        row_maps[r] = lower_row

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


def _csr_to_dense(data, indices, indptr, shape):
    n_rows, n_cols = shape
    row_ind = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    coo = torch.sparse_coo_tensor(
        torch.stack([row_ind, indices.to(torch.int64)]),
        data,
        (n_rows, n_cols),
        device=data.device,
    ).coalesce()
    return coo.to_dense()


def run_all_dtypes_spsv_csv(mtx_paths, csv_path):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            atol, rtol = _tol_for_dtype(value_dtype)
            print("=" * 150)
            print(
                f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}"
            )
            print("Formats: FlagSparse=CSR, cuSPARSE=CSR, PyTorch=Dense solve.")
            print("Err(PT)=|FlagSparse-PyTorch|, Err(CU)=|FlagSparse-cuSPARSE|.  PASS if either error within tolerance.")
            print("-" * 150)
            print(
                f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                f"{'FlagSparse(ms)':>10} {'CSR(ms)':>10} {'CSC(ms)':>10} {'PyTorch(ms)':>11} "
                f"{'FS/CSR':>7} {'FS/PT':>7} {'Status':>6} {'Err(PT)':>10} {'Err(CU)':>10}"
            )
            print("-" * 150)
            for path in mtx_paths:
                try:
                    data, indices, indptr, shape = _load_mtx_to_csr_torch(
                        path, dtype=value_dtype, device=device
                    )
                    indices = indices.to(index_dtype)
                    n_rows, n_cols = shape
                    x_true = torch.randn(n_rows, dtype=value_dtype, device=device)
                    from flagsparse import flagsparse_spmv_csr

                    b, _ = flagsparse_spmv_csr(
                        data, indices, indptr, x_true, shape, return_time=True
                    )
                    x, t_ms = fs.flagsparse_spsv_csr(
                        data, indices, indptr, b, shape, lower=True, return_time=True
                    )
                    pytorch_ms = None
                    err_pt = None
                    ok_pt = False
                    pt_skip_reason = None
                    if _allow_dense_pytorch_ref(shape, value_dtype):
                        try:
                            A_dense = _csr_to_dense(
                                data, indices.to(torch.int64), indptr, shape
                            )
                            if value_dtype in (torch.float16, torch.bfloat16):
                                A_ref = A_dense.to(torch.float32)
                                b_ref = b.to(torch.float32)
                            else:
                                A_ref = A_dense
                                b_ref = b
                            # PyTorch baseline (dense solve)
                            e0 = torch.cuda.Event(True)
                            e1 = torch.cuda.Event(True)
                            torch.cuda.synchronize()
                            e0.record()
                            x_ref = torch.linalg.solve(
                                A_ref, b_ref.unsqueeze(1)
                            ).squeeze(1)
                            e1.record()
                            torch.cuda.synchronize()
                            pytorch_ms = e0.elapsed_time(e1)
                            if value_dtype in (torch.float16, torch.bfloat16):
                                x_ref = x_ref.to(value_dtype)
                            err_pt = float(torch.max(torch.abs(x - x_ref)).item()) if n_rows > 0 else 0.0
                            ok_pt = torch.allclose(x, x_ref, atol=atol, rtol=rtol)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                pt_skip_reason = "PyTorch dense ref OOM; skipped"
                            else:
                                raise
                    else:
                        pt_skip_reason = (
                            f"PyTorch dense ref skipped (> {DENSE_REF_MAX_BYTES // (1024**3)} GiB dense matrix)"
                        )

                    # CuPy baseline (if supported)
                    cupy_ms = None
                    err_cu = None
                    ok_cu = False
                    if (
                        cp is not None
                        and cpx_sparse is not None
                        and cpx_spsolve_triangular is not None
                        and value_dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
                    ):
                        try:
                            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
                            idx_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
                            ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
                            b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b))
                            A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
                            for _ in range(WARMUP):
                                _ = cpx_spsolve_triangular(A_cp, b_cp, lower=True, unit_diagonal=False)
                            cp.cuda.runtime.deviceSynchronize()
                            c0 = cp.cuda.Event()
                            c1 = cp.cuda.Event()
                            c0.record()
                            for _ in range(ITERS):
                                x_cu = cpx_spsolve_triangular(A_cp, b_cp, lower=True, unit_diagonal=False)
                            c1.record()
                            c1.synchronize()
                            cupy_ms = cp.cuda.get_elapsed_time(c0, c1) / ITERS
                            x_cu_t = torch.utils.dlpack.from_dlpack(x_cu.toDlpack()).to(x.dtype)
                            err_cu = float(torch.max(torch.abs(x - x_cu_t)).item()) if n_rows > 0 else 0.0
                            ok_cu = torch.allclose(x, x_cu_t, atol=atol, rtol=rtol)
                        except Exception:
                            cupy_ms = None
                            err_cu = None

                    status = "PASS" if (ok_pt or ok_cu) else "FAIL"
                    if (not ok_pt) and (not ok_cu) and (err_pt is None and err_cu is None):
                        status = "REF_FAIL"
                    rows_out.append(
                        {
                            "matrix": os.path.basename(path),
                            "value_dtype": _dtype_name(value_dtype),
                            "index_dtype": _dtype_name(index_dtype),
                            "n_rows": n_rows,
                            "n_cols": n_cols,
                            "nnz": int(data.numel()),
                            "triton_ms": t_ms,
                            "pytorch_ms": pytorch_ms,
                            "cusparse_ms": cupy_ms,
                            "csc_ms": None,
                            "status": status,
                            "err_pt": err_pt,
                            "err_cu": err_cu,
                        }
                    )
                    name = os.path.basename(path)[:27]
                    if len(os.path.basename(path)) > 27:
                        name = name + "…"
                    print(
                        f"{name:<28} {n_rows:>7} {n_cols:>7} {int(data.numel()):>10} "
                        f"{_fmt_ms(t_ms):>10} {_fmt_ms(cupy_ms):>10} {_fmt_ms(None):>10} {_fmt_ms(pytorch_ms):>11} "
                        f"{_fmt_speedup(cupy_ms, t_ms):>7} {_fmt_speedup(pytorch_ms, t_ms):>7} "
                        f"{status:>6} {_fmt_err(err_pt):>10} {_fmt_err(err_cu):>10}"
                    )
                    if pt_skip_reason:
                        print(f"  NOTE: {pt_skip_reason}")
                except Exception as e:
                    err_msg = str(e)
                    status = "SKIP" if "SpSV requires square matrices" in err_msg else "ERROR"
                    rows_out.append(
                        {
                            "matrix": os.path.basename(path),
                            "value_dtype": _dtype_name(value_dtype),
                            "index_dtype": _dtype_name(index_dtype),
                            "n_rows": "ERR",
                            "n_cols": "ERR",
                            "nnz": "ERR",
                            "triton_ms": None,
                            "pytorch_ms": None,
                            "cusparse_ms": None,
                            "csc_ms": None,
                            "status": status,
                            "err_pt": None,
                            "err_cu": None,
                        }
                    )
                    name = os.path.basename(path)[:27]
                    if len(os.path.basename(path)) > 27:
                        name = name + "…"
                    print(
                        f"{name:<28} {'ERR':>7} {'ERR':>7} {'ERR':>10} "
                        f"{_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>11} "
                        f"{'N/A':>7} {'N/A':>7} {status:>6} {_fmt_err(None):>10} {_fmt_err(None):>10}"
                    )
                    print(f"  {status}: {e}")
            print("-" * 150)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "n_rows",
        "n_cols",
        "nnz",
        "triton_ms",
        "pytorch_ms",
        "cusparse_ms",
        "csc_ms",
        "status",
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
        description="SpSV test: synthetic triangular systems and optional .mtx, export CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic triangular tests"
    )
    parser.add_argument(
        "--csv-csr",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all dtypes/index_dtypes on given .mtx (CSR SpSV) and export results to CSV",
    )
    args = parser.parse_args()

    if args.synthetic:
        run_spsv_synthetic_all()
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if args.csv_csr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-csr")
            return
        run_all_dtypes_spsv_csv(paths, args.csv_csr)
        return

    print("Use --synthetic or --csv-csr to run SpSV CSR tests.")


if __name__ == "__main__":
    main()
