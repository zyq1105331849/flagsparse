"""CSR SpGEMM (A@B) with two-phase structure/value build."""

from ._common import *

SUPPORTED_SPGEMM_VALUE_DTYPES = (torch.float32, torch.float64)


class SpGEMMPrepared:
    """Prepared CSR metadata for repeated SpGEMM runs."""

    __slots__ = (
        "a_data",
        "a_indices",
        "a_indptr",
        "a_shape",
        "b_data",
        "b_indices",
        "b_indptr",
        "b_shape",
        "n_rows",
        "n_inner",
        "n_cols",
        "a_row_work",
        "row_bucket",
        "hash_capacity_hint",
        "block_nnz",
    )

    def __init__(
        self,
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
        a_row_work,
        row_bucket,
        hash_capacity_hint,
        block_nnz,
    ):
        self.a_data = a_data
        self.a_indices = a_indices
        self.a_indptr = a_indptr
        self.a_shape = (int(a_shape[0]), int(a_shape[1]))
        self.b_data = b_data
        self.b_indices = b_indices
        self.b_indptr = b_indptr
        self.b_shape = (int(b_shape[0]), int(b_shape[1]))
        self.n_rows = self.a_shape[0]
        self.n_inner = self.a_shape[1]
        self.n_cols = self.b_shape[1]
        self.a_row_work = a_row_work
        self.row_bucket = row_bucket
        self.hash_capacity_hint = int(hash_capacity_hint)
        self.block_nnz = int(block_nnz)


def _validate_csr(data, indices, indptr, shape, tag):
    if len(shape) != 2:
        raise ValueError(f"{tag}_shape must be a 2-tuple")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError(f"{tag} data/indices/indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"{tag}_shape dimensions must be non-negative")
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"{tag}_indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError(f"{tag}_data and {tag}_indices must have the same length")
    if not data.is_cuda or not indices.is_cuda or not indptr.is_cuda:
        raise ValueError(f"{tag} tensors must be CUDA tensors")
    if data.dtype not in SUPPORTED_SPGEMM_VALUE_DTYPES:
        raise TypeError(f"{tag}_data dtype must be torch.float32 or torch.float64")
    if indices.dtype != torch.int32:
        raise TypeError(f"{tag}_indices dtype must be torch.int32")
    if indptr.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{tag}_indptr dtype must be torch.int32 or torch.int64")

    nnz = int(data.numel())
    indptr_i64 = indptr.to(torch.int64)
    if indptr_i64.numel() > 0 and int(indptr_i64[0].item()) != 0:
        raise ValueError(f"{tag}_indptr[0] must be 0")
    if indptr_i64.numel() > 0 and int(indptr_i64[-1].item()) != nnz:
        raise ValueError(f"{tag}_indptr[-1] must equal nnz={nnz}")
    if indptr_i64.numel() > 1 and bool(torch.any(indptr_i64[1:] < indptr_i64[:-1]).item()):
        raise ValueError(f"{tag}_indptr must be nondecreasing")
    if nnz > 0:
        min_col = int(indices.min().item())
        max_col = int(indices.max().item())
        if min_col < 0 or max_col >= n_cols:
            raise IndexError(f"{tag}_indices out of range for n_cols={n_cols}")
    return n_rows, n_cols, indptr_i64


def _prepare_spgemm_csr_inputs(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    a_rows, a_cols, a_indptr64 = _validate_csr(a_data, a_indices, a_indptr, a_shape, "a")
    b_rows, b_cols, b_indptr64 = _validate_csr(b_data, b_indices, b_indptr, b_shape, "b")
    if a_cols != b_rows:
        raise ValueError(
            f"shape mismatch for A@B: A is {a_rows}x{a_cols}, B is {b_rows}x{b_cols}"
        )
    if a_data.device != b_data.device:
        raise ValueError("A and B tensors must be on the same CUDA device")
    if a_data.dtype != b_data.dtype:
        raise TypeError("A and B value dtype must match")

    a_data = a_data.contiguous()
    a_indices = a_indices.contiguous()
    a_indptr64 = a_indptr64.contiguous()
    b_data = b_data.contiguous()
    b_indices = b_indices.contiguous()
    b_indptr64 = b_indptr64.contiguous()
    return (
        a_data,
        a_indices,
        a_indptr64,
        (a_rows, a_cols),
        b_data,
        b_indices,
        b_indptr64,
        (b_rows, b_cols),
    )


@triton.jit
def _spgemm_row_work_kernel(
    a_indptr_ptr,
    a_indices_ptr,
    b_indptr_ptr,
    row_work_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(a_indptr_ptr + row)
    end = tl.load(a_indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros((), dtype=tl.int32)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        offs = start + chunk_start + tl.arange(0, BLOCK_NNZ)
        mask = offs < end
        k = tl.load(a_indices_ptr + offs, mask=mask, other=0)
        b_start = tl.load(b_indptr_ptr + k, mask=mask, other=0)
        b_end = tl.load(b_indptr_ptr + k + 1, mask=mask, other=0)
        contrib = (b_end - b_start).to(tl.int32)
        acc += tl.sum(tl.where(mask, contrib, 0))
    tl.store(row_work_ptr + row, acc)


def _estimate_hash_capacity(a_row_work):
    if a_row_work.numel() == 0:
        return 256
    p95 = int(torch.quantile(a_row_work.to(torch.float32), 0.95).item())
    p95 = max(p95, 1)
    cap = 1
    while cap < p95:
        cap <<= 1
    return max(256, cap)


def _build_row_bucket(a_row_work):
    # 0: small, 1: medium, 2: long
    bucket = torch.zeros_like(a_row_work, dtype=torch.int8)
    bucket = torch.where(a_row_work > 4096, torch.full_like(bucket, 2), bucket)
    bucket = torch.where((a_row_work > 256) & (a_row_work <= 4096), torch.full_like(bucket, 1), bucket)
    return bucket


def prepare_spgemm_csr(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    block_nnz=256,
):
    if block_nnz <= 0:
        raise ValueError("block_nnz must be positive")
    (
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    ) = _prepare_spgemm_csr_inputs(
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    )
    n_rows = int(a_shape[0])
    if n_rows == 0:
        row_work = torch.empty(0, dtype=torch.int32, device=a_data.device)
    else:
        row_work = torch.empty(n_rows, dtype=torch.int32, device=a_data.device)
        try:
            _spgemm_row_work_kernel[(n_rows,)](
                a_indptr,
                a_indices,
                b_indptr,
                row_work,
                n_rows,
                BLOCK_NNZ=int(block_nnz),
            )
        except Exception:
            b_row_nnz = (b_indptr[1:] - b_indptr[:-1]).to(torch.int32)
            for row in range(n_rows):
                start = int(a_indptr[row].item())
                end = int(a_indptr[row + 1].item())
                if end <= start:
                    row_work[row] = 0
                    continue
                cols = a_indices[start:end].to(torch.int64)
                row_work[row] = torch.sum(b_row_nnz[cols]).to(torch.int32)
    row_bucket = _build_row_bucket(row_work)
    hash_capacity_hint = _estimate_hash_capacity(row_work)
    return SpGEMMPrepared(
        a_data=a_data,
        a_indices=a_indices,
        a_indptr=a_indptr,
        a_shape=a_shape,
        b_data=b_data,
        b_indices=b_indices,
        b_indptr=b_indptr,
        b_shape=b_shape,
        a_row_work=row_work,
        row_bucket=row_bucket,
        hash_capacity_hint=hash_capacity_hint,
        block_nnz=block_nnz,
    )


def _expand_b_positions_for_a_row(a_cols_i32, b_indptr_i64, device):
    if a_cols_i32.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, empty
    a_cols_i64 = a_cols_i32.to(torch.int64)
    starts = b_indptr_i64[a_cols_i64]
    ends = b_indptr_i64[a_cols_i64 + 1]
    counts = ends - starts
    total = int(counts.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, empty
    owner = torch.repeat_interleave(
        torch.arange(a_cols_i64.numel(), device=device, dtype=torch.int64), counts
    )
    prefix = torch.cumsum(counts, dim=0)
    base = prefix - counts
    repeated_starts = torch.repeat_interleave(starts, counts)
    intra = torch.arange(total, device=device, dtype=torch.int64) - torch.repeat_interleave(base, counts)
    b_pos = repeated_starts + intra
    return b_pos, owner, counts


def _spgemm_count_phase(prepared):
    n_rows = prepared.n_rows
    device = prepared.a_data.device
    row_nnz_c = torch.zeros(n_rows, dtype=torch.int64, device=device)
    for row in range(n_rows):
        start = int(prepared.a_indptr[row].item())
        end = int(prepared.a_indptr[row + 1].item())
        if end <= start:
            continue
        a_cols = prepared.a_indices[start:end]
        b_pos, _, _ = _expand_b_positions_for_a_row(
            a_cols, prepared.b_indptr, device=device
        )
        if b_pos.numel() == 0:
            continue
        cols = prepared.b_indices[b_pos]
        row_nnz_c[row] = torch.unique(cols, sorted=True).numel()
    return row_nnz_c


def _spgemm_fill_phase(prepared, c_indptr):
    nnz_c = int(c_indptr[-1].item())
    device = prepared.a_data.device
    c_data = torch.empty(nnz_c, dtype=prepared.a_data.dtype, device=device)
    c_indices = torch.empty(nnz_c, dtype=torch.int32, device=device)
    if nnz_c == 0:
        return c_data, c_indices

    for row in range(prepared.n_rows):
        out_start = int(c_indptr[row].item())
        out_end = int(c_indptr[row + 1].item())
        if out_end <= out_start:
            continue
        start = int(prepared.a_indptr[row].item())
        end = int(prepared.a_indptr[row + 1].item())
        if end <= start:
            continue
        a_cols = prepared.a_indices[start:end]
        a_vals = prepared.a_data[start:end]
        b_pos, owner, _ = _expand_b_positions_for_a_row(
            a_cols, prepared.b_indptr, device=device
        )
        if b_pos.numel() == 0:
            c_indices[out_start:out_end] = 0
            c_data[out_start:out_end] = 0
            continue
        cols = prepared.b_indices[b_pos]
        b_vals = prepared.b_data[b_pos]
        contrib = a_vals[owner] * b_vals
        uniq_cols, inv = torch.unique(cols, sorted=True, return_inverse=True)
        vals = torch.zeros(uniq_cols.numel(), dtype=prepared.a_data.dtype, device=device)
        vals.scatter_add_(0, inv, contrib)
        c_indices[out_start:out_end] = uniq_cols
        c_data[out_start:out_end] = vals

    return c_data, c_indices


def _run_spgemm_prepared(prepared):
    torch.cuda.synchronize()
    t_count0 = time.perf_counter()
    row_nnz_c = _spgemm_count_phase(prepared)
    torch.cuda.synchronize()
    count_ms = (time.perf_counter() - t_count0) * 1000.0

    c_indptr = torch.empty(prepared.n_rows + 1, dtype=torch.int64, device=prepared.a_data.device)
    c_indptr[0] = 0
    if prepared.n_rows > 0:
        c_indptr[1:] = torch.cumsum(row_nnz_c, dim=0)

    torch.cuda.synchronize()
    t_fill0 = time.perf_counter()
    c_data, c_indices = _spgemm_fill_phase(prepared, c_indptr)
    torch.cuda.synchronize()
    fill_ms = (time.perf_counter() - t_fill0) * 1000.0

    return c_data, c_indices, c_indptr, {"count_ms": count_ms, "fill_ms": fill_ms}


def flagsparse_spgemm_csr(
    a_data=None,
    a_indices=None,
    a_indptr=None,
    a_shape=None,
    b_data=None,
    b_indices=None,
    b_indptr=None,
    b_shape=None,
    prepared=None,
    out=None,
    return_time=False,
    return_meta=False,
):
    """CSR SpGEMM: C = A @ B with CSR output (non-transpose path)."""
    prepare_ms = 0.0
    if prepared is None:
        if any(
            x is None
            for x in (
                a_data,
                a_indices,
                a_indptr,
                a_shape,
                b_data,
                b_indices,
                b_indptr,
                b_shape,
            )
        ):
            raise ValueError(
                "A/B CSR tensors and shapes are required when prepared is not provided"
            )
        torch.cuda.synchronize()
        t_prepare0 = time.perf_counter()
        prepared = prepare_spgemm_csr(
            a_data,
            a_indices,
            a_indptr,
            a_shape,
            b_data,
            b_indices,
            b_indptr,
            b_shape,
        )
        torch.cuda.synchronize()
        prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0
    elif not isinstance(prepared, SpGEMMPrepared):
        raise TypeError("prepared must be a SpGEMMPrepared instance")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c_data, c_indices, c_indptr, stage_meta = _run_spgemm_prepared(prepared)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if out is not None:
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise TypeError("out must be a tuple/list of (data, indices, indptr)")
        out_data, out_indices, out_indptr = out
        if not out_data.is_cuda or not out_indices.is_cuda or not out_indptr.is_cuda:
            raise ValueError("out data/indices/indptr must be CUDA tensors")
        if (
            out_data.device != c_data.device
            or out_indices.device != c_indices.device
            or out_indptr.device != c_indptr.device
        ):
            raise ValueError("out data/indices/indptr must be on the same CUDA device as computed C")
        if out_data.shape != c_data.shape or out_data.dtype != c_data.dtype:
            raise ValueError("out data shape/dtype must match computed C data")
        if out_indices.shape != c_indices.shape or out_indices.dtype != c_indices.dtype:
            raise ValueError("out indices shape/dtype must match computed C indices")
        if out_indptr.shape != c_indptr.shape or out_indptr.dtype != c_indptr.dtype:
            raise ValueError("out indptr shape/dtype must match computed C indptr")
        out_data.copy_(c_data)
        out_indices.copy_(c_indices)
        out_indptr.copy_(c_indptr)
        c_data, c_indices, c_indptr = out_data, out_indices, out_indptr

    result = (c_data, c_indices, c_indptr, (prepared.n_rows, prepared.n_cols))
    if return_time and return_meta:
        meta = {
            "prepare_ms": prepare_ms,
            "count_ms": stage_meta["count_ms"],
            "fill_ms": stage_meta["fill_ms"],
            "triton_ms": elapsed_ms,
            "hash_capacity_hint": prepared.hash_capacity_hint,
        }
        return result, elapsed_ms, meta
    if return_time:
        return result, elapsed_ms
    if return_meta:
        meta = {
            "prepare_ms": prepare_ms,
            "count_ms": stage_meta["count_ms"],
            "fill_ms": stage_meta["fill_ms"],
            "hash_capacity_hint": prepared.hash_capacity_hint,
        }
        return result, meta
    return result


def _csr_to_sorted_pairs(data, indices, indptr, n_cols):
    n_rows = int(indptr.numel()) - 1
    row_counts = indptr[1:] - indptr[:-1]
    rows = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        row_counts,
    )
    cols = indices.to(torch.int64)
    keys = rows * max(1, int(n_cols)) + cols
    if keys.numel() == 0:
        return keys, data
    order = torch.argsort(keys)
    return keys[order], data[order]


def _spgemm_pairwise_summary(candidate, reference, value_dtype):
    c_data, c_indices, c_indptr, c_shape = candidate
    r_data, r_indices, r_indptr, r_shape = reference
    if c_shape != r_shape:
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": f"shape mismatch {c_shape} vs {r_shape}",
        }
    c_keys, c_vals = _csr_to_sorted_pairs(c_data, c_indices, c_indptr, c_shape[1])
    r_keys, r_vals = _csr_to_sorted_pairs(r_data, r_indices, r_indptr, r_shape[1])
    if c_keys.numel() != r_keys.numel():
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": f"nnz mismatch {c_keys.numel()} vs {r_keys.numel()}",
        }
    if c_keys.numel() > 0 and not torch.equal(c_keys, r_keys):
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": "sparsity pattern mismatch",
        }
    if c_vals.numel() == 0:
        return {
            "match": True,
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "status": "ok",
        }
    abs_diff = torch.abs(c_vals - r_vals)
    max_abs = float(torch.max(abs_diff).item())
    ref_max = float(torch.max(torch.abs(r_vals)).item())
    max_rel = 0.0 if ref_max == 0.0 else max_abs / ref_max
    atol, rtol = _tolerance_for_dtype(value_dtype)
    match = bool(torch.allclose(c_vals, r_vals, atol=atol, rtol=rtol))
    return {
        "match": match,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "status": "ok" if match else "value mismatch",
    }


def _to_torch_csr(data, indices, indptr, shape):
    return torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data,
        size=shape,
        device=data.device,
    )


def _torch_sparse_to_csr(tensor):
    if tensor.layout == torch.sparse_csr:
        indptr = tensor.crow_indices().to(torch.int64).contiguous()
        indices = tensor.col_indices().to(torch.int32).contiguous()
        data = tensor.values().contiguous()
        shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        return data, indices, indptr, shape
    if tensor.layout == torch.sparse_coo:
        t = tensor.coalesce()
        rows = t.indices()[0].to(torch.int64)
        cols = t.indices()[1].to(torch.int64)
        vals = t.values()
        n_rows, n_cols = int(t.shape[0]), int(t.shape[1])
        if rows.numel() == 0:
            return (
                torch.empty(0, dtype=vals.dtype, device=vals.device),
                torch.empty(0, dtype=torch.int32, device=vals.device),
                torch.zeros(n_rows + 1, dtype=torch.int64, device=vals.device),
                (n_rows, n_cols),
            )
        key = rows * max(1, n_cols) + cols
        order = torch.argsort(key)
        rows = rows[order]
        cols = cols[order]
        vals = vals[order]
        row_counts = torch.bincount(rows, minlength=n_rows)
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=vals.device)
        indptr[1:] = torch.cumsum(row_counts, dim=0)
        return vals, cols.to(torch.int32), indptr, (n_rows, n_cols)
    raise TypeError(f"Unsupported sparse layout: {tensor.layout}")


def benchmark_spgemm_case(
    n_rows=1024,
    n_inner=1024,
    n_cols=1024,
    nnz_a=16384,
    nnz_b=16384,
    value_dtype=torch.float32,
    warmup=10,
    iters=30,
    run_cusparse=True,
):
    """Benchmark CSR SpGEMM and compare with torch/cuSPARSE baselines."""
    if value_dtype not in SUPPORTED_SPGEMM_VALUE_DTYPES:
        raise TypeError("value_dtype must be torch.float32 or torch.float64")
    device = torch.device("cuda")
    a_data, a_indices, a_indptr = _build_random_csr(
        n_rows, n_inner, nnz_a, value_dtype, torch.int32, device
    )
    b_data, b_indices, b_indptr = _build_random_csr(
        n_inner, n_cols, nnz_b, value_dtype, torch.int32, device
    )

    prepared = prepare_spgemm_csr(
        a_data, a_indices, a_indptr, (n_rows, n_inner),
        b_data, b_indices, b_indptr, (n_inner, n_cols),
    )
    op = lambda: flagsparse_spgemm_csr(prepared=prepared, return_time=False)
    triton_result, triton_ms = _benchmark_cuda_op(op, warmup=warmup, iters=iters)

    a_t = _to_torch_csr(a_data, a_indices, a_indptr, (n_rows, n_inner))
    b_t = _to_torch_csr(b_data, b_indices, b_indptr, (n_inner, n_cols))

    pytorch_reason = None
    pytorch_ms = None
    pytorch_result = None
    try:
        torch_op = lambda: torch.sparse.mm(a_t, b_t)
        pytorch_sparse, pytorch_ms = _benchmark_cuda_op(torch_op, warmup=warmup, iters=iters)
        pytorch_result = _torch_sparse_to_csr(pytorch_sparse)
    except Exception as exc:
        pytorch_reason = str(exc)
        a_coo = a_t.to_sparse_coo().coalesce()
        b_coo = b_t.to_sparse_coo().coalesce()
        torch_op = lambda: torch.sparse.mm(a_coo, b_coo)
        pytorch_sparse, pytorch_ms = _benchmark_cuda_op(torch_op, warmup=warmup, iters=iters)
        pytorch_result = _torch_sparse_to_csr(pytorch_sparse)

    triton_summary = _spgemm_pairwise_summary(triton_result, pytorch_result, value_dtype)

    cusparse_ms = None
    cusparse_reason = None
    cusparse_match = None
    if run_cusparse:
        if cp is None or cpx_sparse is None:
            cusparse_reason = "CuPy/cuSPARSE is not available"
        else:
            try:
                a_cp = cpx_sparse.csr_matrix(
                    (_cupy_from_torch(a_data), _cupy_from_torch(a_indices.to(torch.int64)), _cupy_from_torch(a_indptr.to(torch.int64))),
                    shape=(n_rows, n_inner),
                )
                b_cp = cpx_sparse.csr_matrix(
                    (_cupy_from_torch(b_data), _cupy_from_torch(b_indices.to(torch.int64)), _cupy_from_torch(b_indptr.to(torch.int64))),
                    shape=(n_inner, n_cols),
                )
                c_cp, cusparse_ms = _benchmark_cuda_op(lambda: a_cp @ b_cp, warmup=warmup, iters=iters)
                c_coo = c_cp.tocoo()
                rows = _torch_from_cupy(c_coo.row).to(torch.int64)
                cols = _torch_from_cupy(c_coo.col).to(torch.int64)
                vals = _torch_from_cupy(c_coo.data).to(value_dtype)
                c_t = torch.sparse_coo_tensor(
                    torch.stack([rows, cols]), vals, (n_rows, n_cols), device=device
                ).coalesce()
                c_ref = _torch_sparse_to_csr(c_t)
                cusparse_match = _spgemm_pairwise_summary(triton_result, c_ref, value_dtype)["match"]
            except Exception as exc:
                cusparse_reason = str(exc)

    return {
        "parameters": {
            "n_rows": n_rows,
            "n_inner": n_inner,
            "n_cols": n_cols,
            "nnz_a": nnz_a,
            "nnz_b": nnz_b,
            "value_dtype": str(value_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "triton_ms": triton_ms,
            "pytorch_ms": pytorch_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": (pytorch_ms / triton_ms if (pytorch_ms and triton_ms > 0) else None),
            "triton_speedup_vs_cusparse": (cusparse_ms / triton_ms if (cusparse_ms and triton_ms > 0) else None),
        },
        "verification": {
            "triton_match_pytorch": triton_summary["match"],
            "triton_max_abs_error": triton_summary["max_abs_error"],
            "triton_max_relative_error": triton_summary["max_relative_error"],
            "cusparse_match_pytorch": cusparse_match,
        },
        "backend_status": {
            "pytorch_unavailable_reason": pytorch_reason,
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {
            "triton": triton_result,
            "pytorch": pytorch_result,
        },
    }
