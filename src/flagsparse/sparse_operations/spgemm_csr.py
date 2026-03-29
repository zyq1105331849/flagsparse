"""CSR SpGEMM (A@B) with two-phase structure/value build."""

from ._common import *

SUPPORTED_SPGEMM_VALUE_DTYPES = (torch.float32, torch.float64)
_SPGEMM_COUNT_MAX_EXPANDED = 2_000_000
_SPGEMM_FILL_MAX_EXPANDED = 1_200_000
_SPGEMM_MAX_ROWS_PER_CHUNK = 4096
_SPGEMM_BUCKET_SHORT = 0
_SPGEMM_BUCKET_MEDIUM = 1
_SPGEMM_BUCKET_LONG = 2
_SPGEMM_BUCKET_ORDER = (
    _SPGEMM_BUCKET_SHORT,
    _SPGEMM_BUCKET_MEDIUM,
    _SPGEMM_BUCKET_LONG,
)
_SPGEMM_BUCKET_LABELS = {
    _SPGEMM_BUCKET_SHORT: "short",
    _SPGEMM_BUCKET_MEDIUM: "medium",
    _SPGEMM_BUCKET_LONG: "long",
}
_SPGEMM_BUCKET_COUNT_BUDGETS = {
    _SPGEMM_BUCKET_SHORT: 4_000_000,
    _SPGEMM_BUCKET_MEDIUM: _SPGEMM_COUNT_MAX_EXPANDED,
    _SPGEMM_BUCKET_LONG: 300_000,
}
_SPGEMM_BUCKET_FILL_BUDGETS = {
    _SPGEMM_BUCKET_SHORT: 2_400_000,
    _SPGEMM_BUCKET_MEDIUM: _SPGEMM_FILL_MAX_EXPANDED,
    _SPGEMM_BUCKET_LONG: 200_000,
}
_SPGEMM_BUCKET_MAX_ROWS = {
    _SPGEMM_BUCKET_SHORT: 8192,
    _SPGEMM_BUCKET_MEDIUM: _SPGEMM_MAX_ROWS_PER_CHUNK,
    _SPGEMM_BUCKET_LONG: 256,
}
_SPGEMM_LONG_ROW_SLICE_EXPANDED = 200_000


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
        "row_work_ready",
        "bucket_rows",
        "count_chunks",
        "fill_chunks",
        "count_chunks_by_bucket",
        "fill_chunks_by_bucket",
        "long_row_slice_expanded",
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
        row_work_ready,
        bucket_rows,
        count_chunks,
        fill_chunks,
        count_chunks_by_bucket,
        fill_chunks_by_bucket,
        long_row_slice_expanded,
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
        self.row_work_ready = bool(row_work_ready)
        self.bucket_rows = bucket_rows
        self.count_chunks = count_chunks
        self.fill_chunks = fill_chunks
        self.count_chunks_by_bucket = count_chunks_by_bucket
        self.fill_chunks_by_bucket = fill_chunks_by_bucket
        self.long_row_slice_expanded = int(long_row_slice_expanded)
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
    bucket = torch.where(
        a_row_work > 4096,
        torch.full_like(bucket, _SPGEMM_BUCKET_LONG),
        bucket,
    )
    bucket = torch.where(
        (a_row_work > 256) & (a_row_work <= 4096),
        torch.full_like(bucket, _SPGEMM_BUCKET_MEDIUM),
        bucket,
    )
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
    analyze_rows=True,
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
        row_bucket = torch.empty(0, dtype=torch.int8, device=a_data.device)
        hash_capacity_hint = 256
        row_work_ready = True
    elif analyze_rows:
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
        row_work_ready = True
    else:
        row_work = torch.zeros(n_rows, dtype=torch.int32, device=a_data.device)
        row_bucket = torch.zeros(n_rows, dtype=torch.int8, device=a_data.device)
        hash_capacity_hint = 256
        row_work_ready = False
    bucket_rows = _build_bucket_rows(row_bucket, a_data.device)
    count_chunks_by_bucket = {}
    fill_chunks_by_bucket = {}
    count_chunks = None
    fill_chunks = None
    if row_work_ready:
        for bucket_id in _SPGEMM_BUCKET_ORDER:
            rows = bucket_rows[bucket_id]
            count_chunks_by_bucket[bucket_id] = _build_row_id_chunks(
                row_work,
                rows,
                max_expanded=_SPGEMM_BUCKET_COUNT_BUDGETS[bucket_id],
                max_rows_per_chunk=_SPGEMM_BUCKET_MAX_ROWS[bucket_id],
            )
            fill_chunks_by_bucket[bucket_id] = _build_row_id_chunks(
                row_work,
                rows,
                max_expanded=_SPGEMM_BUCKET_FILL_BUDGETS[bucket_id],
                max_rows_per_chunk=_SPGEMM_BUCKET_MAX_ROWS[bucket_id],
            )
        count_chunks = _compose_ordered_chunks(count_chunks_by_bucket)
        fill_chunks = _compose_ordered_chunks(fill_chunks_by_bucket)
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
        row_work_ready=row_work_ready,
        bucket_rows=bucket_rows,
        count_chunks=count_chunks,
        fill_chunks=fill_chunks,
        count_chunks_by_bucket=count_chunks_by_bucket if row_work_ready else None,
        fill_chunks_by_bucket=fill_chunks_by_bucket if row_work_ready else None,
        long_row_slice_expanded=_SPGEMM_LONG_ROW_SLICE_EXPANDED,
        hash_capacity_hint=hash_capacity_hint,
        block_nnz=block_nnz,
    )


def _ensure_row_work(prepared):
    if prepared.row_work_ready:
        return
    if prepared.a_data.numel() == 0 or prepared.b_data.numel() == 0:
        prepared.a_row_work = torch.zeros(prepared.n_rows, dtype=torch.int32, device=prepared.a_data.device)
        prepared.row_bucket = torch.zeros(prepared.n_rows, dtype=torch.int8, device=prepared.a_data.device)
        prepared.hash_capacity_hint = 256
        prepared.row_work_ready = True
        _refresh_row_chunks(prepared)
        return
    row_work = torch.empty(prepared.n_rows, dtype=torch.int32, device=prepared.a_data.device)
    _spgemm_row_work_kernel[(prepared.n_rows,)](
        prepared.a_indptr,
        prepared.a_indices,
        prepared.b_indptr,
        row_work,
        prepared.n_rows,
        BLOCK_NNZ=int(prepared.block_nnz),
    )
    prepared.a_row_work = row_work
    prepared.row_bucket = _build_row_bucket(row_work)
    prepared.hash_capacity_hint = _estimate_hash_capacity(row_work)
    prepared.row_work_ready = True
    _refresh_row_chunks(prepared)


def _build_row_chunks(row_work, max_expanded, max_rows_per_chunk):
    n_rows = int(row_work.numel())
    if n_rows == 0:
        return []
    all_rows = torch.arange(n_rows, device=row_work.device, dtype=torch.int64)
    return _build_row_id_chunks(row_work, all_rows, max_expanded, max_rows_per_chunk)


def _build_bucket_rows(row_bucket, device):
    out = {}
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        rows = torch.nonzero(row_bucket == bucket_id, as_tuple=False).flatten()
        out[bucket_id] = rows.to(device=device, dtype=torch.int64)
    return out


def _build_row_id_chunks(row_work, row_ids, max_expanded, max_rows_per_chunk):
    if row_ids.numel() == 0:
        return []
    work_host = row_work[row_ids].detach().to("cpu", dtype=torch.int64).tolist()
    chunks = []
    idx = 0
    total_rows = len(work_host)
    while idx < total_rows:
        start_idx = idx
        acc = 0
        taken = 0
        while idx < total_rows and taken < int(max_rows_per_chunk):
            w = int(work_host[idx])
            if taken > 0 and acc + w > int(max_expanded):
                break
            acc += w
            idx += 1
            taken += 1
            if taken == 1 and w > int(max_expanded):
                break
        if idx == start_idx:
            idx += 1
        chunks.append(row_ids[start_idx:idx].contiguous())
    return chunks


def _compose_ordered_chunks(chunks_by_bucket):
    ordered = []
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        ordered.extend(chunks_by_bucket.get(bucket_id, []))
    return ordered


def _refresh_row_chunks(prepared):
    prepared.bucket_rows = _build_bucket_rows(prepared.row_bucket, prepared.a_data.device)
    prepared.count_chunks_by_bucket = {}
    prepared.fill_chunks_by_bucket = {}
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        rows = prepared.bucket_rows[bucket_id]
        prepared.count_chunks_by_bucket[bucket_id] = _build_row_id_chunks(
            prepared.a_row_work,
            rows,
            max_expanded=_SPGEMM_BUCKET_COUNT_BUDGETS[bucket_id],
            max_rows_per_chunk=_SPGEMM_BUCKET_MAX_ROWS[bucket_id],
        )
        prepared.fill_chunks_by_bucket[bucket_id] = _build_row_id_chunks(
            prepared.a_row_work,
            rows,
            max_expanded=_SPGEMM_BUCKET_FILL_BUDGETS[bucket_id],
            max_rows_per_chunk=_SPGEMM_BUCKET_MAX_ROWS[bucket_id],
        )
    prepared.count_chunks = _compose_ordered_chunks(prepared.count_chunks_by_bucket)
    prepared.fill_chunks = _compose_ordered_chunks(prepared.fill_chunks_by_bucket)


def _expand_rows_contrib(prepared, row_ids, need_values):
    device = prepared.a_data.device
    if row_ids.numel() == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    row_ids = row_ids.to(torch.int64)
    row_nnz = (prepared.a_indptr[row_ids + 1] - prepared.a_indptr[row_ids]).to(torch.int64)
    total_a = int(row_nnz.sum().item())
    if total_a == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    owner = torch.repeat_interleave(
        torch.arange(row_ids.numel(), device=device, dtype=torch.int64),
        row_nnz,
    )
    prefix = torch.cumsum(row_nnz, dim=0)
    base = prefix - row_nnz
    intra = (
        torch.arange(total_a, device=device, dtype=torch.int64)
        - torch.repeat_interleave(base, row_nnz)
    )
    row_starts = prepared.a_indptr[row_ids]
    a_pos = row_starts[owner] + intra
    rows = row_ids[owner]
    a_cols = prepared.a_indices[a_pos].to(torch.int64)
    b_starts = prepared.b_indptr[a_cols]
    b_ends = prepared.b_indptr[a_cols + 1]
    b_counts = b_ends - b_starts
    total = int(b_counts.sum().item())
    if total == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    b_owner = torch.repeat_interleave(
        torch.arange(a_cols.numel(), device=device, dtype=torch.int64),
        b_counts,
    )
    rows_expanded = rows[b_owner]
    prefix = torch.cumsum(b_counts, dim=0)
    base = prefix - b_counts
    starts_rep = torch.repeat_interleave(b_starts, b_counts)
    intra = (
        torch.arange(total, device=device, dtype=torch.int64)
        - torch.repeat_interleave(base, b_counts)
    )
    b_pos = starts_rep + intra
    cols = prepared.b_indices[b_pos].to(torch.int64)
    keys = rows_expanded * max(1, prepared.n_cols) + cols
    if not need_values:
        return keys, None

    a_vals = prepared.a_data[a_pos]
    vals = a_vals[b_owner] * prepared.b_data[b_pos]
    return keys, vals


def _expand_single_row_slice_contrib(prepared, row, a_ptr_start, a_ptr_end, need_values):
    device = prepared.a_data.device
    if a_ptr_end <= a_ptr_start:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    a_pos = torch.arange(a_ptr_start, a_ptr_end, device=device, dtype=torch.int64)
    a_cols = prepared.a_indices[a_pos].to(torch.int64)
    b_starts = prepared.b_indptr[a_cols]
    b_ends = prepared.b_indptr[a_cols + 1]
    b_counts = b_ends - b_starts
    total = int(b_counts.sum().item())
    if total == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        if need_values:
            empty_v = torch.empty(0, dtype=prepared.a_data.dtype, device=device)
            return empty_i64, empty_v
        return empty_i64, None

    owner = torch.repeat_interleave(
        torch.arange(a_pos.numel(), device=device, dtype=torch.int64),
        b_counts,
    )
    prefix = torch.cumsum(b_counts, dim=0)
    base = prefix - b_counts
    starts_rep = torch.repeat_interleave(b_starts, b_counts)
    intra = (
        torch.arange(total, device=device, dtype=torch.int64)
        - torch.repeat_interleave(base, b_counts)
    )
    b_pos = starts_rep + intra
    cols = prepared.b_indices[b_pos].to(torch.int64)
    keys = int(row) * max(1, prepared.n_cols) + cols
    if not need_values:
        return keys, None
    a_vals = prepared.a_data[a_pos]
    vals = a_vals[owner] * prepared.b_data[b_pos]
    return keys, vals


def _iter_row_a_slices(prepared, row, max_expanded):
    start = int(prepared.a_indptr[row].item())
    end = int(prepared.a_indptr[row + 1].item())
    if end <= start:
        return []
    a_cols = prepared.a_indices[start:end].to(torch.int64)
    b_counts = (prepared.b_indptr[a_cols + 1] - prepared.b_indptr[a_cols]).to(torch.int64)
    counts_host = b_counts.detach().to("cpu").tolist()
    slices = []
    idx = 0
    total = len(counts_host)
    while idx < total:
        seg_start = idx
        acc = 0
        while idx < total:
            w = int(counts_host[idx])
            if idx > seg_start and acc + w > int(max_expanded):
                break
            acc += w
            idx += 1
            if idx == seg_start + 1 and w > int(max_expanded):
                break
        slices.append((start + seg_start, start + idx))
    return slices


def _reduce_sorted_keys_vals(keys_sorted, vals_sorted, out_dtype):
    if keys_sorted.numel() == 0:
        return keys_sorted, vals_sorted
    uniq_keys, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    if uniq_keys.numel() == keys_sorted.numel():
        return uniq_keys, vals_sorted.to(out_dtype)
    acc_dtype = torch.float64 if out_dtype == torch.float32 else vals_sorted.dtype
    vals_acc = vals_sorted.to(acc_dtype)
    prefix = torch.cumsum(vals_acc, dim=0)
    end_idx = torch.cumsum(counts.to(torch.int64), dim=0) - 1
    seg_end = prefix[end_idx]
    seg_begin = torch.zeros_like(seg_end)
    if seg_end.numel() > 1:
        seg_begin[1:] = prefix[end_idx[:-1]]
    uniq_vals = (seg_end - seg_begin).to(out_dtype)
    return uniq_keys, uniq_vals


def _sort_reduce_pairs(keys, vals, out_dtype):
    if keys.numel() == 0:
        return keys, vals
    order = torch.argsort(keys)
    keys_sorted = keys[order]
    vals_sorted = vals[order]
    return _reduce_sorted_keys_vals(keys_sorted, vals_sorted, out_dtype)


def _spgemm_count_phase(prepared):
    n_rows = prepared.n_rows
    device = prepared.a_data.device
    row_nnz_c = torch.zeros(n_rows, dtype=torch.int64, device=device)
    bucket_ms = {bucket_id: 0.0 for bucket_id in _SPGEMM_BUCKET_ORDER}
    long_row_sliced = 0
    if prepared.count_chunks_by_bucket is None:
        _refresh_row_chunks(prepared)
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        chunks = prepared.count_chunks_by_bucket.get(bucket_id, [])
        for row_ids in chunks:
            torch.cuda.synchronize()
            t_bucket0 = time.perf_counter()
            if bucket_id != _SPGEMM_BUCKET_LONG:
                keys, _ = _expand_rows_contrib(prepared, row_ids, need_values=False)
                if keys.numel() > 0:
                    uniq_keys = torch.unique(keys, sorted=True)
                    uniq_rows = torch.div(
                        uniq_keys,
                        max(1, prepared.n_cols),
                        rounding_mode="floor",
                    )
                    rows_unique, counts = torch.unique_consecutive(
                        uniq_rows, return_counts=True
                    )
                    row_nnz_c[rows_unique] = counts.to(torch.int64)
            else:
                for row in row_ids.detach().to("cpu", dtype=torch.int64).tolist():
                    slices = _iter_row_a_slices(
                        prepared,
                        int(row),
                        max_expanded=prepared.long_row_slice_expanded,
                    )
                    if len(slices) > 1:
                        long_row_sliced += 1
                    keys_parts = []
                    for a_start, a_end in slices:
                        keys, _ = _expand_single_row_slice_contrib(
                            prepared,
                            int(row),
                            a_start,
                            a_end,
                            need_values=False,
                        )
                        if keys.numel() == 0:
                            continue
                        keys_parts.append(torch.unique(keys, sorted=True))
                    if not keys_parts:
                        row_nnz_c[int(row)] = 0
                        continue
                    uniq_row_keys = torch.unique(torch.cat(keys_parts), sorted=True)
                    row_nnz_c[int(row)] = int(uniq_row_keys.numel())
            torch.cuda.synchronize()
            bucket_ms[bucket_id] += (time.perf_counter() - t_bucket0) * 1000.0
    meta = {
        "bucket_count_ms_short": bucket_ms[_SPGEMM_BUCKET_SHORT],
        "bucket_count_ms_medium": bucket_ms[_SPGEMM_BUCKET_MEDIUM],
        "bucket_count_ms_long": bucket_ms[_SPGEMM_BUCKET_LONG],
        "long_row_sliced_count_count": int(long_row_sliced),
    }
    return row_nnz_c, meta


def _spgemm_fill_phase(prepared, c_indptr):
    nnz_c = int(c_indptr[-1].item())
    device = prepared.a_data.device
    c_data = torch.empty(nnz_c, dtype=prepared.a_data.dtype, device=device)
    c_indices = torch.empty(nnz_c, dtype=torch.int32, device=device)
    bucket_ms = {bucket_id: 0.0 for bucket_id in _SPGEMM_BUCKET_ORDER}
    long_row_sliced = 0
    if nnz_c == 0:
        meta = {
            "bucket_fill_ms_short": bucket_ms[_SPGEMM_BUCKET_SHORT],
            "bucket_fill_ms_medium": bucket_ms[_SPGEMM_BUCKET_MEDIUM],
            "bucket_fill_ms_long": bucket_ms[_SPGEMM_BUCKET_LONG],
            "long_row_sliced_count_fill": int(long_row_sliced),
        }
        return c_data, c_indices, meta

    if prepared.fill_chunks_by_bucket is None:
        _refresh_row_chunks(prepared)
    for bucket_id in _SPGEMM_BUCKET_ORDER:
        chunks = prepared.fill_chunks_by_bucket.get(bucket_id, [])
        for row_ids in chunks:
            torch.cuda.synchronize()
            t_bucket0 = time.perf_counter()
            if bucket_id != _SPGEMM_BUCKET_LONG:
                keys, vals = _expand_rows_contrib(prepared, row_ids, need_values=True)
                if keys.numel() > 0:
                    uniq_keys, uniq_vals = _sort_reduce_pairs(
                        keys,
                        vals,
                        out_dtype=prepared.a_data.dtype,
                    )
                    uniq_rows = torch.div(
                        uniq_keys,
                        max(1, prepared.n_cols),
                        rounding_mode="floor",
                    )
                    uniq_cols = (uniq_keys - uniq_rows * max(1, prepared.n_cols)).to(torch.int32)
                    _, row_counts = torch.unique_consecutive(uniq_rows, return_counts=True)
                    row_offsets = torch.cumsum(row_counts.to(torch.int64), dim=0) - row_counts.to(torch.int64)
                    local_pos = (
                        torch.arange(uniq_keys.numel(), device=device, dtype=torch.int64)
                        - torch.repeat_interleave(row_offsets, row_counts)
                    )
                    dst = c_indptr[uniq_rows] + local_pos
                    c_indices[dst] = uniq_cols
                    c_data[dst] = uniq_vals
            else:
                for row in row_ids.detach().to("cpu", dtype=torch.int64).tolist():
                    row = int(row)
                    slices = _iter_row_a_slices(
                        prepared,
                        row,
                        max_expanded=prepared.long_row_slice_expanded,
                    )
                    if len(slices) > 1:
                        long_row_sliced += 1
                    key_parts = []
                    val_parts = []
                    for a_start, a_end in slices:
                        keys, vals = _expand_single_row_slice_contrib(
                            prepared,
                            row,
                            a_start,
                            a_end,
                            need_values=True,
                        )
                        if keys.numel() == 0:
                            continue
                        uniq_k, uniq_v = _sort_reduce_pairs(
                            keys,
                            vals,
                            out_dtype=prepared.a_data.dtype,
                        )
                        key_parts.append(uniq_k)
                        val_parts.append(uniq_v)
                    row_start = int(c_indptr[row].item())
                    row_end = int(c_indptr[row + 1].item())
                    row_nnz = row_end - row_start
                    if row_nnz == 0:
                        continue
                    if not key_parts:
                        raise RuntimeError(f"row {row} expected nnz={row_nnz} but got empty fill")
                    row_keys = torch.cat(key_parts)
                    row_vals = torch.cat(val_parts)
                    row_keys, row_vals = _sort_reduce_pairs(
                        row_keys,
                        row_vals,
                        out_dtype=prepared.a_data.dtype,
                    )
                    if row_keys.numel() != row_nnz:
                        raise RuntimeError(
                            f"row {row} fill nnz mismatch: expected {row_nnz}, got {row_keys.numel()}"
                        )
                    row_cols = (row_keys - row * max(1, prepared.n_cols)).to(torch.int32)
                    c_indices[row_start:row_end] = row_cols
                    c_data[row_start:row_end] = row_vals
            torch.cuda.synchronize()
            bucket_ms[bucket_id] += (time.perf_counter() - t_bucket0) * 1000.0

    meta = {
        "bucket_fill_ms_short": bucket_ms[_SPGEMM_BUCKET_SHORT],
        "bucket_fill_ms_medium": bucket_ms[_SPGEMM_BUCKET_MEDIUM],
        "bucket_fill_ms_long": bucket_ms[_SPGEMM_BUCKET_LONG],
        "long_row_sliced_count_fill": int(long_row_sliced),
    }
    return c_data, c_indices, meta


def _run_spgemm_prepared(prepared):
    _ensure_row_work(prepared)
    torch.cuda.synchronize()
    t_count0 = time.perf_counter()
    row_nnz_c, count_meta = _spgemm_count_phase(prepared)
    torch.cuda.synchronize()
    count_ms = (time.perf_counter() - t_count0) * 1000.0

    c_indptr = torch.empty(prepared.n_rows + 1, dtype=torch.int64, device=prepared.a_data.device)
    c_indptr[0] = 0
    if prepared.n_rows > 0:
        c_indptr[1:] = torch.cumsum(row_nnz_c, dim=0)

    torch.cuda.synchronize()
    t_fill0 = time.perf_counter()
    c_data, c_indices, fill_meta = _spgemm_fill_phase(prepared, c_indptr)
    torch.cuda.synchronize()
    fill_ms = (time.perf_counter() - t_fill0) * 1000.0

    bucket_ms_short = float(count_meta["bucket_count_ms_short"] + fill_meta["bucket_fill_ms_short"])
    bucket_ms_medium = float(count_meta["bucket_count_ms_medium"] + fill_meta["bucket_fill_ms_medium"])
    bucket_ms_long = float(count_meta["bucket_count_ms_long"] + fill_meta["bucket_fill_ms_long"])
    return c_data, c_indices, c_indptr, {
        "count_ms": count_ms,
        "fill_ms": fill_ms,
        "bucket_ms_short": bucket_ms_short,
        "bucket_ms_medium": bucket_ms_medium,
        "bucket_ms_long": bucket_ms_long,
        "bucket_count_ms_short": count_meta["bucket_count_ms_short"],
        "bucket_count_ms_medium": count_meta["bucket_count_ms_medium"],
        "bucket_count_ms_long": count_meta["bucket_count_ms_long"],
        "bucket_fill_ms_short": fill_meta["bucket_fill_ms_short"],
        "bucket_fill_ms_medium": fill_meta["bucket_fill_ms_medium"],
        "bucket_fill_ms_long": fill_meta["bucket_fill_ms_long"],
        "bucket_nrows_short": int(prepared.bucket_rows[_SPGEMM_BUCKET_SHORT].numel()),
        "bucket_nrows_medium": int(prepared.bucket_rows[_SPGEMM_BUCKET_MEDIUM].numel()),
        "bucket_nrows_long": int(prepared.bucket_rows[_SPGEMM_BUCKET_LONG].numel()),
        "long_row_sliced_count": int(
            max(
                count_meta["long_row_sliced_count_count"],
                fill_meta["long_row_sliced_count_fill"],
            )
        ),
    }


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
    """CSR SpGEMM: C = A @ B with CSR output (Triton-only main path)."""
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
            "bucket_ms_short": stage_meta["bucket_ms_short"],
            "bucket_ms_medium": stage_meta["bucket_ms_medium"],
            "bucket_ms_long": stage_meta["bucket_ms_long"],
            "bucket_count_ms_short": stage_meta["bucket_count_ms_short"],
            "bucket_count_ms_medium": stage_meta["bucket_count_ms_medium"],
            "bucket_count_ms_long": stage_meta["bucket_count_ms_long"],
            "bucket_fill_ms_short": stage_meta["bucket_fill_ms_short"],
            "bucket_fill_ms_medium": stage_meta["bucket_fill_ms_medium"],
            "bucket_fill_ms_long": stage_meta["bucket_fill_ms_long"],
            "bucket_nrows_short": stage_meta["bucket_nrows_short"],
            "bucket_nrows_medium": stage_meta["bucket_nrows_medium"],
            "bucket_nrows_long": stage_meta["bucket_nrows_long"],
            "long_row_sliced_count": stage_meta["long_row_sliced_count"],
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
            "bucket_ms_short": stage_meta["bucket_ms_short"],
            "bucket_ms_medium": stage_meta["bucket_ms_medium"],
            "bucket_ms_long": stage_meta["bucket_ms_long"],
            "bucket_count_ms_short": stage_meta["bucket_count_ms_short"],
            "bucket_count_ms_medium": stage_meta["bucket_count_ms_medium"],
            "bucket_count_ms_long": stage_meta["bucket_count_ms_long"],
            "bucket_fill_ms_short": stage_meta["bucket_fill_ms_short"],
            "bucket_fill_ms_medium": stage_meta["bucket_fill_ms_medium"],
            "bucket_fill_ms_long": stage_meta["bucket_fill_ms_long"],
            "bucket_nrows_short": stage_meta["bucket_nrows_short"],
            "bucket_nrows_medium": stage_meta["bucket_nrows_medium"],
            "bucket_nrows_long": stage_meta["bucket_nrows_long"],
            "long_row_sliced_count": stage_meta["long_row_sliced_count"],
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
