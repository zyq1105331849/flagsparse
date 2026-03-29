"""COO SpMV **without CSR / indptr** (fp32/fp64).

- ``sort_by_row=True`` (default): lex-sort (row,col), build compact **row-run** offsets
  ``seg_starts`` (length #runs+1, not ``n_rows+1``), one Triton program per run — register
  reduction + single ``tl.store`` per output row (no atomics on ``y``).
- ``sort_by_row=False``: grid over NNZ with ``tl.atomic_add`` (slower / contentions).

Storage: sorted ``data, row, col`` plus optional ``seg_starts`` int32 vector — never ``indptr``.
"""

from ._common import *

import time

import triton
import triton.language as tl


class PreparedCoo:
    """Sorted COO + optional row-run bounds ``seg_starts``; no CSR indptr."""

    __slots__ = (
        "data",
        "row",
        "col",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "seg_starts",
        "n_segs",
        "use_seg_kernel",
    )

    def __init__(self, data, row, col, shape, seg_starts=None):
        self.data = data
        self.row = row
        self.col = col
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows, self.n_cols = self.shape
        self.nnz = int(data.numel())
        self.seg_starts = seg_starts
        if seg_starts is None:
            self.n_segs = 0
            self.use_seg_kernel = False
        else:
            self.n_segs = int(seg_starts.numel()) - 1
            self.use_seg_kernel = self.n_segs > 0


@triton.jit
def _spmv_coo_seg_f32(
    data_ptr,
    col_ptr,
    row_ptr,
    x_ptr,
    y_ptr,
    seg_starts_ptr,
    n_segs,
    BLOCK_INNER: tl.constexpr,
):
    seg = tl.program_id(0)
    if seg >= n_segs:
        return
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_id = tl.load(row_ptr + start)
    acc = tl.zeros((), dtype=tl.float32)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK_INNER)
        m = offs < end
        v = tl.load(data_ptr + offs, mask=m, other=0.0)
        c = tl.load(col_ptr + offs, mask=m, other=0)
        xv = tl.load(x_ptr + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, v * xv, 0.0))
        pos += BLOCK_INNER
    tl.store(y_ptr + row_id, acc)


@triton.jit
def _spmv_coo_seg_f64(
    data_ptr,
    col_ptr,
    row_ptr,
    x_ptr,
    y_ptr,
    seg_starts_ptr,
    n_segs,
    BLOCK_INNER: tl.constexpr,
):
    seg = tl.program_id(0)
    if seg >= n_segs:
        return
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_id = tl.load(row_ptr + start)
    acc = tl.zeros((), dtype=tl.float64)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK_INNER)
        m = offs < end
        v = tl.load(data_ptr + offs, mask=m, other=0.0)
        c = tl.load(col_ptr + offs, mask=m, other=0)
        xv = tl.load(x_ptr + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, v * xv, 0.0))
        pos += BLOCK_INNER
    tl.store(y_ptr + row_id, acc)


@triton.jit
def _spmv_coo_atomic_f32(
    data_ptr,
    row_ptr,
    col_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    v = tl.load(data_ptr + offs, mask=m, other=0.0)
    xv = tl.load(x_ptr + c, mask=m, other=0.0)
    contrib = tl.where(m, v * xv, 0.0).to(tl.float32)
    tl.atomic_add(y_ptr + r, contrib, mask=m, sem="relaxed")


@triton.jit
def _spmv_coo_atomic_f64(
    data_ptr,
    row_ptr,
    col_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    v = tl.load(data_ptr + offs, mask=m, other=0.0)
    xv = tl.load(x_ptr + c, mask=m, other=0.0)
    contrib = tl.where(m, v * xv, 0.0).to(tl.float64)
    tl.atomic_add(y_ptr + r, contrib, mask=m, sem="relaxed")


def _sort_coo_lex_inplace(data, row, col, n_cols):
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    if data.numel() == 0:
        return data.contiguous(), row64, col64
    key = row64 * max(1, int(n_cols)) + col64
    order = torch.argsort(key)
    return (
        data[order].contiguous(),
        row64[order].contiguous(),
        col64[order].contiguous(),
    )


def _seg_starts_from_sorted_rows(row_i32, nnz, device):
    """Boundaries of constant-row runs in sorted COO → int32[n_runs+1]."""
    if nnz == 0:
        return None
    diff = row_i32[1:] != row_i32[:-1]
    breaks = torch.nonzero(diff, as_tuple=False).flatten().to(torch.int32) + 1
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            breaks,
            torch.tensor([nnz], dtype=torch.int32, device=device),
        ]
    )


def _prepare_coo_tensors(data, row, col, shape, sort_by_row):
    if not all(torch.is_tensor(t) for t in (data, row, col)):
        raise TypeError("data, row, col must all be torch.Tensor")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if not all(t.is_cuda for t in (data, row, col)):
        raise ValueError("data, row, col must be CUDA tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have the same length")
    if data.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "this COO SpMV path supports float32/float64 only (use CSR for other dtypes)"
        )
    if sort_by_row:
        data, row64, col64 = _sort_coo_lex_inplace(data, row, col, n_cols)
        seg = _seg_starts_from_sorted_rows(
            row64.to(torch.int32), data.numel(), data.device
        )
    else:
        data = data.contiguous()
        row64 = row.to(torch.int64).contiguous()
        col64 = col.to(torch.int64).contiguous()
        seg = None
    if data.numel() > 0:
        if int(row64.min().item()) < 0 or int(row64.max().item()) >= n_rows:
            raise IndexError("row indices out of range")
        if int(col64.min().item()) < 0 or int(col64.max().item()) >= n_cols:
            raise IndexError("col indices out of range")
        if int(row64.max().item()) > _INDEX_LIMIT_INT32 or int(
            col64.max().item()
        ) > _INDEX_LIMIT_INT32:
            raise ValueError("indices exceed int32 Triton kernel range")
    kr = row64.to(torch.int32)
    kc = col64.to(torch.int32)
    return data, kr, kc, seg


def prepare_spmv_coo(data, row, col, shape, sort_by_row=True):
    """Cache sorted COO + row-run ``seg_starts`` when ``sort_by_row``. No ``indptr``."""
    d, kr, kc, seg = _prepare_coo_tensors(
        data, row, col, shape, sort_by_row
    )
    return PreparedCoo(d, kr, kc, shape, seg_starts=seg)


def _validate_x_coo(x, prepared):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    if x.numel() != prepared.n_cols:
        raise ValueError(
            f"x length must be n_cols={prepared.n_cols}, got {x.numel()}"
        )
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    return x.contiguous()


def _triton_spmv_coo_kernel(
    prepared, x, block_size, num_warps, block_inner
):
    dtype = prepared.data.dtype
    y = torch.zeros(prepared.n_rows, dtype=dtype, device=prepared.data.device)
    nnz = prepared.nnz
    if nnz == 0:
        return y
    if prepared.use_seg_kernel:
        ker = _spmv_coo_seg_f64 if dtype == torch.float64 else _spmv_coo_seg_f32
        grid = (prepared.n_segs,)
        ker[grid](
            prepared.data,
            prepared.col,
            prepared.row,
            x,
            y,
            prepared.seg_starts,
            prepared.n_segs,
            BLOCK_INNER=block_inner,
            num_warps=1,
        )
        return y
    ker = _spmv_coo_atomic_f64 if dtype == torch.float64 else _spmv_coo_atomic_f32
    grid = (triton.cdiv(nnz, block_size),)
    ker[grid](
        prepared.data,
        prepared.row,
        prepared.col,
        x,
        y,
        nnz,
        BLOCK=block_size,
        num_warps=num_warps,
    )
    return y


def flagsparse_spmv_coo(
    data=None,
    row=None,
    col=None,
    x=None,
    shape=None,
    out=None,
    return_time=False,
    prepared=None,
    sort_by_row=True,
    block_size=256,
    num_warps=4,
    block_inner=128,
):
    """COO SpMV with no CSR indptr. See module docstring.

    ``block_inner``: tile for the row-run kernel (``sort_by_row=True``).
    ``block_size`` / ``num_warps``: grid over NNZ when ``sort_by_row=False`` (atomics).
    """
    if prepared is None:
        if any(a is None for a in (data, row, col, x, shape)):
            raise ValueError(
                "data, row, col, x, shape required when prepared is None"
            )
        prepared = prepare_spmv_coo(
            data, row, col, shape, sort_by_row=sort_by_row
        )
    else:
        if x is None:
            raise TypeError("x is required when prepared is set")
        if shape is None:
            shape = prepared.shape
        sh = (int(shape[0]), int(shape[1]))
        if sh != prepared.shape:
            raise ValueError(
                f"shape {sh} does not match prepared.shape {prepared.shape}"
            )
    x = _validate_x_coo(x, prepared)
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a power of 2 in [1, 32]")
    if block_inner <= 0 or (block_inner & (block_inner - 1)) != 0:
        raise ValueError("block_inner must be a positive power of 2")
    t0 = None
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    y = _triton_spmv_coo_kernel(
        prepared,
        x,
        block_size=block_size,
        num_warps=num_warps,
        block_inner=block_inner,
    )
    elapsed_ms = None
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != y.shape or out.dtype != y.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(y)
        y = out
    if return_time:
        return y, elapsed_ms
    return y