"""CSR SDDMM kernels and helpers."""

from ._common import *

SUPPORTED_SDDMM_VALUE_DTYPES = (torch.float32, torch.float64)
SUPPORTED_SDDMM_DIAGNOSTIC_VARIANTS = ("baseline", "acc64", "acc64_out64", "altreduce")


class SDDMMPrepared:
    """Prepared CSR pattern metadata for SDDMM."""

    __slots__ = (
        "indices",
        "indptr",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "row_ids",
        "block_k",
        "num_warps",
    )

    def __init__(self, indices, indptr, shape, row_ids, block_k, num_warps):
        self.indices = indices
        self.indptr = indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = self.shape[0]
        self.n_cols = self.shape[1]
        self.nnz = int(indices.numel())
        self.row_ids = row_ids
        self.block_k = int(block_k)
        self.num_warps = int(num_warps)


def _resolve_sddmm_launch_config(k):
    if k <= 32:
        return 32, 2
    if k <= 64:
        return 64, 4
    if k <= 128:
        return 64, 4
    return 128, 8


def _prepare_sddmm_csr_pattern(indices, indptr, shape):
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple")
    if indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("indices and indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if not indices.is_cuda or not indptr.is_cuda:
        raise ValueError("indices and indptr must be CUDA tensors")
    if indices.dtype != torch.int32:
        raise TypeError("indices dtype must be torch.int32")
    if indptr.dtype not in (torch.int32, torch.int64):
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")

    indptr64 = indptr.to(torch.int64).contiguous()
    indices = indices.contiguous()
    nnz = int(indices.numel())
    if indptr64.numel() > 0 and int(indptr64[0].item()) != 0:
        raise ValueError("indptr[0] must be 0")
    if indptr64.numel() > 0 and int(indptr64[-1].item()) != nnz:
        raise ValueError(f"indptr[-1] must equal nnz={nnz}")
    if indptr64.numel() > 1 and bool(torch.any(indptr64[1:] < indptr64[:-1]).item()):
        raise ValueError("indptr must be nondecreasing")
    if nnz > 0:
        min_col = int(indices.min().item())
        max_col = int(indices.max().item())
        if min_col < 0 or max_col >= n_cols:
            raise IndexError("indices out of range for shape[1]")
    return indices, indptr64, (n_rows, n_cols)


def _build_row_ids(indptr):
    n_rows = int(indptr.numel()) - 1
    if n_rows <= 0:
        return torch.empty(0, dtype=torch.int32, device=indptr.device)
    row_counts = indptr[1:] - indptr[:-1]
    return torch.repeat_interleave(
        torch.arange(n_rows, dtype=torch.int32, device=indptr.device),
        row_counts,
    )


def prepare_sddmm_csr(indices, indptr, shape, k_hint=64):
    indices, indptr, shape = _prepare_sddmm_csr_pattern(indices, indptr, shape)
    row_ids = _build_row_ids(indptr)
    block_k, num_warps = _resolve_sddmm_launch_config(int(k_hint))
    return SDDMMPrepared(
        indices=indices,
        indptr=indptr,
        shape=shape,
        row_ids=row_ids,
        block_k=block_k,
        num_warps=num_warps,
    )


@triton.jit
def _sddmm_csr_real_kernel(
    indices_ptr,
    row_ids_ptr,
    x_ptr,
    y_ptr,
    in_ptr,
    out_ptr,
    nnz,
    k_dim,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    alpha,
    beta,
    HAS_IN: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < nnz

    rows = tl.load(row_ids_ptr + offs_p, mask=mask_p, other=0)
    cols = tl.load(indices_ptr + offs_p, mask=mask_p, other=0)
    acc = tl.zeros([BLOCK_P], dtype=ACC_DTYPE)

    for k0 in tl.range(0, k_dim, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_dim
        x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk
        y_ptrs = y_ptr + cols[:, None] * stride_ym + offs_k[None, :] * stride_yk
        xy_mask = mask_p[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptrs, mask=xy_mask, other=0.0)
        y_vals = tl.load(y_ptrs, mask=xy_mask, other=0.0)
        acc += tl.sum(x_vals.to(ACC_DTYPE) * y_vals.to(ACC_DTYPE), axis=1)

    out_vals = acc * alpha
    if HAS_IN:
        in_vals = tl.load(in_ptr + offs_p, mask=mask_p, other=0.0).to(ACC_DTYPE)
        out_vals += in_vals * beta
    tl.store(out_ptr + offs_p, out_vals, mask=mask_p)


@triton.jit
def _sddmm_csr_real_kernel_altreduce(
    indices_ptr,
    row_ids_ptr,
    x_ptr,
    y_ptr,
    in_ptr,
    out_ptr,
    nnz,
    k_dim,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    alpha,
    beta,
    HAS_IN: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < nnz

    rows = tl.load(row_ids_ptr + offs_p, mask=mask_p, other=0)
    cols = tl.load(indices_ptr + offs_p, mask=mask_p, other=0)
    acc = tl.zeros([BLOCK_P], dtype=ACC_DTYPE)

    for k0 in tl.range(0, k_dim, BLOCK_K):
        for kk in tl.static_range(0, BLOCK_K):
            k_idx = k0 + kk
            valid_k = k_idx < k_dim
            x_vals = tl.load(
                x_ptr + rows * stride_xm + k_idx * stride_xk,
                mask=mask_p & valid_k,
                other=0.0,
            )
            y_vals = tl.load(
                y_ptr + cols * stride_ym + k_idx * stride_yk,
                mask=mask_p & valid_k,
                other=0.0,
            )
            acc += x_vals.to(ACC_DTYPE) * y_vals.to(ACC_DTYPE)

    out_vals = acc * alpha
    if HAS_IN:
        in_vals = tl.load(in_ptr + offs_p, mask=mask_p, other=0.0).to(ACC_DTYPE)
        out_vals += in_vals * beta
    tl.store(out_ptr + offs_p, out_vals, mask=mask_p)


def _validate_sddmm_dense_inputs(data, prepared, x, y):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D dense tensors")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.device != y.device or x.device != prepared.indices.device:
        raise ValueError("x, y, and sparse pattern must be on the same CUDA device")
    if x.dtype not in SUPPORTED_SDDMM_VALUE_DTYPES:
        raise TypeError("x dtype must be torch.float32 or torch.float64")
    if y.dtype != x.dtype:
        raise TypeError("y dtype must match x dtype")
    if data is not None and data.dtype != x.dtype:
        raise TypeError("data dtype must match x/y dtype")
    if x.shape[0] != prepared.n_rows:
        raise ValueError(f"x.shape[0] must be n_rows={prepared.n_rows}, got {x.shape[0]}")
    if y.shape[0] != prepared.n_cols:
        raise ValueError(f"y.shape[0] must be n_cols={prepared.n_cols}, got {y.shape[0]}")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same K dimension")
    if data is not None and data.numel() != prepared.nnz:
        raise ValueError("data length must equal nnz of sparse pattern")
    return int(x.shape[1])


def _prepare_validated_sddmm_out(prepared, x, out, out_dtype=None):
    nnz = prepared.nnz
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if out is None:
        return torch.empty(nnz, dtype=target_dtype, device=x.device)
    if out.ndim != 1 or out.numel() != nnz:
        raise ValueError("out must be a 1D tensor with length nnz")
    if not out.is_cuda or out.device != x.device:
        raise ValueError("out must be a CUDA tensor on the same device as x")
    if out.dtype != target_dtype:
        raise TypeError("out dtype must match the requested output dtype")
    return out


def _normalize_sddmm_diagnostic_variant(variant):
    if variant is None:
        return "baseline"
    variant = str(variant).strip().lower()
    if variant not in SUPPORTED_SDDMM_DIAGNOSTIC_VARIANTS:
        supported = ", ".join(SUPPORTED_SDDMM_DIAGNOSTIC_VARIANTS)
        raise ValueError(f"Unsupported SDDMM diagnostic variant {variant!r}; expected one of: {supported}")
    return variant


def _resolve_sddmm_diagnostic_kernel(variant, value_dtype):
    variant = _normalize_sddmm_diagnostic_variant(variant)
    if variant == "baseline":
        acc_dtype = tl.float64 if value_dtype == torch.float64 else tl.float32
        return _sddmm_csr_real_kernel, acc_dtype
    if variant in ("acc64", "acc64_out64"):
        return _sddmm_csr_real_kernel, tl.float64
    acc_dtype = tl.float64 if value_dtype == torch.float64 else tl.float32
    return _sddmm_csr_real_kernel_altreduce, acc_dtype


def _resolve_sddmm_diagnostic_out_dtype(variant, value_dtype):
    variant = _normalize_sddmm_diagnostic_variant(variant)
    if variant == "acc64_out64":
        return torch.float64
    return value_dtype


def _run_sddmm_prepared(prepared, x, y, data, alpha, beta, out, allow_fallback=False, variant="baseline", out_dtype=None):
    nnz = prepared.nnz
    variant = _normalize_sddmm_diagnostic_variant(variant)
    target_out_dtype = _resolve_sddmm_diagnostic_out_dtype(variant, x.dtype) if out_dtype is None else out_dtype
    out = _prepare_validated_sddmm_out(prepared, x, out, out_dtype=target_out_dtype)
    if nnz == 0:
        return out, {
            "block_k": prepared.block_k,
            "num_warps": prepared.num_warps,
            "fallback_used": False,
            "variant": variant,
            "acc_dtype": "float64" if target_out_dtype == torch.float64 else "float32",
            "out_dtype": str(target_out_dtype).replace("torch.", ""),
        }

    k_dim = int(x.shape[1])
    block_k, num_warps = _resolve_sddmm_launch_config(k_dim)
    block_p = 128
    kernel, acc_dtype = _resolve_sddmm_diagnostic_kernel(variant, x.dtype)
    grid = (triton.cdiv(nnz, block_p),)
    fallback_used = False
    if allow_fallback:
        try:
            kernel[grid](
                prepared.indices,
                prepared.row_ids,
                x,
                y,
                data if data is not None else out,
                out,
                nnz,
                k_dim,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                float(alpha),
                float(beta),
                HAS_IN=data is not None,
                BLOCK_P=block_p,
                BLOCK_K=block_k,
                ACC_DTYPE=acc_dtype,
                num_warps=num_warps,
            )
        except Exception:
            out.copy_(_sddmm_reference(prepared.indices, prepared.indptr, x, y, data, alpha, beta).to(out.dtype))
            fallback_used = True
    else:
        kernel[grid](
            prepared.indices,
            prepared.row_ids,
            x,
            y,
            data if data is not None else out,
            out,
            nnz,
            k_dim,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            float(alpha),
            float(beta),
            HAS_IN=data is not None,
            BLOCK_P=block_p,
            BLOCK_K=block_k,
            ACC_DTYPE=acc_dtype,
            num_warps=num_warps,
        )
    return out, {
        "block_k": block_k,
        "num_warps": num_warps,
        "fallback_used": fallback_used,
        "variant": variant,
        "acc_dtype": "float64" if acc_dtype == tl.float64 else "float32",
        "out_dtype": str(out.dtype).replace("torch.", ""),
    }


def flagsparse_sddmm_csr(
    data=None,
    indices=None,
    indptr=None,
    x=None,
    y=None,
    shape=None,
    alpha=1.0,
    beta=0.0,
    prepared=None,
    out=None,
    return_time=False,
    return_meta=False,
    allow_fallback=False,
):
    """CSR SDDMM: out[p] = alpha * dot(x[row(p)], y[col(p)]) + beta * data[p]."""
    prepare_ms = 0.0
    if prepared is None:
        if any(v is None for v in (indices, indptr, shape)):
            raise ValueError("indices, indptr, and shape are required when prepared is not provided")
        torch.cuda.synchronize()
        t_prepare0 = time.perf_counter()
        k_hint = int(x.shape[1]) if (x is not None and x.ndim == 2) else 64
        prepared = prepare_sddmm_csr(indices, indptr, shape, k_hint=k_hint)
        torch.cuda.synchronize()
        prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0
    elif not isinstance(prepared, SDDMMPrepared):
        raise TypeError("prepared must be a SDDMMPrepared instance")

    if x is None or y is None:
        raise ValueError("x and y are required")
    if data is None and float(beta) != 0.0:
        raise ValueError("data is required when beta is non-zero")
    k_dim = _validate_sddmm_dense_inputs(data, prepared, x, y)
    if k_dim == 0:
        out = _prepare_validated_sddmm_out(prepared, x, out)
        if beta == 0.0 or data is None:
            out.zero_()
        else:
            out.copy_(data * beta)
        meta = {"prepare_ms": prepare_ms, "block_k": prepared.block_k, "num_warps": prepared.num_warps, "fallback_used": False}
        if return_time and return_meta:
            return out, 0.0, meta
        if return_time:
            return out, 0.0
        if return_meta:
            return out, meta
        return out

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_tensor, launch_meta = _run_sddmm_prepared(
        prepared,
        x.contiguous(),
        y.contiguous(),
        data.contiguous() if data is not None else None,
        alpha,
        beta,
        out,
        allow_fallback=allow_fallback,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if return_time and return_meta:
        meta = {"prepare_ms": prepare_ms, **launch_meta}
        return out_tensor, elapsed_ms, meta
    if return_time:
        return out_tensor, elapsed_ms
    if return_meta:
        meta = {"prepare_ms": prepare_ms, **launch_meta}
        return out_tensor, meta
    return out_tensor


def _sddmm_reference(indices, indptr, x, y, data, alpha, beta):
    n_rows = int(indptr.numel()) - 1
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, dtype=torch.int64, device=indices.device),
        indptr[1:] - indptr[:-1],
    )
    if row_ids.numel() == 0:
        return torch.empty(0, dtype=x.dtype, device=x.device)
    vals = torch.sum(x[row_ids] * y[indices.to(torch.int64)], dim=1)
    vals = alpha * vals
    if data is not None:
        vals = vals + beta * data
    return vals


def _cupy_sampled_dot_reference(indices, indptr, x, y, data, alpha, beta, chunk_nnz=262144):
    _require_cupy()
    n_rows = int(indptr.numel()) - 1
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, dtype=torch.int64, device=indices.device),
        indptr[1:] - indptr[:-1],
    )
    row_ids_cp = _cupy_from_torch(row_ids)
    col_ids_cp = _cupy_from_torch(indices.to(torch.int64))
    x_cp = _cupy_from_torch(x)
    y_cp = _cupy_from_torch(y)
    nnz = int(indices.numel())
    if nnz == 0:
        vals = torch.empty(0, dtype=x.dtype, device=x.device)
        if data is not None and beta != 0.0:
            vals = vals + data * beta
        return vals

    out_cp = cp.empty((nnz,), dtype=x_cp.dtype)
    chunk_nnz = max(1, int(chunk_nnz))
    for start in range(0, nnz, chunk_nnz):
        end = min(nnz, start + chunk_nnz)
        rows = row_ids_cp[start:end]
        cols = col_ids_cp[start:end]
        out_cp[start:end] = cp.sum(x_cp[rows] * y_cp[cols], axis=1)
    out = _torch_from_cupy(out_cp)
    out = out * alpha
    if data is not None and beta != 0.0:
        out = out + data * beta
    return out


def benchmark_sddmm_case(
    n_rows=1024,
    n_cols=1024,
    nnz=16384,
    k_dim=64,
    value_dtype=torch.float32,
    warmup=10,
    iters=30,
    alpha=1.0,
    beta=0.0,
    run_cusparse=False,
):
    """Benchmark SDDMM and compare with sampled-dot reference."""
    if value_dtype not in SUPPORTED_SDDMM_VALUE_DTYPES:
        raise TypeError("value_dtype must be torch.float32 or torch.float64")
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, torch.int32, device
    )
    x = _build_random_dense((n_rows, k_dim), value_dtype, device)
    y = _build_random_dense((n_cols, k_dim), value_dtype, device)

    prepared = prepare_sddmm_csr(indices, indptr, (n_rows, n_cols), k_hint=k_dim)
    op = lambda: flagsparse_sddmm_csr(
        data=data,
        x=x,
        y=y,
        alpha=alpha,
        beta=beta,
        prepared=prepared,
        return_time=False,
    )
    triton_values, triton_ms = _benchmark_cuda_op(op, warmup=warmup, iters=iters)
    ref_op = lambda: _sddmm_reference(indices, indptr.to(torch.int64), x, y, data, alpha, beta)
    ref_values, pytorch_ms = _benchmark_cuda_op(ref_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(value_dtype)
    match = bool(torch.allclose(triton_values, ref_values, atol=atol, rtol=rtol))
    max_abs = (
        float(torch.max(torch.abs(triton_values - ref_values)).item())
        if triton_values.numel() > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_reason = None
    cusparse_match = None
    if run_cusparse:
        if cp is None:
            cusparse_reason = "CuPy is not available"
        else:
            try:
                ref_cu, cusparse_ms = _benchmark_cuda_op(
                    lambda: _cupy_sampled_dot_reference(
                        indices=indices,
                        indptr=indptr.to(torch.int64),
                        x=x,
                        y=y,
                        data=data,
                        alpha=alpha,
                        beta=beta,
                    ),
                    warmup=warmup,
                    iters=iters,
                )
                cusparse_match = bool(torch.allclose(triton_values, ref_cu, atol=atol, rtol=rtol))
            except Exception as exc:
                cusparse_reason = str(exc)

    return {
        "parameters": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "k_dim": k_dim,
            "value_dtype": str(value_dtype),
            "warmup": warmup,
            "iters": iters,
            "alpha": alpha,
            "beta": beta,
        },
        "performance": {
            "triton_ms": triton_ms,
            "pytorch_ms": pytorch_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": (pytorch_ms / triton_ms if triton_ms > 0 else None),
            "triton_speedup_vs_cusparse": (cusparse_ms / triton_ms if (cusparse_ms and triton_ms > 0) else None),
        },
        "verification": {
            "triton_match_pytorch": match,
            "triton_max_abs_error": max_abs,
            "cusparse_match_pytorch": cusparse_match,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {
            "triton": triton_values,
            "pytorch": ref_values,
        },
    }
