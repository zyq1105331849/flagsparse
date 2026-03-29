"""Sparse matrix formats aligned with CuPy/cupyx."""

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError as exc:
    raise ImportError(
        "CuPy is required for sparse format utilities. "
        "Install a CUDA-matched wheel, for example: pip install cupy-cuda12x"
    ) from exc

try:
    import torch
except ImportError:
    torch = None


def _resolve_dtype(dtype):
    if dtype is None:
        return cp.dtype(cp.float32)
    if torch is not None and isinstance(dtype, torch.dtype):
        try:
            cupy_bfloat16 = cp.dtype("bfloat16")
        except TypeError:
            cupy_bfloat16 = cp.float32
        mapping = {
            torch.float16: cp.float16,
            torch.bfloat16: cupy_bfloat16,
            torch.float32: cp.float32,
            torch.float64: cp.float64,
            torch.complex64: cp.complex64,
            torch.complex128: cp.complex128,
            torch.int32: cp.int32,
            torch.int64: cp.int64,
        }
        if dtype not in mapping:
            raise TypeError(f"Unsupported torch dtype: {dtype}")
        return cp.dtype(mapping[dtype])
    return cp.dtype(dtype)


def _to_cupy_array(x, dtype=None):
    target_dtype = _resolve_dtype(dtype) if dtype is not None else None
    if isinstance(x, cp.ndarray):
        return x.astype(target_dtype, copy=False) if target_dtype else x
    if torch is not None and torch.is_tensor(x):
        arr = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        return arr.astype(target_dtype, copy=False) if target_dtype else arr
    arr = cp.asarray(x)
    return arr.astype(target_dtype, copy=False) if target_dtype else arr


def _random_values(nnz, dtype):
    dtype = cp.dtype(dtype)
    if dtype == cp.dtype(cp.complex64):
        real = cp.random.standard_normal(nnz, dtype=cp.float32)
        imag = cp.random.standard_normal(nnz, dtype=cp.float32)
        return (real + 1j * imag).astype(dtype, copy=False)
    if dtype == cp.dtype(cp.complex128):
        real = cp.random.standard_normal(nnz, dtype=cp.float64)
        imag = cp.random.standard_normal(nnz, dtype=cp.float64)
        return (real + 1j * imag).astype(dtype, copy=False)
    return cp.random.standard_normal(nnz).astype(dtype, copy=False)


class CSRMatrix:
    def __init__(self, values, indices=None, indptr=None, shape=None, dtype=None):
        if isinstance(values, cpx_sparse.csr_matrix):
            self.matrix = values
        else:
            resolved_dtype = _resolve_dtype(dtype)
            data = _to_cupy_array(values, dtype=resolved_dtype)
            cols = _to_cupy_array(indices, dtype=cp.int64)
            row_ptr = _to_cupy_array(indptr, dtype=cp.int64)
            self.matrix = cpx_sparse.csr_matrix(
                (data, cols, row_ptr), shape=shape, dtype=resolved_dtype
            )

    @property
    def values(self):
        return self.matrix.data

    @property
    def indices(self):
        return self.matrix.indices

    @property
    def indptr(self):
        return self.matrix.indptr

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def dtype(self):
        return self.matrix.dtype

    def to_dense(self):
        return self.matrix.toarray()

    def to_coo(self):
        return COOMatrix(self.matrix.tocoo())

    def __repr__(self):
        return f"CSRMatrix(shape={self.shape}, nnz={self.matrix.nnz})"


class CSCMatrix:
    """Compressed Sparse Column matrix (CuPy-style)."""
    def __init__(self, values, indices=None, indptr=None, shape=None, dtype=None):
        if isinstance(values, cpx_sparse.csc_matrix):
            self.matrix = values
        else:
            resolved_dtype = _resolve_dtype(dtype)
            data = _to_cupy_array(values, dtype=resolved_dtype)
            rows = _to_cupy_array(indices, dtype=cp.int64)
            col_ptr = _to_cupy_array(indptr, dtype=cp.int64)
            self.matrix = cpx_sparse.csc_matrix(
                (data, rows, col_ptr), shape=shape, dtype=resolved_dtype
            )

    @property
    def values(self):
        return self.matrix.data

    @property
    def indices(self):
        return self.matrix.indices

    @property
    def indptr(self):
        return self.matrix.indptr

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def dtype(self):
        return self.matrix.dtype

    def to_dense(self):
        return self.matrix.toarray()

    def to_coo(self):
        return COOMatrix(self.matrix.tocoo())

    def __repr__(self):
        return f"CSCMatrix(shape={self.shape}, nnz={self.matrix.nnz})"


class BSRMatrix:
    """Block Sparse Row matrix (CuPy/SciPy-style). blocksize=(R,C)."""
    def __init__(self, data, indices=None, indptr=None, shape=None, blocksize=None, dtype=None):
        if isinstance(data, cpx_sparse.bsr_matrix):
            self.matrix = data
        else:
            resolved_dtype = _resolve_dtype(dtype)
            data_arr = _to_cupy_array(data, dtype=resolved_dtype)
            inds = _to_cupy_array(indices, dtype=cp.int64)
            ptr = _to_cupy_array(indptr, dtype=cp.int64)
            if blocksize is None:
                raise ValueError("BSRMatrix requires blocksize (R, C)")
            if isinstance(blocksize, int):
                blocksize = (blocksize, blocksize)
            self.matrix = cpx_sparse.bsr_matrix(
                (data_arr, inds, ptr), shape=shape, blocksize=blocksize
            )

    @property
    def data(self):
        return self.matrix.data

    @property
    def indices(self):
        return self.matrix.indices

    @property
    def indptr(self):
        return self.matrix.indptr

    @property
    def blocksize(self):
        return self.matrix.blocksize

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def dtype(self):
        return self.matrix.dtype

    def to_dense(self):
        return self.matrix.toarray()

    def to_coo(self):
        return COOMatrix(self.matrix.tocoo())

    def __repr__(self):
        return f"BSRMatrix(shape={self.shape}, blocksize={self.blocksize}, nnz_blocks={self.indices.size})"


class SELLMatrix:
    """
    Sliced ELLPACK format. Stores: values, indices (column), slice_ptr, rows_per_slice.
    CuPy-compatible interface; backend uses CuPy arrays.
    """
    def __init__(self, values, indices, slice_ptr, rows_per_slice, shape, dtype=None):
        self._values = _to_cupy_array(values, dtype=_resolve_dtype(dtype))
        self._indices = _to_cupy_array(indices, dtype=cp.int64)
        self._slice_ptr = _to_cupy_array(slice_ptr, dtype=cp.int64)
        self._rows_per_slice = _to_cupy_array(rows_per_slice, dtype=cp.int64)
        self._shape = tuple(shape)

    @property
    def values(self):
        return self._values

    @property
    def indices(self):
        return self._indices

    @property
    def slice_ptr(self):
        return self._slice_ptr

    @property
    def rows_per_slice(self):
        return self._rows_per_slice

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._values.dtype

    def to_dense(self):
        return self.to_coo().to_dense()

    def to_coo(self):
        return _sell_to_coo(self)

    def __repr__(self):
        nnz = int(self._values.size)
        return f"SELLMatrix(shape={self.shape}, nnz={nnz}, n_slices={self._slice_ptr.size - 1})"


class BLOCKEDELLMatrix:
    """
    Blocked ELL format. data shape (n_block_rows, max_blocks_per_row, r, c),
    indices shape (n_block_rows, max_blocks_per_row). CuPy arrays.
    """
    def __init__(self, data, indices, block_shape, shape, dtype=None):
        self._data = _to_cupy_array(data, dtype=_resolve_dtype(dtype))
        self._indices = _to_cupy_array(indices, dtype=cp.int64)
        self._block_shape = tuple(block_shape)
        self._shape = tuple(shape)

    @property
    def data(self):
        return self._data

    @property
    def indices(self):
        return self._indices

    @property
    def block_shape(self):
        return self._block_shape

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._data.dtype

    def to_dense(self):
        return self.to_coo().to_dense()

    def to_coo(self):
        return _blocked_ell_to_coo(self)

    def __repr__(self):
        return f"BLOCKEDELLMatrix(shape={self.shape}, block_shape={self._block_shape})"


class COOMatrix:
    def __init__(self, row_indices, col_indices=None, values=None, shape=None, dtype=None):
        if isinstance(row_indices, cpx_sparse.coo_matrix):
            self.matrix = row_indices
        else:
            resolved_dtype = _resolve_dtype(dtype)
            rows = _to_cupy_array(row_indices, dtype=cp.int64)
            cols = _to_cupy_array(col_indices, dtype=cp.int64)
            data = _to_cupy_array(values, dtype=resolved_dtype)
            self.matrix = cpx_sparse.coo_matrix(
                (data, (rows, cols)), shape=shape, dtype=resolved_dtype
            )

    @property
    def row_indices(self):
        return self.matrix.row

    @property
    def col_indices(self):
        return self.matrix.col

    @property
    def values(self):
        return self.matrix.data

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def dtype(self):
        return self.matrix.dtype

    def to_dense(self):
        return self.matrix.toarray()

    def to_csr(self):
        return CSRMatrix(self.matrix.tocsr())

    def to_csc(self):
        return CSCMatrix(self.matrix.tocsc())

    def to_bsr(self, blocksize=None):
        if blocksize is None:
            blocksize = (1, 1)
        return BSRMatrix(self.matrix.tobsr(blocksize=blocksize))

    def to_sell(self, slice_size=32):
        return coo_to_sell(self, slice_size=slice_size)

    def to_blocked_ell(self, block_shape):
        return coo_to_blocked_ell(self, block_shape=block_shape)

    def __repr__(self):
        return f"COOMatrix(shape={self.shape}, nnz={self.matrix.nnz})"


def _sell_to_coo(sell_mat):
    n_rows, n_cols = sell_mat.shape
    slice_ptr = sell_mat.slice_ptr
    rows_per_slice = sell_mat.rows_per_slice
    values = sell_mat.values
    indices = sell_mat.indices
    n_slices = int(slice_ptr.size) - 1
    rows_list = []
    cols_list = []
    vals_list = []
    base_row = 0
    for s in range(n_slices):
        start = int(slice_ptr[s])
        end = int(slice_ptr[s + 1])
        rps = int(rows_per_slice[s])
        if rps <= 0:
            base_row += rps
            continue
        max_nnz = (end - start) // rps
        for r in range(rps):
            row = base_row + r
            for k in range(max_nnz):
                idx = start + r * max_nnz + k
                col = int(indices[idx])
                val = values[idx]
                nonzero = (val != 0).item() if hasattr(val, "item") else (val != 0)
                if row < n_rows and 0 <= col < n_cols and nonzero:
                    rows_list.append(row)
                    cols_list.append(col)
                    vals_list.append(val)
        base_row += rps
    if not rows_list:
        return COOMatrix(
            cp.array([], dtype=cp.int64),
            cp.array([], dtype=cp.int64),
            cp.array([], dtype=sell_mat.dtype),
            sell_mat.shape,
            dtype=sell_mat.dtype,
        )
    return COOMatrix(
        cp.asarray(rows_list, dtype=cp.int64),
        cp.asarray(cols_list, dtype=cp.int64),
        cp.asarray(vals_list, dtype=sell_mat.dtype),
        sell_mat.shape,
        dtype=sell_mat.dtype,
    )


def _blocked_ell_to_coo(be_mat):
    data = be_mat.data
    indices = be_mat.indices
    br, bc = be_mat.block_shape
    n_block_rows, max_blocks = indices.shape
    n_rows, n_cols = be_mat.shape
    rows_list = []
    cols_list = []
    vals_list = []
    for i in range(n_block_rows):
        for j in range(max_blocks):
            bcol = int(indices[i, j])
            if bcol < 0:
                continue
            block = data[i, j]
            for di in range(br):
                for dj in range(bc):
                    r = i * br + di
                    c = bcol * bc + dj
                    if r < n_rows and c < n_cols:
                        v = block[di, dj]
                        nonzero = (v != 0).item() if hasattr(v, "item") else (v != 0)
                        if nonzero:
                            rows_list.append(r)
                            cols_list.append(c)
                            vals_list.append(v)
    if not rows_list:
        return COOMatrix(
            cp.array([], dtype=cp.int64),
            cp.array([], dtype=cp.int64),
            cp.array([], dtype=be_mat.dtype),
            be_mat.shape,
            dtype=be_mat.dtype,
        )
    return COOMatrix(
        cp.asarray(rows_list, dtype=cp.int64),
        cp.asarray(cols_list, dtype=cp.int64),
        cp.asarray(vals_list, dtype=be_mat.dtype),
        be_mat.shape,
        dtype=be_mat.dtype,
    )


def _coo_to_sell_impl(rows, cols, data, shape, slice_size):
    n_rows, n_cols = shape
    if slice_size is None or slice_size <= 0:
        slice_size = 32
    slice_size = int(slice_size)
    sort_idx = cp.lexsort((cols, rows))
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    data = data[sort_idx]
    nnz_per_row = cp.bincount(rows, minlength=n_rows)
    n_slices = (n_rows + slice_size - 1) // slice_size
    slice_ptr = cp.zeros(n_slices + 1, dtype=cp.int64)
    rows_per_slice = cp.zeros(n_slices, dtype=cp.int64)
    total_entries = 0
    for s in range(n_slices):
        r0 = s * slice_size
        r1 = min(r0 + slice_size, n_rows)
        rps = r1 - r0
        rows_per_slice[s] = rps
        if rps > 0:
            max_nnz = int(cp.max(nnz_per_row[r0:r1]))
        else:
            max_nnz = 0
        total_entries += rps * max_nnz
        slice_ptr[s + 1] = total_entries
    values = cp.zeros(total_entries, dtype=data.dtype)
    indices = cp.zeros(total_entries, dtype=cp.int64)
    row_start = cp.zeros(n_rows + 1, dtype=cp.int64)
    row_start[1:] = cp.cumsum(nnz_per_row)
    base = 0
    for s in range(n_slices):
        r0 = s * slice_size
        r1 = min(r0 + slice_size, n_rows)
        rps = int(rows_per_slice[s])
        if rps == 0:
            continue
        max_nnz = (int(slice_ptr[s + 1]) - int(slice_ptr[s])) // rps
        for r in range(rps):
            row = r0 + r
            start = int(row_start[row])
            end = int(row_start[row + 1])
            nnz = end - start
            dst_start = base + r * max_nnz
            if nnz > 0:
                values[dst_start : dst_start + nnz] = data[start:end]
                indices[dst_start : dst_start + nnz] = cols[start:end]
        base = int(slice_ptr[s + 1])
    return SELLMatrix(values, indices, slice_ptr, rows_per_slice, shape, dtype=data.dtype)


def _coo_to_blocked_ell_impl(rows, cols, data, shape, block_shape):
    br, bc = block_shape
    n_rows, n_cols = shape
    if n_rows % br != 0 or n_cols % bc != 0:
        raise ValueError(
            f"shape {shape} must be divisible by block_shape {block_shape}"
        )
    n_block_rows = n_rows // br
    n_block_cols = n_cols // bc
    block_rows = rows // br
    block_cols = cols // bc
    in_block_r = rows % br
    in_block_c = cols % bc
    nnz = rows.size
    blocks = {}
    for k in range(nnz):
        i = int(block_rows[k])
        j = int(block_cols[k])
        ir = int(in_block_r[k])
        ic = int(in_block_c[k])
        key = (i, j)
        if key not in blocks:
            blocks[key] = cp.zeros((br, bc), dtype=data.dtype)
        blocks[key][ir, ic] += data[k]
    max_blocks_per_row = 0
    for i in range(n_block_rows):
        count = sum(1 for (bi, _) in blocks if bi == i)
        max_blocks_per_row = max(max_blocks_per_row, count)
    if max_blocks_per_row == 0:
        max_blocks_per_row = 1
    data_out = cp.zeros((n_block_rows, max_blocks_per_row, br, bc), dtype=data.dtype)
    indices_out = cp.full((n_block_rows, max_blocks_per_row), -1, dtype=cp.int64)
    for i in range(n_block_rows):
        cols_in_row = sorted([j for (bi, j) in blocks if bi == i])
        for t, j in enumerate(cols_in_row):
            data_out[i, t] = blocks[(i, j)]
            indices_out[i, t] = j
    return BLOCKEDELLMatrix(
        data_out, indices_out, block_shape, shape, dtype=data.dtype
    )


def create_csr_matrix(values, indices, indptr, shape, dtype=None):
    return CSRMatrix(values, indices, indptr, shape, dtype=dtype)


def create_coo_matrix(row_indices, col_indices, values, shape, dtype=None):
    return COOMatrix(row_indices, col_indices, values, shape, dtype=dtype)


def coo_to_csr(coo_matrix):
    if not isinstance(coo_matrix, COOMatrix):
        raise TypeError("coo_matrix must be an instance of COOMatrix")
    return coo_matrix.to_csr()


def coo_to_csc(coo_matrix):
    """Convert COO to CSC (CuPy-style: .tocsc())."""
    if not isinstance(coo_matrix, COOMatrix):
        raise TypeError("coo_matrix must be an instance of COOMatrix")
    return coo_matrix.to_csc()


def coo_to_bsr(coo_matrix, blocksize=None):
    """Convert COO to BSR. blocksize: (R, C) or int for square block."""
    if not isinstance(coo_matrix, COOMatrix):
        raise TypeError("coo_matrix must be an instance of COOMatrix")
    return coo_matrix.to_bsr(blocksize=blocksize)


def coo_to_sell(coo_matrix, slice_size=32):
    """Convert COO to SELL (Sliced ELLPACK). slice_size: rows per slice."""
    if not isinstance(coo_matrix, COOMatrix):
        raise TypeError("coo_matrix must be an instance of COOMatrix")
    rows = coo_matrix.row_indices
    cols = coo_matrix.col_indices
    data = coo_matrix.values
    return _coo_to_sell_impl(rows, cols, data, coo_matrix.shape, slice_size)


def coo_to_blocked_ell(coo_matrix, block_shape):
    """Convert COO to BLOCKED-ELL. block_shape: (r, c) block dimensions."""
    if not isinstance(coo_matrix, COOMatrix):
        raise TypeError("coo_matrix must be an instance of COOMatrix")
    rows = coo_matrix.row_indices
    cols = coo_matrix.col_indices
    data = coo_matrix.values
    return _coo_to_blocked_ell_impl(
        rows, cols, data, coo_matrix.shape, block_shape
    )


def create_csc_matrix(values, indices, indptr, shape, dtype=None):
    """Create CSC from (data, row_indices, col_ptr), CuPy-style."""
    return CSCMatrix(values, indices, indptr, shape, dtype=dtype)


def create_bsr_matrix(data, indices, indptr, shape, blocksize, dtype=None):
    """Create BSR from (data, indices, indptr), shape, blocksize=(R,C)."""
    return BSRMatrix(data, indices, indptr, shape, blocksize=blocksize, dtype=dtype)


def create_sell_matrix(values, indices, slice_ptr, rows_per_slice, shape, dtype=None):
    """Create SELL from values, indices, slice_ptr, rows_per_slice, shape."""
    return SELLMatrix(
        values, indices, slice_ptr, rows_per_slice, shape, dtype=dtype
    )


def create_blocked_ell_matrix(data, indices, block_shape, shape, dtype=None):
    """Create BLOCKED-ELL from data, indices, block_shape, shape."""
    return BLOCKEDELLMatrix(data, indices, block_shape, shape, dtype=dtype)


def generate_random_sparse_matrix(
    n_rows, n_cols, density=0.1, dtype=cp.float32, device=None
):
    if device is not None:
        cp.cuda.Device(device).use()
    total = int(n_rows) * int(n_cols)
    nnz = max(0, int(total * float(density)))
    nnz = min(nnz, total)
    if nnz == 0:
        rows = cp.asarray([], dtype=cp.int64)
        cols = cp.asarray([], dtype=cp.int64)
        vals = cp.asarray([], dtype=_resolve_dtype(dtype))
    else:
        linear = cp.random.permutation(total)[:nnz]
        rows = linear // n_cols
        cols = linear % n_cols
        vals = _random_values(nnz, _resolve_dtype(dtype))
    coo_matrix = COOMatrix(rows, cols, vals, (n_rows, n_cols), dtype=dtype)
    csr_matrix = coo_to_csr(coo_matrix)
    return coo_matrix, csr_matrix


def read_mtx_file(file_path, dtype=cp.float32, device=None):
    if device is not None:
        cp.cuda.Device(device).use()

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = []
    header_info = None
    for line in lines:
        line = line.strip()
        if line.startswith("%"):
            continue
        if not header_info and line:
            header_parts = line.split()
            n_rows = int(header_parts[0])
            n_cols = int(header_parts[1])
            nnz = int(header_parts[2])
            header_info = (n_rows, n_cols, nnz)
            continue
        if line:
            data_lines.append(line)

    if header_info is None:
        raise ValueError("Could not parse matrix dimensions from .mtx file")

    n_rows, n_cols, nnz = header_info
    rows_host = []
    cols_host = []
    vals_host = []
    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) >= 3:
            rows_host.append(int(parts[0]) - 1)
            cols_host.append(int(parts[1]) - 1)
            vals_host.append(float(parts[2]))

    resolved_dtype = _resolve_dtype(dtype)
    rows = cp.asarray(rows_host, dtype=cp.int64)
    cols = cp.asarray(cols_host, dtype=cp.int64)
    vals = cp.asarray(vals_host, dtype=resolved_dtype)
    coo_matrix = COOMatrix(rows, cols, vals, (n_rows, n_cols), dtype=resolved_dtype)
    csr_matrix = coo_to_csr(coo_matrix)
    return coo_matrix, csr_matrix
