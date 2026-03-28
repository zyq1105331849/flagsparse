# FlagSparse

GPU sparse operations package (SpMV, gather, scatter, sparse formats).

## Install

```bash
pip install . --no-deps --no-build-isolation
```

Use `--no-build-isolation` to avoid downloading build deps when offline.

Runtime dependencies (install when needed):

```bash
pip install torch triton cupy-cuda12x
```

## Layout

- `src/flagsparse/` — core package (`sparse_operations/` is emitted as several `.py` modules from string literals in `flagsparse.py`)
- `tests/` — pytest tests
- `benchmark/` — performance benchmarks

## Tests

Run from project root, or `cd tests` then run scripts (paths like `../matrix` for .mtx dir).

**test_spmv.py** — CSR SpMV (SuiteSparse .mtx, synthetic, or CSV export):

```bash
python tests/test_spmv.py <dir_or_file.mtx>              # batch run, default float32
python tests/test_spmv.py <dir/> --dtype float64         # optional: --index-dtype int32, --warmup 10, --iters 50, --no-cusparse
python tests/test_spmv.py --synthetic                    # synthetic benchmark
python tests/test_spmv.py <dir/> --csv-csr results.csv   # all dtypes, export CSV
```

**test_spmv_coo.py** — COO SpMV:

```bash
python tests/test_spmv_coo.py --synthetic                # synthetic
python tests/test_spmv_coo.py <dir/> --csv-coo out.csv   # .mtx batch, export CSV
```

**test_spsv.py** — SpSV (triangular solve, square matrices only):

```bash
python tests/test_spsv.py --synthetic                     # synthetic
python tests/test_spsv.py <dir/> --csv-csr spsv.csv      # .mtx batch, export CSV
```

**test_gather.py** / **test_scatter.py** — gather/scatter benchmarks (run with pytest or `python tests/test_gather.py`).
