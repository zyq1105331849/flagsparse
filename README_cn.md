# FlagSparse

GPU 稀疏运算库（SpMV、gather、scatter、多种稀疏格式）。

## 安装

```bash
pip install . --no-deps --no-build-isolation
```

离线时可加 `--no-build-isolation` 避免拉取构建依赖。

运行时依赖（按需安装）：

```bash
pip install torch triton cupy-cuda12x
```

## 目录说明

- `src/flagsparse/` — 核心包（`sparse_operations/` 由 `flagsparse.py` 内嵌字符串生成多个 `.py`）
- `tests/` — pytest 测试
- `benchmark/` — 性能基准

## 测试用法

在项目根目录执行，或先 `cd tests` 再执行（.mtx 目录可用 `../matrix` 等相对路径）。

**test_spmv.py** — CSR SpMV（SuiteSparse .mtx、合成数据或导出 CSV）：

```bash
python tests/test_spmv.py <目录或文件.mtx>               # 批量跑，默认 float32
python tests/test_spmv.py <目录/> --dtype float64        # 可选：--index-dtype int32、--warmup 10、--iters 50、--no-cusparse
python tests/test_spmv.py --synthetic                    # 合成数据基准
python tests/test_spmv.py <目录/> --csv-csr results.csv  # 全 dtype，导出 CSV
```

**test_spmv_coo.py** — COO SpMV：

```bash
python tests/test_spmv_coo.py --synthetic                # 合成数据
python tests/test_spmv_coo.py <目录/> --csv-coo out.csv  # .mtx 批量，导出 CSV
```

**test_spsv.py** — SpSV（三角求解，仅方阵）：

```bash
python tests/test_spsv.py --synthetic                     # 合成数据
python tests/test_spsv.py <目录/> --csv-csr spsv.csv     # .mtx 批量，导出 CSV
```

**test_gather.py** / **test_scatter.py** — gather/scatter 基准（pytest 或 `python tests/test_gather.py`）。
