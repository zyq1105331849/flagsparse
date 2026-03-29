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

**test_spmv.py** — CSR SpMV（SuiteSparse `.mtx`、合成数据或 CSR CSV）：

```bash
python tests/test_spmv.py <目录或文件.mtx>               # 批量跑，默认 float32
python tests/test_spmv.py <目录/> --dtype float64        # 可选：--index-dtype int32|int64、--warmup、--iters、--no-cusparse
python tests/test_spmv.py --synthetic                    # 合成基准
python tests/test_spmv.py <目录/> --csv-csr results.csv  # 全部 value×index dtype 写入一个 CSV（运行过程中逐矩阵打印）
```

**test_spmv_coo.py** — COO SpMV（需 `--synthetic` 或 `--csv-coo`，不能单独批量跑 .mtx）：

```bash
python tests/test_spmv_coo.py --synthetic
python tests/test_spmv_coo.py <目录/> --csv-coo out.csv
```

**test_spmv_opt.py** — SpMV 基线 vs 优化对比（仅 `float32` / `float64`）：

```bash
python tests/test_spmv_opt.py <目录或文件.mtx> [...]
python tests/test_spmv_opt.py <目录/> --csv out.csv
```

**test_spmm.py** — CSR SpMM（`.mtx` 批量、合成或 `--csv`）：

```bash
python tests/test_spmm.py <目录或文件.mtx>
python tests/test_spmm.py --synthetic                    # 可选：--skip-api-checks、--skip-alg1-coverage
python tests/test_spmm.py <目录/> --csv results.csv     # CSV 内为 float32/float64 + int32；控制台逐矩阵输出
# 常用：--dtype、--index-dtype、--dense-cols、--block-n、--block-nnz、--max-segments、--warmup、--iters、--no-cusparse
```

**test_spmm_coo.py** — 原生 COO SpMM：

```bash
python tests/test_spmm_coo.py <目录或文件.mtx>
python tests/test_spmm_coo.py --synthetic                # 可选：--route rowrun|atomic|compare、--skip-api-checks、--skip-coo-coverage
python tests/test_spmm_coo.py <目录/> --csv out.csv     # 仅支持 --route rowrun 或 atomic（compare 不能配 --csv）
# 与 CSR SpMM 类似的调参：--dense-cols、--block-n、--block-nnz、--warmup、--iters、--no-cusparse 等
```

**test_spsv.py** — SpSV（三角求解，**仅方阵**）。CSR 与 COO 共用本脚本；**不存在** `test_spsv_coo.py`。

```bash
python tests/test_spsv.py --synthetic
python tests/test_spsv.py <目录/> --csv-csr spsv.csv
python tests/test_spsv.py <目录/> --csv-coo out.csv     # 列与 CSR 相同；可选 --coo-mode auto|direct|csr（默认 auto）
```

**test_gather.py** / **test_scatter.py** — gather/scatter 基准（pytest 或 `python tests/test_gather.py`）。
