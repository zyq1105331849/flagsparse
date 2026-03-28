import math

import torch
import flagsparse as ast

VALUE_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
]
INDEX_DTYPES = [torch.int32, torch.int64]
TEST_CASES = [
    (32_768, 1_024),
    (131_072, 4_096),
    (524_288, 16_384),
    (1_048_576, 65_536),
]
WARMUP = 20
ITERS = 200


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _fmt_speedup(value):
    if value is None:
        return "N/A"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}x"


def _fmt_err(value):
    if value is None:
        return "N/A"
    return f"{value:.2e}"


def _print_row(
    dense_size,
    nnz,
    pytorch_ms,
    triton_ms,
    cusparse_ms,
    triton_vs_pytorch,
    triton_vs_cusparse,
    status,
    triton_err,
    cusparse_err,
):
    print(
        f"{dense_size:>10,d} {nnz:>10,d} "
        f"{_fmt_ms(pytorch_ms):>12} {_fmt_ms(triton_ms):>16} {_fmt_ms(cusparse_ms):>13} "
        f"{_fmt_speedup(triton_vs_pytorch):>12} {_fmt_speedup(triton_vs_cusparse):>12} "
        f"{status:>6} {_fmt_err(triton_err):>12} {_fmt_err(cusparse_err):>12}"
    )


def run_comprehensive_test():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return

    print("=" * 132)
    print("FLAGSPARSE SCATTER BENCHMARK")
    print("=" * 132)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {WARMUP} | Iterations: {ITERS}")
    print("Speedup columns are >1.0 when FlagSparse is faster.")
    print("cuSPARSE baseline uses COO SpMV equivalent scatter.")
    print("Scatter tests use unique indices by default for deterministic correctness checks.")
    print()

    total_cases = 0
    failed_cases = 0
    cusparse_unavailable_count = 0

    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("-" * 132)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<10} | "
                f"Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * 132)
            print(
                f"{'Dense':>10} {'NNZ':>10} "
                f"{'PyTorch(ms)':>12} {'FlagSparse(ms)':>16} {'cuSPARSE(ms)':>13} "
                f"{'FS/PT':>12} {'FS/CS':>12} {'OK':>6} "
                f"{'Err(FS)':>12} {'Err(CS)':>12}"
            )
            print("-" * 132)

            combo_cusparse_reason = None

            for dense_size, nnz in TEST_CASES:
                result = ast.benchmark_scatter_case(
                    dense_size=dense_size,
                    nnz=nnz,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=WARMUP,
                    iters=ITERS,
                    run_cusparse=True,
                    unique_indices=True,
                )
                perf = result["performance"]
                verify = result["verification"]
                backend = result["backend_status"]

                triton_ok = verify["triton_match_pytorch"]
                cusparse_ok = verify["cusparse_match_pytorch"]
                overall_ok = triton_ok and (cusparse_ok is None or cusparse_ok)
                status = "PASS" if overall_ok else "FAIL"

                if not overall_ok:
                    failed_cases += 1
                total_cases += 1

                if backend["cusparse_unavailable_reason"]:
                    combo_cusparse_reason = backend["cusparse_unavailable_reason"]
                    cusparse_unavailable_count += 1

                _print_row(
                    dense_size=dense_size,
                    nnz=nnz,
                    pytorch_ms=perf["pytorch_ms"],
                    triton_ms=perf["triton_ms"],
                    cusparse_ms=perf["cusparse_ms"],
                    triton_vs_pytorch=perf["triton_speedup_vs_pytorch"],
                    triton_vs_cusparse=perf["triton_speedup_vs_cusparse"],
                    status=status,
                    triton_err=verify["triton_max_error"],
                    cusparse_err=verify["cusparse_max_error"],
                )

            print("-" * 132)
            if combo_cusparse_reason:
                print("cuSPARSE backend unavailable for this combo:")
                print(f"  {combo_cusparse_reason}")
            print()

    print("=" * 132)
    print(f"Total cases: {total_cases}")
    print(f"Failed cases: {failed_cases}")
    print(f"Cases without cuSPARSE baseline: {cusparse_unavailable_count}")
    print("=" * 132)
    if failed_cases == 0:
        print("All correctness checks passed.")
    else:
        print("Some correctness checks failed. Please inspect rows marked FAIL.")


if __name__ == "__main__":
    run_comprehensive_test()
