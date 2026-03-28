"""Run scatter benchmark (FlagSparse vs PyTorch vs cuSPARSE)."""
import flagsparse as fs
if __name__ == "__main__":
    fs.comprehensive_scatter_test()
