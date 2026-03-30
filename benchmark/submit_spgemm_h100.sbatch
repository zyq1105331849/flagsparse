#!/bin/bash
#SBATCH -J flagsparse_spgemm
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

module load apptainer

# Fixed runtime environment.
IMAGE_DIR="/data/home/scwb320/run/images"
IMAGE_NAME="flagtree.sif"
VENV_PATH="/data/home/scwb320/run/gcx/vpy"

# Update these three paths before submission.
PROJECT_DIR="/data/home/scwb320/alphasparse_triton/flagsparse/flagsparse_package"
MTX_DIR="/data/home/scwb320/datasets/mtx"
OUT_DIR="/data/home/scwb320/run/flagsparse_spgemm_out"

CSV_NAME="spgemm_${SLURM_JOB_ID}.csv"

mkdir -p "${OUT_DIR}"

echo "[INFO] job_id=${SLURM_JOB_ID}"
echo "[INFO] node_list=${SLURM_NODELIST}"
echo "[INFO] project_dir=${PROJECT_DIR}"
echo "[INFO] mtx_dir=${MTX_DIR}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] csv=${OUT_DIR}/${CSV_NAME}"

srun -n 1 apptainer exec \
  --nv \
  --cleanenv \
  --bind /data:/data \
  "${IMAGE_DIR}/${IMAGE_NAME}" \
  /bin/bash -lc "
set -euo pipefail

source \"${VENV_PATH}/bin/activate\"

unset TRITON_LIBCUDA_PATH
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
export PYTHONUNBUFFERED=1

cd \"${PROJECT_DIR}\"

python tests/test_spgemm.py \"${MTX_DIR}\" \
  --csv \"${OUT_DIR}/${CSV_NAME}\" \
  --warmup 10 \
  --iters 50 \
  --input-mode auto \
  --ref-blocked-retry \
  --ref-isolated-retry \
  --ref-cleanup
"
