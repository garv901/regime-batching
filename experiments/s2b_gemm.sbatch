#!/bin/bash
#SBATCH -J s2b_gemm
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH -o logs/%x_%j.out
# S2b: per-M GEMM regime sweep (cuBLASLt vs Triton autotune) for the linear layers of the study models.
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
nvidia-smi --query-gpu=index,name,memory.used --format=csv
python microbench/gemm_sweep.py --out results/microbench/gemm_sweep.parquet
echo "done $(date)"
