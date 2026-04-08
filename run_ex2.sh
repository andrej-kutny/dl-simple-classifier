#!/bin/bash
#SBATCH --job-name=ex2
#SBATCH --output=logs/%j_ex2.out
#SBATCH --error=logs/%j_ex2.err
#SBATCH --time=0-02:59:59
#SBATCH --gres=gpu:1
#SBATCH --partition=priority

mkdir -p logs

export TF_GPU_ALLOCATOR=cuda_malloc_async

# usage: sbatch run_ex2.sh path/to/model_final.keras
# example of how to follow the logs
# tail logs/687_ex2.out --follow

python python_files/ex2.py "$@"
