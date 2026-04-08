#!/bin/bash
#SBATCH --job-name=ex2_ex3
#SBATCH --output=logs/%j_ex2_ex3.out
#SBATCH --error=logs/%j_ex2_ex3.err
#SBATCH --time=0-03:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=priority

mkdir -p logs

export TF_GPU_ALLOCATOR=cuda_malloc_async

# usage: sbatch run_ex2_ex3.sh path/to/model_final.keras -e 25
# example of how to follow the logs
# tail logs/687_ex2_ex3.out --follow

python python_files/ex2.py "$@" && python python_files/ex3.py "$@"
