#!/bin/bash
#SBATCH --job-name=ex1a
#SBATCH --output=logs/%j_ex1a.out
#SBATCH --error=logs/%j_ex1a.err
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=priority

mkdir -p logs

export TF_GPU_ALLOCATOR=cuda_malloc_async

# example of how to follow the logs
# tail logs/687_ex1a.out --follow

python python_files/ex1a.py "$@"
