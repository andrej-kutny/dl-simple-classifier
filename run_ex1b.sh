#!/bin/bash
#SBATCH --job-name=ex1b
#SBATCH --output=logs/%j_ex1b.out
#SBATCH --error=logs/%j_ex1b.err
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=priority

mkdir -p logs

export TF_GPU_ALLOCATOR=cuda_malloc_async

python python_files/ex1b.py "$@"
