#!/bin/bash
#SBATCH --job-name=ex1a
#SBATCH --output=logs/%j_ex1a.out
#SBATCH --error=logs/%j_ex1a.err
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=priority

mkdir -p logs

python python_files/ex1a.py "$@" > logs/ex1a_output.txt
