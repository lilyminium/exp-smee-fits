#!/usr/bin/env bash
#SBATCH -J download-optimizations
#SBATCH -p cpu
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yml

mkdir logs

python download-optimizations.py --n-threads 40 > logs/download-optimizations.txt 2>&1
