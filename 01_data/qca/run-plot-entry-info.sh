#!/usr/bin/env bash
#SBATCH -J plot-dataset-info
#SBATCH -p preemptable
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yml

python plot-entry-info.py               \
    -i output/combined/trajectory-all   \
    -o metadata/combined/trajectory-all > logs/plot-entry-info-trajectory-all.txt 2>&1

python plot-entry-info.py               \
    -i output/combined/minimum-all          \
    -o metadata/combined/minimum-all > logs/plot-entry-info-minimum-all.txt 2>&1

