#!/usr/bin/env bash
#SBATCH -J plot-datasets.py
#SBATCH -p preemptable
#SBATCH -t 00:10:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

mkdir output

python plot-datasets.py \
    -i output/compare-datasets.pkl \
    -o output/compare-datasets-all.png


