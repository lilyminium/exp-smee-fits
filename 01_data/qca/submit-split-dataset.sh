#!/usr/bin/env bash
#SBATCH -J split
#SBATCH -p preemptable
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yml

# convert back to dataset
python filter-by-smiles.py \
    -i output/filtered/minimum-all \
    -s smiles/combined-diverse-split.json \
    -o output/filtered/minimum-diverse > logs/filter-by-smiles-minimum.txt 2>&1

python filter-by-smiles.py \
    -i output/filtered/trajectory-all \
    -s smiles/combined-diverse-split.json \
    -o output/filtered/trajectory-diverse > logs/filter-by-smiles-trajectory.txt 2>&1
