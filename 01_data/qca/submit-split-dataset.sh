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

SUFFIX="-f5-e10"

# convert back to dataset
python filter-by-smiles.py \
    -i output/filtered${SUFFIX}/minimum-all \
    -s smiles/combined-diverse-split.json \
    -o output/filtered${SUFFIX}/minimum-diverse > logs/filter-by-smiles-minimum${SUFFIX}.txt 2>&1

python filter-by-smiles.py \
    -i output/filtered${SUFFIX}/trajectory-all \
    -s smiles/combined-diverse-split.json \
    -o output/filtered${SUFFIX}/trajectory-diverse > logs/filter-by-smiles-trajectory${SUFFIX}.txt 2>&1
