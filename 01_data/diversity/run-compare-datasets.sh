#!/usr/bin/env bash
#SBATCH -J compare-datasets
#SBATCH -p preemptable
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

mkdir output

python compare-datasets.py \
    -od ../qca/output/filtered/minimum-diverse/train           \
    -sd ../spice/smiles.json \
    -of output/compare-datasets-diverse-train.pkl \
    -oi output/compare-datasets-diverse-train.png

python compare-datasets.py \
    -od ../qca/output/filtered/minimum-all           \
    -sd ../spice/smiles.json \
    -of output/compare-datasets-diverse-all.pkl \
    -oi output/compare-datasets-diverse-all.png
