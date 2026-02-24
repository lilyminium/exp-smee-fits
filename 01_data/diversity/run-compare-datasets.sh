#!/usr/bin/env bash
#SBATCH -J compare-datasets
#SBATCH -p preemptable
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

mkdir output

python compare-datasets.py \
    -i  "../qca/output/combined/minimum-all" "QCArchive" \
    -i  "../spice/smiles.json" "SPICE" \
    -i  "../industry-set/smiles.json" "Industry Benchmark v1.2" \
    --seed 0 \
    -o  output > compare-datasets.txt 2>&1



