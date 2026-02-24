#!/usr/bin/env bash
#SBATCH -J get-smiles
#SBATCH -p preemptable
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yml

python get-industry-benchmark-set.py > get-industry-benchmark-set.txt 2>&1
