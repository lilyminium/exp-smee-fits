#!/usr/bin/env bash
#SBATCH -J label
#SBATCH -p preemptable
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out

source ~/.bashrc

conda activate smee-stack-cuda

mkdir -p data/optimization
mkdir logs

FFNAME="openff-2.3.0"

python label-with-forcefield.py \
    -i topology-values          \
    -f ${FFNAME}.offxml         \
    -o topology-labels         \
    > logs/label-with-forcefield-${FFNAME}.log 2>&1

