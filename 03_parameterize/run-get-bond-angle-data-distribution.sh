#!/usr/bin/env bash
#SBATCH -J get-distribution
#SBATCH -p preemptable
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96gb
#SBATCH --output slurm-%x.%A.out

source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yaml

FORCEFIELD="openff-2.3.0"
DATA_DIR="../01_data/qca/output/filtered-f5-e10"
PROCESSES="${SLURM_CPUS_PER_TASK}"



mkdir -p logs

python      get-bond-angle-data-distribution.py       \
    -ff     "${FORCEFIELD}.offxml"      \
    -i      "${DATA_DIR}/trajectory-diverse"    \
    -np      "${PROCESSES}" \
    -o      "distributions/${FORCEFIELD}_trajectory-diverse" \
    > logs/get-distribution_${FORCEFIELD}_trajectory-diverse.log 2>&1
