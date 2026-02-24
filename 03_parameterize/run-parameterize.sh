#!/usr/bin/env bash
#SBATCH -J parameterize
#SBATCH -p preemptable
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --output slurm-%x.%A.out

source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yaml

FORCEFIELD="openff-2.3.0"
DATA_DIR="../01_data/qca/output/combined"

SUBSET="all"

echo $SLURM_TASKS_PER_NODE


mkdir logs

python      parameterize.py       \
    -ff     "${FORCEFIELD}.offxml"      \
    -i      "${DATA_DIR}/minimum-${SUBSET}"    \
    -o      "output/${FORCEFIELD}_${SUBSET}.pkl" \
    -np     $SLURM_TASKS_PER_NODE > logs/parameterize_${FORCEFIELD}_${SUBSET}.log 2>&1



