#!/usr/bin/env bash
#SBATCH -J plot
#SBATCH -p preemptable
#SBATCH -t 00:10:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out

source ~/.bashrc

conda activate smee-stack-cuda

REFERENCE_FF="openff-2.3.0"

mkdir images
mkdir -p images/rmsd-tfd

# plot rmsd and tfd
python plot-rmsd-tfd.py  -i "rmsd-tfd"  -o "images/rmsd-tfd" > logs/plot-rmsd-tfd.log 2>&1

# plot mm vs qm parameters
python plot-mm-vs-qm-parameters.py                      \
     -i "topology-value-by-parameter/${REFERENCE_FF}"   \
     -ff "${REFERENCE_FF}.offxml"  \
     -o "images/mm-vs-qm/${REFERENCE_FF}" > logs/plot-mm-vs-qm-parameters-${REFERENCE_FF}.log 2>&1

# get ddes for all to all
python get-all-to-all-dde.py > logs/get-all-to-all-ddes.log 2>&1

# plot all-to-all ddes
python plot-ddes.py \
    -i "ddEs"       \
    -o "images/ddEs" > logs/plot-ddes.log 2>&1
