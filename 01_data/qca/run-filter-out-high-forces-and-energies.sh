#!/usr/bin/env bash
#SBATCH -J filter-high-forces-and-energies
#SBATCH -p preemptable
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=96gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yml

python filter-out-high-forces-and-energies.py       \
    -i  output/combined/trajectory-all              \
    -o  output/filtered-f5-e10/trajectory-all              \
    -tf 5 -te 10 > logs/filter-out-high-forces-and-energies-trajectory.txt 2>&1


python filter-out-high-forces-and-energies.py       \
    -i  output/combined/minimum-all                 \
    -o  output/filtered-f5-e10/minimum-all                 \
    -tf 5 -te 10 > logs/filter-out-high-forces-and-energies-minimum.txt 2>&1
