#!/usr/bin/env bash
#SBATCH -J filter-high-forces
#SBATCH -p preemptable
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=96gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yml

python filter-out-high-forces.py                    \
    -i  output/combined/trajectory-all              \
    -o  output/filtered/trajectory-all              \
    -t  5 > logs/filter-out-high-forces-trajectory.txt 2>&1


python filter-out-high-forces.py                    \
    -i  output/combined/minimum-all                 \
    -o  output/filtered/minimum-all                \
    -t  5 > logs/filter-out-high-forces-minimum.txt 2>&1
