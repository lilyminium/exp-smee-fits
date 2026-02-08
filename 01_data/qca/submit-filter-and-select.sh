#!/usr/bin/env bash
#SBATCH -J filter-and-select
#SBATCH -p cpu
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output slurm-%x.%A.out


source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yml

# combine opt and torsiondrives

#python combine-opt-and-torsiondrive.py      \
#    -i output/optimizations/dataset/minimum \
#    -i output/torsiondrives/dataset/minimum \
#    -o output/combined/minimum-all > logs/combine-opt-and-torsiondrive-minimum.txt 2>&1

#python combine-opt-and-torsiondrive.py          \
#    -i output/optimizations/dataset/trajectory  \
#    -i output/torsiondrives/dataset/trajectory  \
#    -o output/combined/trajectory-all > logs/combine-opt-and-torsiondrive-trajectory.txt 2>&1

# filter parameterizable -- we can do this just once

#python select-parameterizable-smiles.py        \
#    -i output/combined/minimum-all                 \
#    -o smiles/combined-parameterizable.smi     \
#    -ff openff-2.3.0.offxml                    \
#    -np 8 > logs/select-parameterizable-smiles.txt 2>&1

# select diverse
python select-diverse-smiles.py                \
    -i smiles/combined-parameterizable.smi     \
    -o smiles/combined-diverse.smi             \
    -n 12000 > logs/select-diverse-smiles.txt 2>&1

# select train-valid-test split
python split-train-test.py \
    -i smiles/combined-diverse.smi \
    -o smiles/combined-diverse-split.json \
    -tf 0.8  -vf 0.1 > logs/split-train-test.txt 2>&1

# convert back to dataset
python filter-by-smiles.py \
    -i output/combined/minimum-all \
    -s smiles/combined-diverse-split.json \
    -o output/combined/minimum-diverse > logs/filter-by-smiles-minimum.txt 2>&1

python filter-by-smiles.py \
    -i output/combined/trajectory-all \
    -s smiles/combined-diverse-split.json \
    -o output/combined/trajectory-diverse > logs/filter-by-smiles-trajectory.txt 2>&1
