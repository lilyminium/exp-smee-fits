#!/usr/bin/env bash
#SBATCH -J plot-intermediate
#SBATCH -p preemptable
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --array=26
#SBATCH --output slurm-%x.%A.out

source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yaml

# EXPERIMENT_NAME="01_nomini_weight-1_mean_eq"
# EXPERIMENT_NAME="02_mini_weight-1_mean_mb-256_eq"
# EXPERIMENT_NAME="03_mini_weight-10_mean_mb-256_eq"
# EXPERIMENT_NAME="04_mini_weight-1_mean_mb-256_traj"
# EXPERIMENT_NAME="05_mini_weight-10_mean_mb-256_traj"
# EXPERIMENT_NAME="06_mini_weight-1_min_mb-256_traj"
# EXPERIMENT_NAME="07_mini_weight-10_min_mb-256_traj"
# EXPERIMENT_NAME="08_mini-10_min_mb-256_traj-f5e10"
# EXPERIMENT_NAME="09_mini-50_min_mb-256_traj-f5e10"
# EXPERIMENT_NAME="10_nomini_weight-10_min_traj-f5e10"
# EXPERIMENT_NAME="11_nomini_weight-10_min_eq-f5e10"
# EXPERIMENT_NAME="12_mini-10_min_mb-512_traj-f5e10"
# EXPERIMENT_NAME="13_mini-10_min_mb-1024_traj-f5e10"
# EXPERIMENT_NAME="14_nomini_weight-30_min_traj-f5e10"
# EXPERIMENT_NAME="15_nomini_weight-30_mean_traj-f5e10"
# EXPERIMENT_NAME="16_nomini_weight-50_min_eq-f5e10"
# EXPERIMENT_NAME="17_nomini_weight-50_mean_eq-f5e10"
# EXPERIMENT_NAME="18_mini_weight-50_min_mb-1024_traj-f5e10"
# EXPERIMENT_NAME="19_mini_weight-100_min_mb-1024_traj-f5e10"
# EXPERIMENT_NAME="20_mini_weight-100_min_mb-2048_traj-f5e10"
# EXPERIMENT_NAME="21_mini_weight-100_mean_mb-2048_traj-f5e10"
# EXPERIMENT_NAME="22_mini_weight-100_min_mb-2048_eq-f410"
# EXPERIMENT_NAME="23_mini_weight-500_min_mb-2048_traj-f5e10"

EXPERIMENT_NAMES=(
    "01_nomini_weight-1_mean_eq"
    "02_mini_weight-1_mean_mb-256_eq"
    "03_mini_weight-10_mean_mb-256_eq"
    "04_mini_weight-1_mean_mb-256_traj"
    "05_mini_weight-10_mean_mb-256_traj"
    "06_mini_weight-1_min_mb-256_traj"
    "07_mini_weight-10_min_mb-256_traj"
    "08_mini-10_min_mb-256_traj-f5e10"
    "09_mini-50_min_mb-256_traj-f5e10"
    "10_nomini_weight-10_min_traj-f5e10"
    "11_nomini_weight-10_min_eq-f5e10"
    "12_mini-10_min_mb-512_traj-f5e10"
    "13_mini-10_min_mb-1024_traj-f5e10"
    "14_nomini_weight-30_min_traj-f5e10"
    "15_nomini_weight-30_mean_traj-f5e10"
    "16_nomini_weight-50_min_eq-f5e10"
    "17_nomini_weight-50_mean_eq-f5e10"
    "18_mini_weight-50_min_mb-1024_traj-f5e10"
    "19_mini_weight-100_min_mb-1024_traj-f5e10"
    "20_mini_weight-100_min_mb-2048_traj-f5e10"
    "21_mini_weight-100_mean_mb-2048_traj-f5e10"
    "22_mini_weight-100_min_mb-2048_eq-f410"
    "23_mini_weight-500_min_mb-2048_traj-f5e10"
    "24_nomini_weight-100_min_traj-f5e10"
    "25_nomini_weight-100_min_eq-f5e10"
    "26_nomini_weight-500_min_traj-f5e10"
    "27_nomini_weight-1000_min_traj-f5e10"
    "28_mini_weight-100_min_mb-2048_eq-f5e10_e200"
    "29_mini_weight-100_min_mb-2048_traj-f5e10_e200"
)
EXPERIMENT_NAME="${EXPERIMENT_NAMES[${SLURM_ARRAY_TASK_ID}]}"


DIRECTORY="experiments/${EXPERIMENT_NAME}"


cd $DIRECTORY


python ../../plot-intermediate-values.py -i intermediate -o output -n 10
