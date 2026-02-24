#!/usr/bin/env bash
#SBATCH -J train
#SBATCH -p gpu
#SBATCH -t 3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out

source ~/.bashrc

conda activate smee-stack-cuda

conda env export > environment.yaml

EXPERIMENT_NAME="01_nomini_weight-1_mean_eq"
EXPERIMENT_NAME="02_mini_weight-1_mean_mb-256_eq"
#EXPERIMENT_NAME="03_mini_weight-10_mean_mb-256_eq"
#EXPERIMENT_NAME="04_mini_weight-1_mean_mb-256_traj"
#EXPERIMENT_NAME="05_mini_weight-10_mean_mb-256_traj"
#EXPERIMENT_NAME="06_mini_weight-1_min_mb-256_traj"
#EXPERIMENT_NAME="07_mini_weight-10_min_mb-256_traj"
EXPERIMENT_NAME="08_mini-10_min_mb-256_traj-f5e10"
#EXPERIMENT_NAME="09_mini-50_min_mb-256_traj-f5e10"
#EXPERIMENT_NAME="10_nomini_weight-10_min_traj-f5e10"
#EXPERIMENT_NAME="11_nomini_weight-10_min_eq-f5e10"
#EXPERIMENT_NAME="12_mini-10_min_mb-512_traj-f5e10"
#EXPERIMENT_NAME="13_mini-10_min_mb-1024_traj-f5e10"
EXPERIMENT_NAME="14_nomini_weight-30_min_traj-f5e10"
EXPERIMENT_NAME="15_nomini_weight-30_mean_traj-f5e10"
EXPERIMENT_NAME="16_nomini_weight-50_min_eq-f5e10"
EXPERIMENT_NAME="17_nomini_weight-50_mean_eq-f5e10"
EXPERIMENT_NAME="18_mini_weight-50_min_mb-1024_traj-f5e10"
EXPERIMENT_NAME="19_mini_weight-100_min_mb-1024_traj-f5e10"
EXPERIMENT_NAME="20_mini_weight-100_min_mb-2048_traj-f5e10"
EXPERIMENT_NAME="21_mini_weight-100_mean_mb-2048_traj-f5e10"
EXPERIMENT_NAME="22_mini_weight-100_min_mb-2048_eq-f410"
EXPERIMENT_NAME="23_mini_weight-500_min_mb-2048_traj-f5e10"
EXPERIMENT_NAME="24_nomini_weight-100_min_traj-f5e10"
EXPERIMENT_NAME="25_nomini_weight-100_min_eq-f5e10"
EXPERIMENT_NAME="26_nomini_weight-500_min_traj-f5e10"
EXPERIMENT_NAME="27_nomini_weight-1000_min_traj-f5e10"
EXPERIMENT_NAME="28_mini_weight-100_min_mb-2048_eq-f5e10_e200"
EXPERIMENT_NAME="29_mini_weight-100_min_mb-2048_traj-f5e10_e200"
EXPERIMENT_NAME="30_nomini_weight-500_min_traj-f5e10_scales"
EXPERIMENT_NAME="31_nomini_weight-500_min_traj-f5e10_scales_e600"
EXPERIMENT_NAME="32_nomini_weight-1000_min_traj-f5e10_scales_e600"
EXPERIMENT_NAME="33_mini_weight-500_min_mb-1024_traj-f5e10"
EXPERIMENT_NAME="34_mini_weight-1000_min_mb-1024_traj-f5e10"
EXPERIMENT_NAME="35_mini_weight-1000_min_mb-1024_traj-f5e10_scales"
EXPERIMENT_NAME="36_mini_weight-1000_min_mb-2048_traj-f5e10_scales_e200"
EXPERIMENT_NAME="37_nomini_weight-1000_min_traj-f5e10_scales"
DIRECTORY="experiments/${EXPERIMENT_NAME}"


mkdir logs

cd $DIRECTORY

echo "Slurm Job ID: ${SLURM_JOB_ID}" > "job_id.txt"

python ../../train.py                     \
    -c "config.yaml"   \
    -o "."               \
    >  "train.log" 2>&1

if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    python ../../plot-intermediate-values.py -i intermediate -o output -n 10 > "plot-intermediate.log" 2>&1
else
    echo "Training failed with exit code $?. Check train.log for details."
    exit 1
fi



if [ $? -eq 0 ]; then
    cd ../../../05_benchmarks/qm && sbatch --export=FFNAME="${EXPERIMENT_NAME}" run-benchmark-experiment.sh
else
    echo "Training failed with exit code $?. Skipping benchmark submission."
    exit 1
fi
