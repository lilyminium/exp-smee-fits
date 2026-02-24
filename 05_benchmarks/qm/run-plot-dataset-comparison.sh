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
OUTPUTDIR="images-dataset"

mkdir -p "$OUTPUTDIR"
mkdir -p ${OUTPUTDIR}/rmsd-tfd
mkdir -p logs

FF_ARGS=(
    -ff "Trajectory (FF24)" "24_nomini_weight-100_min_traj-f5e10"
	-ff "Minima (FF25)" "25_nomini_weight-100_min_eq-f5e10"
	-ff "Sage 2.3.0" "openff-2.3.0"
)

# plot rmsd and tfd
python plot-rmsd-tfd.py -i "rmsd-tfd" "${FF_ARGS[@]}" \
	-o "${OUTPUTDIR}/rmsd-tfd"


# get ddes for all to all
python get-all-to-all-dde.py > logs/get-all-to-all-ddes.log 2>&1

# plot all-to-all ddes
python plot-ddes.py \
    -i "ddEs"       \
	"${FF_ARGS[@]}" \
    -o "${OUTPUTDIR}/ddEs"
