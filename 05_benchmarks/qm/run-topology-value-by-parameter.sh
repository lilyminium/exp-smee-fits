#!/usr/bin/env bash
#SBATCH -J benchmark
#SBATCH -p cpu
#SBATCH -t 10:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48gb
#SBATCH --output slurm-%x.%A.out

source ~/.bashrc

conda activate smee-stack-cuda

# Validate that FFNAME was provided
if [ -z "$FFNAME" ]; then
    echo "Error: FFNAME variable not provided"
    echo "Usage: sbatch --export=FFNAME=<experiment-name> run-topology-value-by-parameter.sh"
    exit 1
fi

REFERENCE_FF="openff-2.3.0"
FORCEFIELD="../forcefields/${FFNAME}.offxml"

LOGDIR="logs/${FFNAME}"
mkdir -p ${LOGDIR}

for top in Bonds Angles ProperTorsions ImproperTorsions; do

    echo "Computing topology values by parameter for ${top}"

    # compute topology values by parameters
    python topology-value-by-parameter.py       \
        -i "topology-values"                    \
        -o "topology-value-by-parameter/${REFERENCE_FF}"        \
        -t $top                                 \
        -l "topology-labels/${REFERENCE_FF}"    \
        -ff $FFNAME > ${LOGDIR}/topology-value-by-parameter-${REFERENCE_FF}-${top}.log 2>&1
done
