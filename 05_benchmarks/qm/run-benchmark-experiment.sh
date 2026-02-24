#!/usr/bin/env bash
#SBATCH -J benchmark
#SBATCH -p cpu
#SBATCH -t 20:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out

source ~/.bashrc

conda activate smee-stack-cuda

# Validate that FFNAME was provided
if [ -z "$FFNAME" ]; then
    echo "Error: FFNAME variable not provided"
    echo "Usage: sbatch --export=FFNAME=<experiment-name> run-benchmark-experiment.sh"
    exit 1
fi

echo "Running benchmark for force field: $FFNAME"

REFERENCE_FF="openff-2.3.0"
EXPERIMENT_DIR="../../04_experiment/experiments/${FFNAME}"

# go there, look at the job id in job_id.txt, and run seff to get the resources used for reference, then echo to $DIRECTORY/training-resources.txt for reference
REFERENCE_JOB_ID=$(cat "${EXPERIMENT_DIR}/job_id.txt" | awk '{print $4}')
echo "Reference training job ID: ${REFERENCE_JOB_ID}"
echo "Getting training resources used for reference job ID ${REFERENCE_JOB_ID}..."
seff ${REFERENCE_JOB_ID} > "${EXPERIMENT_DIR}/training-resources.txt"
echo "Saved training resources for reference job to ${EXPERIMENT_DIR}/training-resources.txt"

FF_ORIGIN="${EXPERIMENT_DIR}/final-force-field.offxml"

mkdir ../forcefields
FORCEFIELD="../forcefields/${FFNAME}.offxml"

echo "Copying ${FF_ORIGIN} to ${FORCEFIELD}"
cp ${FF_ORIGIN} ${FORCEFIELD}

LOGDIR="logs/${FFNAME}"
mkdir -p ${LOGDIR}

python benchmark-mm-optimization.py                             \
        --n-workers                     300                     \
        --worker-type                   "slurm"                 \
        --batch-size                    100                     \
        --memory                        16                      \
        --walltime                      480                     \
        --queue                         "preemptable"           \
        --conda-environment             "smee-stack-cuda"       \
    -d       "data/optimization"                                \
    -ff      $FORCEFIELD > ${LOGDIR}/benchmark-opt.log 2>&1

# compute RMSDs and TFDs
python get-rmsds-and-tfds.py                                    \
        --n-workers                     300                     \
        --worker-type                   "slurm"                 \
        --batch-size                    200                     \
        --memory                        8                       \
        --walltime                      480                     \
        --queue                         "preemptable"           \
        --conda-environment             "smee-stack-cuda"       \
    -d       "data/optimization"                                \
    -r       "rmsd-tfd"                                         \
    -ff      $FORCEFIELD > ${LOGDIR}/get-rmsds-and-tfds.log 2>&1

# get all to all RMSD matrix for ddEs
python get-all-to-all-rmsds-and-tfds.py                         \
        --n-workers                     300                     \
        --worker-type                   "slurm"                 \
        --batch-size                    200                     \
        --memory                        8                       \
        --walltime                      480                     \
        --queue                         "preemptable"           \
        --conda-environment             "smee-stack-cuda"       \
    -d       "data/optimization"                                \
    -r       "all-to-all-rmsd"                     \
    -ff      $FORCEFIELD > ${LOGDIR}/get-all-to-all-rmsd.log 2>&1

# compute topology values for ICRMSD comparison
python compute-topology-values.py                              \
       --n-workers                     300                     \
       --worker-type                   "slurm"                 \
       --batch-size                    500                     \
       --memory                        8                       \
       --walltime                      480                     \
       --queue                         "preemptable"           \
       --conda-environment             "smee-stack-cuda"       \
   -i       "data/optimization/${FFNAME}"                      \
   -o       "topology-values/${FFNAME}" > ${LOGDIR}/compute-topology-values.log 2>&1


sbatch --export=FFNAME="${FFNAME}" run-topology-value-by-parameter.sh
