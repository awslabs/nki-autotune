#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=slurm_logs/test.out
#SBATCH --error=slurm_logs/test.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Starting at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"

# Run the Python script
python benchmark_matmul.py

echo "Job finished at $(date)"