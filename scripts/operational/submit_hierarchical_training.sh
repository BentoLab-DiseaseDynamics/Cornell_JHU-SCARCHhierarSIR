#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --account=arb24_0001
#SBATCH --partition=cac_cpu
#SBATCH --time=1:00:00
#SBATCH -c 8

# Submit as follows:
# sbatch submit_hierarchical_training.sh

# Load any necessary modules
module load anaconda3

# Activate the virtual environment
conda activate HIERARCHSIR

# Run your Python script
python hierarchical_training.py

# Deactivate the virtual environment after the run
conda deactivate