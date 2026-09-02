#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=arb24_0001
#SBATCH --partition=cac_cpu
#SBATCH --time=48:00:00
#SBATCH --qos=longrun
#SBATCH -c 8

# Submit as follows:
# sbatch submit_hierarchical_training.sh

# Load any necessary modules
module load anaconda3

# Activate the virtual environment
source /opt/ohpc/pub/software/anaconda3/etc/profile.d/conda.sh
conda activate SCARCH_HIERARSIR
unset PYTHONHOME
unset PYTHONPATH

# Run your Python script
python -u hierarchical_training.py

# Deactivate the virtual environment after the run
conda deactivate