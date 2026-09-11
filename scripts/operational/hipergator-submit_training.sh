#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=epi
#SBATCH --qos=epi
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --output=training_%j.log

# Submit as follows:
# sbatch submit_hierarchical_training.sh

# Load any necessary modules
module purge
module load conda

# Activate the virtual environment
conda activate SCARCH_HIERARSIR

# Run your Python script
python -u hierarchical_training.py

# Deactivate the virtual environment after the run
conda deactivate