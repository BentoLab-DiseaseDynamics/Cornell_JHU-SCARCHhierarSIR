#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=cmavian
#SBATCH --partition=hpg-default
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2gb
#SBATCH --output=training_%j.log

# Submit as follows:
# sbatch submit_hierarchical_training.sh

# Load any necessary modules
module purge
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