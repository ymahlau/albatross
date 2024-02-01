#!/bin/bash
#SBATCH --job-name=nd7_4nd7
#SBATCH --output=slurm-%j-%a-out.txt
#SBATCH --time=96:00:00 # (HH:MM:SS)
#SBATCH --partition=partition_name
#SBATCH --cpus-per-task=42
#SBATCH --mem=100G
#SBATCH --verbose
#SBATCH --gres=gpu:rtx3090:3
#SBATCH --array=0-4
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
cd ..
srun -c 42 --gres=gpu:rtx3090:3 -v "path_to_python_executable" start_training_az.py $SLURM_ARRAY_TASK_ID hydra.job.chdir=True
