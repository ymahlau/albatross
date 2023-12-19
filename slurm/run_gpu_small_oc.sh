#!/bin/bash
#SBATCH --job-name=resp_oc
#SBATCH --output=slurm-%j-%a-out.txt
#SBATCH --time=48:00:00 # (HH:MM:SS)
#SBATCH --partition=tnt
#SBATCH --cpus-per-task=42
#SBATCH --mem=100G
#SBATCH --verbose
#SBATCH --gres=gpu:rtx3090:3
#SBATCH --array=0-24
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
cd ..
srun -c 42 --gres=gpu:rtx3090:3 -v /bigwork/nhmlmahy/miniforge3/envs/albatross-env/bin/python start_training_oc.py $SLURM_ARRAY_TASK_ID hydra.job.chdir=True
