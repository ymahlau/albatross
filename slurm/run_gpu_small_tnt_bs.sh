#!/bin/bash
#SBATCH --job-name=d7
#SBATCH --output=slurm-%j-%a-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=gpu_normal
#SBATCH --cpus-per-task=25
#SBATCH --mem=100G
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:turing:3
#SBATCH --constraint=enife
#SBATCH --array=0-4
echo "Hier beginnt die Ausführung/Berechnung"
cd ..
srun -c 25 --gres=gpu:3 -v /home/mahlau/nobackup/env/miniforge3/envs/albatross-env/bin/python start_training_az_det.py $SLURM_ARRAY_TASK_ID hydra.job.chdir=True
