#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm-%j-%a-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=cpu_normal_stud
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --verbose
#SBATCH --array=0
echo "Hier beginnt die Ausführung/Berechnung"
srun -c 8 -v /home/mahlauya/nobackup/miniconda3/envs/battlesnake-rl/bin/python start_evaluation.py config=cfg_trnm $SLURM_ARRAY_TASK_ID hydra.job.chdir=True