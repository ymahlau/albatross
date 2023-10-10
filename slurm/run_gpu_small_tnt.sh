#!/bin/bash
#SBATCH --job-name=gpu_small
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=gpu_normal_stud
#SBATCH --cpus-per-task=17
#SBATCH --mem=75G
#SBATCH --verbose
#SBATCH --gres=gpu:turing:2
#SBATCH --constraint=enife
echo "Hier beginnt die Ausführung/Berechnung"
srun -c 17 --gres=gpu:turing:2 -v /home/mahlauya/nobackup/miniconda3/envs/battlesnake-rl/bin/python start_training.py config=config_duct_0 hydra.job.chdir=True

# for limit: --qos='_mahlauya+'