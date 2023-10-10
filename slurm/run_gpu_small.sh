#!/bin/bash
#SBATCH --job-name=gpu_small
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=tnt
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --verbose
#SBATCH --gres=gpu:rtx3090:2
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
srun -c 28 --gres=gpu:rtx3090:2 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_training.py config=config_small hydra.job.chdir=True
