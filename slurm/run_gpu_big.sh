#!/bin/bash
#SBATCH --job-name=big
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=200:00:00 # (HH:MM:SS)
#SBATCH --partition=tnt
#SBATCH --cpus-per-task=111
#SBATCH --mem=750G
#SBATCH --verbose
#SBATCH --gres=gpu:8
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
srun -c 111 --gres=gpu:8 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_training.py config=cfg_resnet hydra.job.chdir=True

