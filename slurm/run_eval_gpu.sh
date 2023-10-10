#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=14
#SBATCH --mem=75G
#SBATCH --verbose
#SBATCH --gres=gpu:1
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
#srun -c 64 --gres=gpu:1 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_evaluation.py config=config_tournament hydra.job.chdir=True
srun -c 14 --gres=gpu:1 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_depth.py config=config_depth_0 hydra.job.chdir=True
