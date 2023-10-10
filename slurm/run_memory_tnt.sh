#!/bin/bash
#SBATCH --job-name=mem
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=1:00:00 # (HH:MM:SS)
#SBATCH --partition=gpu_normal_stud
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --verbose
#SBATCH --gres=gpu:turing:2
#SBATCH --constraint=enife
echo "Hier beginnt die Ausführung/Berechnung"
srun -c 8 --gres=gpu:1 -v /home/mahlauya/nobackup/miniconda3/envs/battlesnake-rl/bin/python start_memory_test.py


