#!/bin/bash
#SBATCH --job-name=mem
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=1:00:00 # (HH:MM:SS)
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --verbose
#SBATCH --gres=gpu:v100:1
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
srun -c 4 --gres=gpu:v100:1 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_memory_test.py
