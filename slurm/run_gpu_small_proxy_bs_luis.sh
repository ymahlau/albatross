#!/bin/bash
#SBATCH --job-name=proxy_4nd7_nd7
#SBATCH --output=slurm-%j-%a-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=tnt
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --verbose
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --array=0-9
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
cd ..
srun -c 28 --gres=gpu:rtx3090:2 -v /bigwork/nhmlmahy/miniforge3/envs/albatross-env/bin/python start_training_proxy_bs_luis.py $SLURM_ARRAY_TASK_ID hydra.job.chdir=True