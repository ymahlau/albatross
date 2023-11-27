#!/bin/bash
#SBATCH --job-name=gpu_small
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=tnt
#SBATCH --cpus-per-task=17
#SBATCH --mem=100G
#SBATCH --verbose
#SBATCH --gres=gpu:rtx3090:2
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
cd ..
srun -c 17 --gres=gpu:rtx3090:2 -v /bigwork/nhmlmahy/miniforge3/envs/albatross-env/bin/python start_training.py config=cfg_oc_proxy_0 hydra.job.chdir=True
