#!/bin/bash
#SBATCH --job-name=gpu_small
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=gpu_normal
#SBATCH --cpus-per-task=17
#SBATCH --mem=50G
#SBATCH --verbose
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
echo "Hier beginnt die Ausführung/Berechnung"
cd ..
srun -c 17 --gres=gpu:2 -v /home/mahlau/nobackup/miniforge3/envs/albatross-env/bin/python start_training.py config=cfg_oc_proxy_4 hydra.job.chdir=True
