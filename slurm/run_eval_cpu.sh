#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm-%j-%a-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=lena,amo,haku,taurus,imuk,stahl,phd
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --verbose
#SBATCH --array=0-75
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
srun -c 8 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_evaluation.py config=cfg_trnm_4d7 $SLURM_ARRAY_TASK_ID hydra.job.chdir=True
#srun -c 16 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_depth.py config=config_depth0 hydra.job.chdir=True
# srun -c 2 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_temp.py