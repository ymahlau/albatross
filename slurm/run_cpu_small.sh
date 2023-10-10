#!/bin/bash
#SBATCH --job-name=cpu
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=amo,tnt,ai,haku,lena,taurus,enos,stahl,pci,pcikoe,itp,iqo,imuk,phd,iwes
#SBATCH --cpus-per-task=16
#SBATCH --mem=25G
#SBATCH --verbose
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
srun -c 16 -v /bigwork/nhmlmahy/miniconda3/envs/battlesnake-rl/bin/python start_training.py config=config_small hydra.job.chdir=True
