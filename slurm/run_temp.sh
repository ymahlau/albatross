#!/bin/bash
#SBATCH --job-name=temp
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=lena,amo,haku,taurus,imuk,stahl,phd,tnt,ai
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --verbose
echo "Hier beginnt die Ausführung/Berechnung"
module load GCC/11.2.0
cd ..
srun -c 8 -v /bigwork/nhmlmahy/miniconda3/envs/albatross-env/bin/python start_temp.py

# interactive session with:
#  srun --partition=amo,tnt,ai,haku,lena --mincpus 8 --mem 20G --time=2:00:00 --pty bash -l -i

# good slurm command:
# squeue --state=R -p tnt -o "%.8u %.10i %.10l %.10M %.10m %.12c %.10R"
# scontrol show job <id>
# --qos='_mahlauya+'
