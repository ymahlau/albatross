#!/bin/bash
#SBATCH --job-name=nd7_depth
#SBATCH --output=slurm-%j-%a-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS), infinite
#SBATCH --partition=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --exclude=helena1,helena2,helena3,helena4,cc1l01,nox
#SBATCH --array=0-49
cd ..
echo "Hier beginnt die Ausführung/Berechnung"
srun -c 2 -v /home/mahlau/nobackup/env/miniforge3/envs/albatross-env/bin/python start_temp.py $SLURM_ARRAY_TASK_ID

# interactive session with:
#  srun --partition=amo,tnt,ai,haku,lena --mincpus 8 --mem 20G --time=2:00:00 --pty bash -l -i

# good slurm command:
# squeue --state=R -p tnt -o "%.8u %.10i %.10l %.10M %.10m %.12c %.10R"
# scontrol show job <id>
# --qos='_mahlauya+'
# for fast cpu: #SBATCH --exclude=helena1,helena2,helena3,helena4,cc1l01,nox