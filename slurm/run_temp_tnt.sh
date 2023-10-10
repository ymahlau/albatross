#!/bin/bash
#SBATCH --job-name=temp
#SBATCH --output=slurm-%j-%a-out.txt
#SBATCH --time=24:00:00 # (HH:MM:SS)
#SBATCH --partition=cpu_normal_stud
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --verbose
#SBATCH --array=0
echo "Hier beginnt die Ausführung/Berechnung"
srun -c 8 -v /home/mahlauya/nobackup/miniconda3/envs/battlesnake-rl/bin/python start_temp.py $SLURM_ARRAY_TASK_ID

# interactive session with:
#  srun --partition=amo,tnt,ai,haku,lena --mincpus 8 --mem 20G --time=2:00:00 --pty bash -l -i

# good slurm command:
# squeue --state=R -p tnt -o "%.8u %.10i %.10l %.10M %.10m %.12c %.10R"
# scontrol show job <id>
# --qos='_mahlauya+'
