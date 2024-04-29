#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=slurm_logs/pretrain%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --qos=m
#SBATCH --time=8:00:00

srun /bin/bash run_pretrain.sh