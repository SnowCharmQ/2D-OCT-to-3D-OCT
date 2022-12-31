#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH -J trainJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

source activate olin
python -u train.py