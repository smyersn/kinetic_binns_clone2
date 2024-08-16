#!/bin/bash

#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 7:00:00
#SBATCH --constraint=rhel8
#SBATCH --output=myjob.out
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=smyersn@ad.unc.edu

source ~/.bashrc
conda activate binns
python generate_data.py