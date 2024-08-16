#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=5g
#SBATCH -t 2:00:00
#SBATCH --constraint=rhel8
#SBATCH --output=myjob.out
#SBATCH --mail-type=end
#SBATCH --mail-user=smyersn@ad.unc.edu

source ~/.bashrc
conda activate binns
python generate_data.py