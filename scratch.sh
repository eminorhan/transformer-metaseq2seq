#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=160GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=meta
#SBATCH --output=meta_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/transformer-metaseq2seq/scratch.py

echo "Done"
