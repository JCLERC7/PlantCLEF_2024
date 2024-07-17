#!/bin/bash

#SBATCH --job-name=multiGPU-PlantCLEF_2024
#SBATCH --partition=shared-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:titan:2
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --output=logs/torchrun_%j.log
#SBATCH --error=logs/torchrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joel.clerc1@hes-so.ch

mkdir -p logs

ml load GCC/11.3.0
ml load OpenMPI/4.1.4
ml load PyTorch/1.12.1-CUDA-11.7.0

echo "Environment Information:"
which python
python --version
which torchrun

echo "GPU Information:"
nvidia-smi

NUM_PROCESSES=2

torchrun --standalone --nproc_per_node=$NUM_PROCESSES main.py -p data/Training_data/light_dataset -e 10
