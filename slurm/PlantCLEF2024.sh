#!/bin/bash

#SBATCH --job-name=multiNODE-PlantCLEF_2024
#SBATCH --partition=private-ruch-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node = 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --excusive
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/torchrun_%j.log
#SBATCH --error=logs/torchrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joel.clerc1@hes-so.ch

mkdir -p logs

ml load GCCcore/11.3.0
ml load Python/3.10.4

nohup tensorboard --logdir logs --host 0.0.00 --port 6006 > logs/tensorboard.log 2>&1

echo "Environment Information:"
which python
python --version
which torchrun

echo "GPU Information:"
nvidia-smi

NNODES=2
NPROC_PER_NODE=3

NODELIST=$(control show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=$(echo $NODELIST | awk '{print $1}')
MASTER_PORT=12345

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT main.py -p data/Training_data/dataset -e 100

deactivate