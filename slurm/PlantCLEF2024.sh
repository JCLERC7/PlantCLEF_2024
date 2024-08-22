#!/bin/bash

#SBATCH --job-name=multiNODE-PlantCLEF_2024
#SBATCH --partition=private-ruch-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/torchrun_%j.log
#SBATCH --error=logs/torchrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joel.clerc1@hes-so.ch

mkdir -p logs

ml load GCCcore/12.2.0
ml load Python/3.10.8

nohup tensorboard --logdir logs --host 0.0.00 --port 6006 > logs/tensorboard.log 2>&1

echo "Environment Information:"
which python
python --version
which torchrun

echo "GPU Information:"
nvidia-smi

echo "Allocated Nodes: $SLURM_JOB_NODELIST"

NNODES=2
NPROC_PER_NODE=3

NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST )
NODES_ARRAY=($NODELIST)

echo "Node list: $NODELIST"

HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}' | grep -oP '(\d{1,3}\.){3}\d{1,3}')
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
MASTER_PORT=8004

echo Node IP: $HEAD_NODE_IP
export LOGLEVEL=INFO
echo "Master address=$MASTER_ADDR"

srun torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE_IP:$MASTER_PORT main.py -p data/Training_data/dataset -e 50 -b 2