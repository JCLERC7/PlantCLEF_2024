#!/bin/bash

#SBATCH --job-name=check-ports
#SBATCH --partition=private-ruch-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:05:00
#SBATCH --output=logs/check_ports_%j.log
#SBATCH --error=logs/check_ports_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the list of nodes allocated to this job
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo "Master address: $MASTER_ADDR"

echo "Host node address: $HOST_NODE_ADDR"

# Print the list of nodes and their listening ports
echo "Allocated Nodes and their Listening Ports:"
for node in "${nodes_array[@]}"; do
    echo "Node: $node"

    srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address
    
    # List all listening ports using netstat
    srun --nodes=1 --ntasks=1 -w "$node" netstat -tuln
	
	gpu_info=$(srun --nodes=1 --ntasks=1 -w "$node" nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo "  Number of GPUs: $gpu_info"
    
    echo "----------------------------------------"
done

NODE1=${nodes_array[0]}
NODE2=${nodes_array[1]}

echo "Checking connection from $NODE1 to $NODE2"
srun --nodes=1 --ntasks=1 -w $NODE1 ping -c 4 $NODE2