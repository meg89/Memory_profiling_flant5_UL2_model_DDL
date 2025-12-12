#!/bin/bash
##salloc --nodes 1 --ntasks-per-node=4 --gpus-per-node=4 --qos interactive --time 1:00:00 --constraint gpu --account=m4431_g
##salloc --nodes 2 --ntasks-per-node=4 --gpus-per-node=4 --qos interactive --time 1:00:00 --constraint gpu --account=m4431_g
#SBATCH --job-name=flant5-zero1
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=megha.agrawal@rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/flant5_z1%j.out
#SBATCH --error=logs/flant5_z1%j.err

#change the directory path and load the modules
cd /pscratch/sd/m/megha89/flant5-proj/zero1
source ../venv-ds/bin/activate

echo "=========================================="
echo "4-GPU Fixed Batch Training"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo ""
echo "=========================================="

mkdir -p logs
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=11111

./generate_hostfile.sh

# Launch training with DeepSpeed
lsof -i :$MASTER_PORT

deepspeed --num_gpus=4 --num_nodes=1 --hostfile=ds_hostfile --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT src/train.py --config configs/flanT5-config.yaml

#deepspeed --num_gpus=4 --num_nodes=2 --hostfile=ds_hostfile --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT src/train.py --config configs/flanT5-config.yaml
echo "Training completed successfully at $(date)"

python src/evaluate_flant5.py --config configs/flanT5-config.yaml --model_path=outputs/final_model/ --split test --batch_size=8 --num_samples=128 --save_predictions

