#!/bin/bash
##salloc --nodes 3 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 --qos interactive --time 1:00:00 --constraint gpu --account=m4431_g
#SBATCH --job-name=ul2-govrpt
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=ma2407@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/ul2_%j.out
#SBATCH --error=logs/ul2_%j.err

#change the directory path and load the modules
cd /pscratch/sd/m/megha89/ul2-proj
#source ../flant5-proj/venv-ds/bin/activate
source venv-ul/bin/activate
echo "=========================================="
echo "Multi-GPU Fixed Batch Training"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "=========================================="

# Set environment for compilation -needed only for zero3-qwz operation
#export TORCH_CUDA_ARCH_LIST="8.0"  # For A100
#export DS_BUILD_OPS=1
#export DS_BUILD_QUANTIZER=1

# Build DeepSpeed kernels
#cd /pscratch/sd/m/megha89/flant5-proj
#python -c "import deepspeed; deepspeed.ops.op_builder.QuantizerBuilder().load()"

#----------------needed only for zero3-qwz operation-----------

mkdir -p logs
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=11111

./generate_hostfile.sh

# ✅✅✅ CRITICAL FIX: Prevent memory fragmentation
#export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# Launch training with DeepSpeed
lsof -i :$MASTER_PORT
#kill -9 $(lsof -ti:29500)

deepspeed --num_gpus=4 --num_nodes=3 --hostfile=ds_hostfile --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT src/train_ZERO3.py --config configs/ul2-config.yaml

#deepspeed --num_gpus=2 --num_nodes=4 --hostfile=ds_hostfile --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT src/train_ZERO3.py --config configs/ul2-config.yaml
echo "Training completed successfully at $(date)"

#python src/evaluate_ul2.py --config configs/ul2-config.yaml --model_path=ul2_govreport/final_model/ --split test --batch_size=8 --num_samples=768 --save_predictions

srun torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT python src/evaluate_ul2.py --config configs/ul2-config.yaml --model_path=ul2_govreport/final_model/ --split test --batch_size=8 --num_samples=768 --save_predictions

torchrun --standalone --nproc_per_node=4 src/evaluate_ul2.py --config configs/ul2-config.yaml --model_path=ul2_govreport/final_model/ --split test --batch_size=1 --num_samples=768 --save_prediction
