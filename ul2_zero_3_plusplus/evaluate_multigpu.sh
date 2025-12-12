
# UL2-20B Evaluation on GovReport - Multi-GPU (Data Parallel)
# This script splits the dataset across 4 GPUs for faster evaluation

echo "=========================================="
echo "UL2-20B Evaluation - Multi-GPU"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "GPUs: 4"
echo "Start time: $(date)"
echo "=========================================="


# Activate virtual environment
source /pscratch/sd/m/megha89/ul2-proj/venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p eval_results

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export OMP_NUM_THREADS=8

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Configuration
CONFIG_FILE="/pscratch/sd/m/megha89/ul2-proj/configs/ul2-config.yaml"
MODEL_PATH="/pscratch/sd/m/megha89/ul2-proj/ul2_govreport/final_model"
OUTPUT_DIR="eval_results/ul2_govreport_multi_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Running Multi-GPU Evaluation"
echo "Strategy: Data Parallelism (split dataset)"
echo "=========================================="

# Run evaluation in parallel on 4 GPUs
# Each GPU processes a portion of the dataset
for GPU_ID in 0 1 2 3; do
    echo "Starting evaluation on GPU ${GPU_ID}..."
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/evaluate_ul2.py \
        --config ${CONFIG_FILE} \
        --model_path ${MODEL_PATH} \
        --split test \
        --batch_size 1 \
        --output_dir ${OUTPUT_DIR}/gpu_${GPU_ID} \
        --device cuda:0 &
    
    # Store PID
    PIDS[$GPU_ID]=$!
done

# Wait for all processes to complete
echo ""
echo "Waiting for all GPU processes to complete..."
for GPU_ID in 0 1 2 3; do
    wait ${PIDS[$GPU_ID]}
    EXIT_CODE=$?
    echo "GPU ${GPU_ID} finished with exit code: ${EXIT_CODE}"
done

echo ""
echo "=========================================="
echo "Merging Results from All GPUs"
echo "=========================================="

# Merge results (this is a simplified merge - you may need custom logic)
python - <<EOF

import json
import os

output_dir = "${OUTPUT_DIR}"
merged_results = {
    'rouge1': [],
    'rouge2': [],
    'rougeL': [],
    'rougeLsum': []
}

for gpu_id in range(4):
    result_file = f"{output_dir}/gpu_{gpu_id}/eval_results_test.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            for metric in merged_results.keys():
                if metric in data.get('metrics', {}):
                    merged_results[metric].append(data['metrics'][metric])

# Average the results
final_results = {}
for metric, values in merged_results.items():
    if values:
        final_results[metric] = sum(values) / len(values)

print("Merged ROUGE Scores:")
print("="*50)
for metric, score in final_results.items():
    print(f"{metric}: {score:.4f}")
print("="*50)

# Save merged results
with open(f"{output_dir}/merged_results.json", 'w') as f:
    json.dump({
        'metrics': final_results,
        'num_gpus': 4,
        'strategy': 'data_parallel'
    }, f, indent=2)
EOF

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

