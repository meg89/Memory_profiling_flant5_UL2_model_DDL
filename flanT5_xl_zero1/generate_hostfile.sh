rm ds_hostfile
NODE_RANK=0 
for host in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    echo "$host slots=$SLURM_GPUS_PER_NODE # node_rank=$NODE_RANK" >> ds_hostfile 
    NODE_RANK=$((NODE_RANK + 1)) 
done



