#!/bin/bash

mkdir -p job


for i in $(seq 30 87); do
    start=$((i * 1000))
    end=$((start + 1000))

    cat <<EOF > job/internvl_job_${i}.sh
#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=10:00:00
#PBS -P gcc50517
#PBS -N internvl_job_$i

export HF_HOME="/home/acc12645mq/amemiya/workspace/hf_cache/huggingface"

module load cuda/11.8/11.8.0

cd /home/acc12645mq/amemiya/workspace/Holistic-CLIP/synthetic_caption
source .venv/bin/activate

python /home/acc12645mq/amemiya/workspace/Holistic-CLIP/synthetic_caption/compute_recon.py --dataset_root /home/acc12645mq/amemiya/workspace/data/fractal_dataset --start_idx $start --end_idx $end --ckpt_path /home/acc12645mq/amemiya/workspace/Holistic-CLIP/synthetic_caption/checkpoints/droid.pth
EOF
    qsub -m n job/internvl_job_${i}.sh
done
