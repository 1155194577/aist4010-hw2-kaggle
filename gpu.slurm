#!/bin/bash
#SBATCH -J a2
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu_2h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:45:00
#SBATCH -w projgpu33
#
#   sbatch gpu.slurm --epochs 10 --batch-size 64
#   sbatch gpu.slurm --epochs 10 --batch-size 64 --limit 512 --eval-limit 256
#   PREDICT=1 sbatch gpu.slurm --epochs 10 --batch-size 64   # also write submission.csv
#
# -w projgpu33 pins the node because the 3 GB singularity image is cached in
# that node's /dev/shm; drop -w and the script re-pulls it (~5 min, via proxy).
# --cpus-per-task must stay <= 6 per GPU or Slurm demands --constraint=highcpucount.
# PREDICT=1 (or ./submit.sh --predict) runs predict.py on the checkpoint this
# job just wrote, leaving submission.csv next to train.py.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this from the repo directory}"

IMAGE=/dev/shm/pytorch-2.5.1.sif

if [[ ! -f "$IMAGE" ]]; then
    echo "image missing on $(hostname), pulling it now (~5 min)"
    export SINGULARITY_CACHEDIR=/dev/shm/sgc SINGULARITY_TMPDIR=/dev/shm/stmp
    mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
    ( cd /dev/shm && singularity pull --name pytorch-2.5.1.sif \
        docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime )
fi

echo "job ${SLURM_JOB_ID} on $(hostname), $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

singularity exec --nv "$IMAGE" python -u train.py --device cuda "$@"

if [[ "${PREDICT:-0}" == "1" ]]; then
    echo "== inference: newest checkpoints/*_best.pt =="
    singularity exec --nv "$IMAGE" python -u predict.py --device cuda --out submission.csv
fi
