#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/inference/
#$ -cwd
# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

set -e

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

HF_MODEL_PATH=/path/to/huggingface-checkpoint/idefics2
HF_PROCESSOR_PATH=/path/to/huggingface-processor/idefics2
IMAGE_PATH=images/drive_situation_image.jpg
PROMPT="In the situation in the image, is it permissible to start the car when the light turns green?"

python tools/inference/inference.py \
  --model-path $HF_MODEL_PATH \
  --processor-path $HF_PROCESSOR_PATH \
  --image-path $IMAGE_PATH \
  --prompt $PROMPT
