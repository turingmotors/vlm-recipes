#!/bin/sh
#$ -cwd
#$ -l node_f=16
#$ -l h_rt=50:00:00
#$ -o outputs/llava-v1.6/34b/tikz/step2/continual/$JOB_ID.log
#$ -e outputs/llava-v1.6/34b/tikz/step2/continual/$JOB_ID.log
#$ -p -5

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

set -e

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# training config
SEQ_LENGTH=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=128
TRAIN_EPOCHS=1

# freeze
VISION_MODEL_FREEZE=true
TEXT_MODEL_FREEZE=false

FREEZE_ARGS=""
if [ "$VISION_MODEL_FREEZE" = true ] || [ "$TEXT_MODEL_FREEZE" = true ]; then
  echo "Freezing model"
  FREEZE_ARGS="--use-freeze"
  FREEZE_ARGS="${FREEZE_ARGS} --no-save-optimizer-state"

  if [ "$VISION_MODEL_FREEZE" = true ]; then
    FREEZE_ARGS="${FREEZE_ARGS} --freeze-vlm-vision-model"
  fi
  if [ "$TEXT_MODEL_FREEZE" = true ]; then
    FREEZE_ARGS="${FREEZE_ARGS} --freeze-vlm-text-model"
  fi
fi

# optimizer config
LR=2.0E-5
MIN_LR=2.0E-6

WEIGHT_DECAY=0.0
GRAD_CLIP=1
# model config
CHECKPOINT_DIR=/gs/bs/tge-gc24sp03/hf_checkpoints/llava-v1.6-34b-hf
CHECKPOINT_SAVE_DIR=/gs/bs/tge-gc24sp03/checkpoints/llava-v1.6-34b/tikz-local-step2-continual/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-GC${GRAD_CLIP}-BS${GLOBAL_BATCH_SIZE}-loss-mask-step2-all

mkdir -p ${CHECKPOINT_SAVE_DIR}

# dataset
DATASET_PATH="/gs/bs/tge-gc24sp03/datasets/tikz/step2-1-2-3-5-6-7-10-11-merge_train.json"

# job name
JOB_NAME="llava-v1.6-34b-t4-tikz-local-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-loss-mask"

# base model: https://huggingface.co/llava-hf/llava-v1.6-34b-hf/blob/main/config.json
# text model: https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B/blob/main/config.json

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -bind-to none \
  -x PATH \
  python examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --epoch ${TRAIN_EPOCHS} \
  --split 949,50,1 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --lr-decay-iters ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-5 \
  --save-interval 500 \
  --eval-interval 500 \
  --eval-iters 10 \
  --vocab-size 64064 \
  --vlm-text-hidden-size 7168 \
  --vlm-text-intermediate-size 20480 \
  --vlm-text-num-attention-heads 56 \
  --vlm-text-num-hidden-layers 60 \
  --vlm-text-num-key-value-heads 8 \
  --vlm-text-rope-theta 5000000.0 \
  --vlm-vision-vocab-size 32000 \
  --vlm-vision-model-type "clip_vision_model" \
  --vlm-vision-hidden-size 1024 \
  --vlm-vision-intermediate-size 4096 \
  --vlm-vision-num-attention-heads 16 \
  --vlm-vision-num-hidden-layers 24 \
  --vlm-vision-image-size 336 \
  --vlm-vision-patch-size 14 \
  --vlm-vision-projection-dim 768 \
  --pad-token-id 0 \
  --rms-norm-eps 1e-5 \
  --vlm-text-model-type "llama" \
  --bf16 \
  --mixed-precision \
  --instruction-tuning \
  --instruction-tuning-type "LLaVA_PreTrain" \
  --visual-instruction-text-train-data-path ${DATASET_PATH} \
  --visual-instruction-vision-train-data-path "" \
  --visual-instruction-text-valid-data-path ${DATASET_PATH} \
  --visual-instruction-vision-valid-data-path "" \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --low-cpu-fsdp \
  --sharding-strategy FULL_SHARD \
  --checkpoint-type LOCAL_STATE_DICT \
  --fsdp-activation-checkpointing \
  ${FREEZE_ARGS} \
  --use-mpi \
  --wandb-entity "prj-jalm" \
  --wandb-project "diagram-vlm" \
  --wandb-name "${JOB_NAME}"
