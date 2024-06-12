#!/bin/sh
#$ -cwd
#$ -l node_f=2
#$ -l h_rt=2:00:00
#$ -o outputs/idefics2/$JOB_ID
#$ -e outputs/idefics2/$JOB_ID
#$ -p -5

# Load modules
module use ~/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.3/2.19.3
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

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
TRAIN_STEPS=25000

# optimizer config
LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.01
GRAD_CLIP=1
# model config
CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/idefics2-8b
CHECKPOINT_SAVE_DIR=/gs/bs/tge-gc24sp03/checkpoints/idefics2-8b/LR${LR}-MINLR${MIN_LR}-WARMUP${LR_WARMUP_STEPS}-WD${WEIGHT_DECAY}-GC${GRAD_CLIP}-BS${GLOBAL_BATCH_SIZE}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# job name
JOB_NAME="idefics2-8b-t4-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

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
  --eval-interval 100 \
  --eval-iters 10 \
  --vocab-size 32003 \
  --pad-token-id 0 \
  --rms-norm-eps 1e-5 \
  --vlm-text-model-type "mistral" \
  --bf16 \
  --mixed-precision \
  --instruction-tuning \
  --visual-instruction-text-train-data-path "nielsr/docvqa_1200_examples" \
  --visual-instruction-vision-train-data-path "nielsr/docvqa_1200_examples" \
  --visual-instruction-text-valid-data-path "nielsr/docvqa_1200_examples" \
  --visual-instruction-vision-valid-data-path "nielsr/docvqa_1200_examples" \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --low-cpu-fsdp \
  --sharding-strategy FULL_SHARD \
  --checkpoint-type LOCAL_STATE_DICT \
  --fsdp-activation-checkpointing \
  --size-based-auto-wrap-policy \
  --min-params 2e7 \
  --use-mpi \
  --wandb-entity "okoge" \
  --wandb-project "vlm-recipes" \
  --wandb-name "${JOB_NAME}"
