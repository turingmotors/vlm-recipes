#!/bin/bash

set -e

# open file limit
ulimit -n 65536

# move to project directory
cd $HOME/vlm-recipes
# python virtualenv
source .env/bin/activate

# cudnn関連のエラーを避けるために.env以下のディレクトリを指定します
export LD_LIBRARY_PATH=$(pwd)/.env/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# distributed settings (for single node)
export MASTER_ADDR=localhost
export MASTER_PORT=6000
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# hostfile
export NUM_GPU_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_TYPE="h100"
NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))
export RANK=0
export WORLD_SIZE=$NUM_GPUS

# training config
SEQ_LENGTH=256
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TRAIN_EPOCHS=1
TRAIN_STEPS=25000  # no meaning (利用されない)

# optimizer config
LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000  # no meaning (利用されない)
LR_DECAY_STEPS=25000  # no meaning (利用されない)
WEIGHT_DECAY=0.01
GRAD_CLIP=1
# model config
CHECKPOINT_DIR=$HOME/vlm-recipes/hf_models/idefics2-8b
CHECKPOINT_SAVE_DIR=$HOME/vlm-recipes/checkpoints/idefics2-8b_hf

mkdir -p ${CHECKPOINT_SAVE_DIR}

# job name
JOB_NAME="idefics2-8b-gcp-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# for debug
export TORCH_USE_CUDA_DSA=1
# distributed args
DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPU_PER_NODE --nnodes ${NUM_NODES} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

# run
torchrun $DISTRIBUTED_ARGS examples/finetuning.py \
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
        --adam-eps 1e-8 \
        --save-interval 50000 \
        --eval-interval 100 \
        --eval-iters 10 \
        --vocab-size 128260 \
        --vlm-text-hidden-size 4096 \
        --vlm-text-intermediate-size 14336 \
        --vlm-text-num-attention-heads 32 \
        --vlm-text-num-hidden-layers 32 \
        --vlm-text-num-key-value-heads 8 \
        --vlm-text-rope-theta 10000.0 \
        --rms-norm-eps 1e-05 \
        --vlm-text-model-type "llama" \
        --bf16 \
        --mixed-precision \
        --instruction-tuning \
        --instruction-tuning-type "LLaVA_PreTrain" \
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
        --vlm-vision-model-type "idefics2" \
        --vlm-vision-hidden-size 1152 \
        --vlm-vision-intermediate-size 4304 \
        --vlm-vision-num-attention-heads 16 \
        --vlm-vision-num-hidden-layers 27  \
        --vlm-vision-image-size 980 \
        --vlm-vision-patch-size 14 \
        --vlm-perceiver-model-type "idefics2" \
        --vlm-perceiver-hidden-act "silu" \
        --vlm-perceiver-resampler-n-latents 64 \
        --vlm-perceiver-resampler-depth 3 \
        --vlm-perceiver-resampler-n-heads 16 \
        --vlm-perceiver-resampler-head-dim 96 \
        --vlm-perceiver-num-key-value-heads 4 \
        --vlm-perceiver-attention-dropout 0.0 \
        --use-freeze \
        --freeze-vlm-vision-model \
        --no-save-optimizer-state \
        --fsdp-use-orig-param \
        --wandb-entity $WANDB_ENTITY \
        --wandb-project $WANDB_PROJECT \
        --wandb-name $JOB_NAME
