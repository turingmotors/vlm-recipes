#!/bin/bash

# Exit the script if any command fails
set -e

# Activate the Python virtual environment
source .env/bin/activate

# Define the paths to the base Hugging Face checkpoint, PyTorch checkpoint, and output directory for Hugging Face format
BASE_MODEL_CHECKPOINT=/path/to/base-huggingface-checkpoint/idefics2-8b
CHECK_POINT_PATH=/path/to/pytorch_checkpoint/iter_*******/model.pt
HF_OUTPUT_PATH=/path/to/huggingface-checkpoint/trained_idefics2_hf

# Create the output directory for Hugging Face checkpoint if it does not already exist
mkdir -p $HF_OUTPUT_PATH

# Log the start of the checkpoint conversion process
echo "--------------------"
echo "Start converting ${CHECK_POINT_PATH} to ${HF_OUTPUT_PATH}"
echo "--------------------"

# Run the Python script that converts the PyTorch checkpoint to Hugging Face format
python tools/checkpoint-convert/convert_ckpt.py \
    --base-model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $HF_OUTPUT_PATH

# Log the start of the rsync process to copy necessary files from the base model to the output directory
echo "--------------------"
echo "Start rsync from ${BASE_MODEL_CHECKPOINT} to ${HF_OUTPUT_PATH}"
echo "--------------------"

# Use rsync to synchronize various configuration files from the base model directory to the Hugging Face output directory
rsync -avhP ${BASE_MODEL_CHECKPOINT}/config.json ${HF_OUTPUT_PATH}/
rsync -avhP ${BASE_MODEL_CHECKPOINT}/generation_config.json ${HF_OUTPUT_PATH}/
rsync -avhP ${BASE_MODEL_CHECKPOINT}/preprocessor_config.json ${HF_OUTPUT_PATH}/
rsync -avhP ${BASE_MODEL_CHECKPOINT}/processor_config.json ${HF_OUTPUT_PATH}/
rsync -avhP ${BASE_MODEL_CHECKPOINT}/special_tokens_map.json ${HF_OUTPUT_PATH}/
rsync -avhP ${BASE_MODEL_CHECKPOINT}/tokenizer_config.json ${HF_OUTPUT_PATH}/
rsync -avhP ${BASE_MODEL_CHECKPOINT}/tokenizer.json ${HF_OUTPUT_PATH}/

# Log that the script has completed successfully
echo "--------------------"
echo "Conversion and rsync completed successfully"
echo "--------------------"
