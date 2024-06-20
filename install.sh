#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -o outputs/install/$JOB_ID.log
#$ -e outputs/install/$JOB_ID.log
#$ -p -5

set -e

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

# pip version up
pip install --upgrade pip

# pip install requirements
pip install -r requirements.txt

# huggingface transformers (require 4.42.0 or later)
pip install git+https://github.com/huggingface/transformers.git@main

# distirbuted training requirements
pip install mpi4py

# huggingface requirements
pip install huggingface_hub

# install flash-atten
pip install ninja packaging wheel
pip install flash-attn==2.3.6 --no-build-isolation
