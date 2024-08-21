<div align="center">

vlm-recipes
===========================
<h4>User-friendly tool for seamless continual pre-training and visual instruction tuning of Vision-Language Models</h4>

<img src="images/vlm-recipes-logo.webp" alt="vlm-recipes" width="300px">

<div align="left">

vlm-recipes is a tool designed to make the training of Vision-Language Models (VLMs) easy and efficient. With an intuitive interface and flexible configuration options, researchers and developers can effortlessly manage training on any VLM architecture or dataset. The tool supports distributed training on large GPU clusters using PyTorch FullyShardedDataParallel (FSDP) as its backend and offers extensive customization, enabling users to leverage cutting-edge techniques with ease.

What sets vlm-recipes apart is its seamless integration with Hugging Face Transformers, allowing you to continue training or perform fine-tuning on VLMs with minimal changes. This means there’s no need to convert Hugging Face Transformers checkpoints or deal with complex workflows—just focus on refining your model.

| Feature                         | [vlm-recipes](#vlm-recipes) | [llm-recipes](https://github.com/okoge-kaz/llm-recipes) |
|---------------------------------|-------------|---------------|
| **VLM Support**                 | ✅          | ❌            |
| **LLM Support**                 | ❌          | ✅            |

The currently supported VLMs are as follows:

- [Idefics2](https://arxiv.org/abs/2405.02246)
- [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

This library is experimental and under active development.
We plan to add some **breaking changes** in the future to improve the usability and performance of the library.

> To check out the companion project **llm-recipes**, click here!
> https://github.com/okoge-kaz/llm-recipes


# Table of Contents

- [vlm-recipes](#vlm-recipes)
- [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Multi-node Support](#multi-node-support)
    - [FlashAttention](#flashattention)
  - [Usage](#usage)
    - [Visual Instruction Tuning](#visual-instruction-tuning)
      - [1. **Data Preparation**](#1-data-preparation)
      - [2. **Change Dataset Class**](#2-change-dataset-class)
      - [3. **Training**](#3-training)
    - [LLM Continual Pre-Training](#llm-continual-pre-training)
  - [Checkpoint formats](#checkpoint-formats)
    - [vlm-recipes format](#vlm-recipes-format)
    - [PyTorch format to Hugging Face format](#pytorch-format-to-hugging-face-format)
  - [Inference](#inference)
  - [Projects Using vlm-recipes](#projects-using-vlm-recipes)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Installation

This package has been tested with Python 3.10 and 3.11. The recommended environment is with CUDA Toolkit 12.1.

To install the required packages, simply run:

```bash
pip install -r requirements.txt
```
> Note: The requirements.txt assumes that CUDA Toolkit 12.1 is installed on your system.

### Multi-node Support

For multi-node support, ensure you have the following dependencies installed:

```bash
module load openmpi/4.x.x

pip install mpi4py
```

### FlashAttention

For GPU-accelerated FlashAttention, follow these steps:

```bash
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
```

## Usage

### Visual Instruction Tuning

- `src/llama_recipes/utils/visual_instruct.py`: DataLoader for Visual Instruction Tuning
- `src/llama_recipes/datasets/llava_pretrain.py`: LLaVA format dataset

#### 1. **Data Preparation**

If you use LLaVA formatted datasets (e.g., [LLaVA-PreTrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), [LLaVA-Instruct](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)), please prepare the dataset in the following format:

```jsonl
{
  "image": "/image/path/to/image_1.png",
  "conversations": [
      {
        "from": "human",
        "value": "<image>\nCould you explain what is happening in this image?"
      },
      {
        "from": "gpt",
        "value": "This is a picture of a cat sitting on a chair."
      }
  ]
}
```

#### 2. **Change Dataset Class**

If you want to train with your own dataset, please change the dataset class in `src/llama_recipes/datasets/llava_pretrain.py` or make your own dataset class.

#### 3. **Training**

We provide example scripts for visual instruction tuning for [Idefics2](https://arxiv.org/abs/2405.02246) in `scripts/tsubame/llava_pretrain/idefics2-8b.sh` and [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) in `scripts/tsubame/llava_pretrain/llava-next-7b.sh`. You can modify the script to suit your needs.

### LLM Continual Pre-Training

This section is currently under development. 🚧
We will release this section with more information soon.

## Checkpoint formats

### vlm-recipes format
vlm-recipes supports PyTorch checkpoint format: The PyTorch format is a simple checkpoint format. The example of the PyTorch format is as follows:

```bash
model.pt  optimizer.pt  rng.pt  sampler.pt  scheduler.pt
```

### PyTorch format to Hugging Face format

You can convert the PyTorch format to the Hugging Face format using the following command:

```bash
ITERATION=1000
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

BASE_MODEL_CHECKPOINT=/path/to/huggingface-checkpoint/idefics2-8b
CHECK_POINT_PATH=/path/to/train/checkpoint/${FORMATTED_ITERATION}/model.pt
HF_OUTPUT_PATH=/path/to/converted/checkpoint/${FORMATTED_ITERATION}

mkdir -p $HF_OUTPUT_PATH

python tools/checkpoint-convert/convert_ckpt.py \
    --model $BASE_MODEL_CHECKPOINT \
    --ckpt $CHECK_POINT_PATH \
    --out $HF_OUTPUT_PATH
```

(The complete conversion script is located at `tools/checkpoint-convert/scripts/convert.sh`)

## Inference

After checkpoint conversion, you can use the Hugging Face Transformers library to load the converted checkpoint and perform inference.

The following is an example of how to do inference using the converted checkpoint (huggingface format):

```bash
python tools/inference/inference.py \
  --model-path /path/to/huggingface-checkpoint/idefics2 \
  --processor-path /path/to/huggingface-processor/idefics2 \
  --image-path images/drive_situation_image.jpg \
  --prompt "In the situation in the image, is it permissible to start the car when the light turns green?"
```

(The complete inference script is located at `tools/inference/inference.sh`)

## Projects Using vlm-recipes

Below are some of the projects where we have directly used vlm-recipes:

- [Turing(company)](https://tur.ing/en)'s [GENIAC](https://www.meti.go.jp/english/policy/mono_info_service/geniac/index.html) project (VLM training)

## Citation

```bibtex
@software{
author = {Kazuki Fujii and Daiki Shiono and Yu Yamaguchi and Taishi Nakamura and Rio Yokota},
month = {Aug},
title = {{vlm-recipes}},
url = {https://github.com/turingmotors/vlm-recipes},
version = {0.1.0},
year = {2024}
}
```

## Acknowledgement

This repository is based on results obtained from a project, JPNP20017, subsidized by the New Energy and Industrial Technology Development Organization (NEDO).
