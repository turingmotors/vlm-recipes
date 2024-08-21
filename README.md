<div align="center">

vlm-recipes
===========================
<h4>User-friendly tool for seamless continual pre-training and visual instruction tuning of Vision and Language Models</h4>

<img src="images/vlm-recipes-logo.webp" alt="vlm-recipes" width="300px">

<div align="left">

vlm-recipes is a tool designed to make the training of Vision and Language Models (VLMs) easy and efficient. With an intuitive interface and flexible configuration options, researchers and developers can effortlessly manage training on any VLM architecture or dataset. The tool supports distributed training on large GPU clusters using PyTorch FullyShardedDataParallel (FSDP) as its backend and offers extensive customization, enabling users to leverage cutting-edge techniques with ease.

What sets vlm-recipes apart is its seamless integration with Hugging Face Transformers, allowing you to continue training or perform fine-tuning on VLM models with minimal changes. This means there’s no need to convert checkpoints or deal with complex workflows—just focus on refining your model.

| Feature                         | vlm-recipes | llm-recipes |
|---------------------------------|-------------|---------------|
| **VLM Support**                 | ✅          | ❌            |
| **LLM Support**                 | ❌          | ✅            |

This library is experimental and under active development.
We plan to add some **breaking changes** in the future to improve the usability and performance of the library.

# Table of Contents

- [Installation](#installation)
  - [Multi-node Support](#multi-node-support)
  - [FlashAttention](#flashattention)
- [Usage](#usage)
  - [Visual Instruction Tuning](#visual-instruction-tuning)
    - [Data Preparation](#data-preparation)
    - [Change Dataset Class](#change-dataset-class)
    - [Training](#training)
- [Checkpoint formats](#checkpoint-formats)
  - [PyTorch format to Hugging Face format](#pytorch-format-to-hugging-face-format)
- [Inference](#inference)
- [Citation](#citation)

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

If you use LLaVA_PreTrain dataset, please prepare the dataset in the following format:

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

## Checkpoint formats

### PyTorch format to Hugging Face format

## Citation

```bibtex
@software{f
author = {Kazuki Fujii and Daiki Shiono and Yu Yamaguchi and Taishi Nakamura and Rio Yokota},
month = {Aug},
title = {{vlm-recipes}},
url = {https://github.com/turingmotors/vlm-recipes},
version = {0.1.0},
year = {2024}
}
```
