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

# Table of Contents

- [Installation](#installation)
  - [Multi-node Support](#multi-node-support)
  - [FlashAttention](#flashattention)
- [Usage](#usage)
  - [MoE Instruction Tuning](#moe-instruction-tuning)
  - [MoE Continual Pre-Training](#moe-continual-pre-training)
- [Checkpoint formats](#checkpoint-formats)
  - [DeepSpeed format to Hugging Face format](#deepspeed-format-to-hugging-face-format)
- [Inference](#inference)
- [Training Speed and Scalability](#training-speed-and-scalability)
- [Projects Using moe-recipes](#projects-using-moe-recipes)
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

#### 1. **Data Preparation**

#### 2. **Change Dataset Class**


#### 3. **Indexing**


#### 4. **Training**


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
