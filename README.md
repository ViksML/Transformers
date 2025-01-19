# GPT Language Model Training

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A PyTorch implementation of GPT (Generative Pre-trained Transformer) model training with optimizations for CUDA, MPS, and CPU devices.

## Project Structure

- `train.py` - Main training script that orchestrates the model training process
- `models/`
  - `gpt.py` - Implementation of the GPT model architecture
- `utils/`
  - `training.py` - Training utilities and helper functions
- `config/`
  - `model_config.py` - Configuration settings for model architecture and training
- `checkpoints/` - Directory for saving model checkpoints during training

## Features

- Automatic device selection (CUDA/MPS/CPU)
- Optimized configurations per device
- Gradient accumulation
- Cosine learning rate scheduling
- Checkpoint saving
- Progress monitoring

## Training Parameters

- Max steps: 100,000
- Logging interval: 50 steps
- Target loss: 0.099999
- Checkpoints saved in `checkpoints/` directory

## Device-specific Configurations

### GPU (CUDA)
- 8 layers, 512 embedding dim
- Batch size: 8
- Sequence length: 512

### MPS (Apple Silicon)
- 6 layers, 384 embedding dim
- Batch size: 4
- Sequence length: 128

### CPU
- 6 layers, 384 embedding dim
- Batch size: 4
- Sequence length: 256

## Training Logs

### Latest Training Run
Device: CUDA (NVIDIA GPU)

Model Architecture:
- Layers: 8
- Heads: 8
- Embedding Dim: 512
- Block Size: 512
- Vocab Size: 50,304

Training Parameters:
- Batch Size: 8
- Sequence Length: 512
- Gradient Accumulation Steps: 4
- Effective Batch Size: 32
- Learning Rate: 0.0003

Model Statistics:
- Total Parameters: 51,237,888
- Trainable Parameters: 51,237,888
- Model Size: 203.46 MB

Training Progress:
- Dataset: 338,025 tokens
- Epochs: 1 epoch = 82 batches
- Optimizer: Fused AdamW
- Target Loss: 0.099999
- Achieved Target: Yes, at step 1,832
- Final Best Loss: 0.099628

Key Loss Milestones:
| Step   | Loss    | Learning Rate |
|--------|---------|---------------|
| 0      | 10.9322 | 3.00e-07     |
| 500    | 4.1621  | 1.50e-04     |
| 1000   | 2.0675  | 3.00e-04     |
| 1500   | 0.3920  | 3.00e-04     |
| 1832   | 0.0996  | 3.00e-04     |

Training completed successfully after 1,833 steps, achieving the target loss threshold.
