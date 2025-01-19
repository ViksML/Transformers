# GPT Language Model Training

A PyTorch implementation of GPT (Generative Pre-trained Transformer) model training with optimizations for CUDA, MPS, and CPU devices.

## Project Structure

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
