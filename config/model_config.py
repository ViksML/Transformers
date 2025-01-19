from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

def get_training_config(device):
    if device == 'cuda':  # Colab GPU configuration
        block_size = 512  # Reduced from 1024 for Colab memory
        return {
            'batch_size': 8,            # Reduced batch size for Colab GPU
            'seq_length': block_size,
            'max_lr': 3e-4,             # Slightly reduced learning rate
            'min_lr': 1e-5,
            'warmup_steps': 1000,
            'max_steps': 100000,
            'gradient_accumulation_steps': 4,  # Increased for effective batch size
            'log_interval': 50,
            'model_config': GPTConfig(
                block_size=block_size,
                n_layer=8,              # Reduced from 12
                n_head=8,              # Reduced from 12
                n_embd=512,            # Reduced from 768
                dropout=0.1
            )
        }
    elif device == 'mps':
        return {
            'batch_size': 4,
            'seq_length': 128,
            'max_lr': 5e-4,
            'min_lr': 1e-5,
            'warmup_steps': 1000,
            'max_steps': 100000,
            'gradient_accumulation_steps': 8,
            'log_interval': 50,
            'model_config': GPTConfig(
                block_size=256,
                n_layer=6,
                n_head=8,
                n_embd=384,
                dropout=0.1
            )
        }
    else:  # CPU configuration
        return {
            'batch_size': 4,
            'seq_length': 256,
            'max_lr': 1e-4,
            'min_lr': 1e-5,
            'warmup_steps': 1000,
            'max_steps': 100000,
            'gradient_accumulation_steps': 8,
            'log_interval': 50,
            'model_config': GPTConfig(
                block_size=256,
                n_layer=6,
                n_head=6,
                n_embd=384,
                dropout=0.1
            )
        }