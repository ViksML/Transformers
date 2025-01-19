import os
import time
import torch
import math

class CosineWarmupScheduler:
    def __init__(self, max_lr, min_lr, warmup_steps, max_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

# Add model saving function
def save_checkpoint(model, optimizer, config, step, loss, best_loss, save_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': step,
        'loss': loss,
        'best_loss': best_loss
    }
    torch.save(checkpoint, save_path)
    #print(f"Checkpoint saved: {save_path}")

# Add load checkpoint function
def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    model = GPT(config['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=config['max_lr'],
        device_type=device
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, config, checkpoint['step'], checkpoint['loss'], checkpoint['best_loss']

# Add detailed logging function
def log_config(config, model, device):
    print("\n=== Training Configuration ===")
    print(f"Device: {device}")
    print("\nModel Architecture:")
    print(f"- Layers: {config['model_config'].n_layer}")
    print(f"- Heads: {config['model_config'].n_head}")
    print(f"- Embedding Dim: {config['model_config'].n_embd}")
    print(f"- Block Size: {config['model_config'].block_size}")
    print(f"- Vocab Size: {config['model_config'].vocab_size}")

    print("\nTraining Parameters:")
    print(f"- Batch Size: {config['batch_size']}")
    print(f"- Sequence Length: {config['seq_length']}")
    print(f"- Gradient Accumulation Steps: {config['gradient_accumulation_steps']}")
    print(f"- Effective Batch Size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"- Learning Rate: {config['max_lr']}")

    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    print("\nModel Statistics:")
    print(f"- Total Parameters: {total_params:,}")
    print(f"- Trainable Parameters: {trainable_params:,}")
    print(f"- Model Size: {total_size/1024/1024:.2f} MB")
    print("="*30 + "\n")

def train_model(model, train_loader, config, device):
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=config['max_lr'],
        device_type=device
    )

    scheduler = CosineWarmupScheduler(
        max_lr=config['max_lr'],
        min_lr=config['min_lr'],
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps']
    )

    model.train()
    total_tokens = 0
    best_loss = float('inf')

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    print("\n=== Starting Training ===")
    print(f"Training for {config['max_steps']:,} steps")
    print(f"Logging every {config['log_interval']} steps\n")

    for step in range(config['max_steps']):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0

        for micro_step in range(config['gradient_accumulation_steps']):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if device == 'mps':
                logits, loss = model(x, y)
            else:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)

            loss = loss / config['gradient_accumulation_steps']
            loss.backward()
            accumulated_loss += float(loss.detach().cpu().item())

            del logits, loss
            if device == 'mps':
                torch.mps.empty_cache()

        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Learning rate update
        lr = scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        # Modified logging frequency
        if step % config['log_interval'] == 0:
            elapsed = time.time() - t0
            print(f"Step {step:6d}/{config['max_steps']:6d} | "
                  f"Loss: {accumulated_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Best: {best_loss:.4f} | "
                  f"Time: {elapsed:.2f}s")

        # Track best loss
        if accumulated_loss < best_loss:
            best_loss = accumulated_loss

        if accumulated_loss < 0.099999:
            print(f"\n🎉 Target loss achieved at step {step:,}!")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                loss=accumulated_loss,
                best_loss=best_loss,
                save_path=f'checkpoints/target_achieved_model.pt'
            )
            break

    print("\n=== Training Complete ===")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Total Steps: {step + 1:,}")
    return model