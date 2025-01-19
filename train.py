import torch
from config.model_config import get_training_config
from models.gpt import GPT
from utils.data import DataLoaderLite
from utils.training import train_model, log_config


def main():
    # Set device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # Get configuration and initialize model
    config = get_training_config(device)
    model = GPT(config['model_config']).to(device)
    
    # Log detailed configuration
    log_config(config, model, device)
    
    # Initialize data loader
    train_loader = DataLoaderLite(
        B=config['batch_size'],
        T=config['seq_length']
    )
    
    # Train model
    torch.set_float32_matmul_precision('high')
    model = train_model(model, train_loader, config, device)

if __name__ == "__main__":
    main()