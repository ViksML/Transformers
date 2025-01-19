import torch
import tiktoken

class DataLoaderLite:
    
    def __init__(self, B, T):
        self.B = B  # batch size
        self.T = T  # sequence length

        # Load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()

        # Initialize tokenizer
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'Loaded {len(self.tokens):,} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T):,} batches')

        # Initialize position
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        # Get the next batch of tokens
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]

        # Reset position if we're at the end
        if len(buf) < B*T + 1:
            self.current_position = 0
            buf = self.tokens[:B*T + 1]

        # Create input and target tensors
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Update position
        self.current_position += B*T

        return x, y