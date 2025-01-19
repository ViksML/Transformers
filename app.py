import gradio as gr
import torch
from models.gpt import GPTConfig, GPT
import tiktoken

# Add gradio to requirements
def update_requirements():
    return "gradio>=4.0.0"

# Load the model and tokenizer
def load_model():
    # Load model configuration
    config = GPTConfig(
        n_layer=8,
        n_head=8,
        n_embd=512,
        block_size=512,
        vocab_size=50304,
        bias=False
    )
    
    model = GPT(config)
    
    # Load checkpoint
    checkpoint = torch.load('checkpoints/transformer_model.pt', map_location='cpu')
    # Extract just the model state dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def generate_text(
    prompt, 
    max_tokens=100, 
    temperature=0.8, 
    top_k=40
):
    # Encode the prompt
    enc = tiktoken.get_encoding("gpt2")
    encoded = enc.encode(prompt)
    
    # Convert to tensor
    x = torch.tensor(encoded, dtype=torch.long)[None,...]
    
    # Generate
    with torch.no_grad():
        y = model.generate(
            x, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )[0]
    
    # Decode and return
    output = enc.decode(y.tolist())
    return output

# Load model globally
model = load_model()

# Example prompts
examples = [
    ["The quick brown fox", 50, 0.8, 40],
    ["Once upon a time", 100, 0.9, 50],
    ["The meaning of life is", 150, 0.7, 30],
    ["In the distant future", 200, 0.8, 40],
]

# Create Gradio interface with submit button
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=1, maximum=500, value=100, step=1, label="Max Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top K"),
    ],
    outputs=gr.Textbox(label="Generated Text", interactive=False),
    title="GPT Text Generation",
    description="""This is a GPT-style text generation model. You can adjust:
    - Max Tokens: Controls the length of generated text
    - Temperature: Higher values (>1.0) make the output more random, lower values make it more focused
    - Top K: Controls diversity by limiting the tokens considered for next-word prediction""",
    examples=examples,
    submit_btn="Generate Text",  # Add submit button
    clear_btn="Clear",  # Add clear button
    theme=gr.themes.Soft()  # Optional: adds a nice theme
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True enables sharing via gradio.live 