import gradio as gr
import torch
from models.gpt import GPTConfig, GPT
import tiktoken
import os

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

# Add gradio to requirements
def update_requirements():
    return "gradio>=4.0.0"

def load_model():
    try:
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
        
        # Check if model checkpoint exists
        checkpoint_path = 'checkpoints/transformer_model.pt'
        if not os.path.exists(checkpoint_path):
            raise ModelError(f"Model checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        except Exception as e:
            raise ModelError(f"Failed to load model checkpoint: {str(e)}")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_text(
    prompt, 
    max_tokens=100, 
    temperature=0.8, 
    top_k=40
):
    try:
        # Input validation
        if not prompt or prompt.isspace():
            return "Error: Please enter a non-empty prompt"
        
        if len(prompt) > 500:  # Arbitrary limit
            return "Error: Prompt too long. Please limit to 500 characters"
            
        # Encode the prompt
        try:
            enc = tiktoken.get_encoding("gpt2")
            encoded = enc.encode(prompt)
        except Exception as e:
            return f"Error encoding prompt: {str(e)}"
        
        if len(encoded) > model.config.block_size:
            return f"Error: Prompt too long. Maximum length is {model.config.block_size} tokens"
        
        # Convert to tensor
        x = torch.tensor(encoded, dtype=torch.long)[None,...]
        
        # Generate
        try:
            with torch.no_grad():
                y = model.generate(
                    x, 
                    max_new_tokens=max_tokens,
                    temperature=max(0.1, temperature),  # Prevent division by zero
                    top_k=max(1, top_k)  # Ensure at least 1 token is considered
                )[0]
        except torch.cuda.OutOfMemoryError:
            return "Error: Out of memory. Try reducing max tokens or input length"
        except Exception as e:
            return f"Error during generation: {str(e)}"
        
        # Decode and return
        try:
            output = enc.decode(y.tolist())
            if not output:
                return "Error: No text generated"
            return output
        except Exception as e:
            return f"Error decoding output: {str(e)}"
            
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Try to load model globally
try:
    model = load_model()
except Exception as e:
    print(f"Fatal error loading model: {str(e)}")
    model = None

def check_model_loaded(fn):
    def wrapper(*args, **kwargs):
        if model is None:
            return "Error: Model not loaded. Please check server logs."
        return fn(*args, **kwargs)
    return wrapper

# Wrap generate_text with model check
generate_text = check_model_loaded(generate_text)

# Example prompts
examples = [
    ["The quick brown fox", 50, 0.8, 40],
    ["Once upon a time", 100, 0.9, 50],
    ["The meaning of life is", 150, 0.7, 30],
    ["In the distant future", 200, 0.8, 40],
]

# Create input components
input_text = gr.Textbox(label="Input Text", placeholder="Enter your prompt here...")
max_tokens_slider = gr.Slider(minimum=1, maximum=500, value=100, step=1, label="Max Tokens")
temperature_slider = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature")
top_k_slider = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top K")
output_text = gr.Textbox(label="Generated Text", interactive=False)

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GPT Text Generation")
    gr.Markdown("""This is a GPT-style text generation model. You can adjust:
    - Max Tokens: Controls the length of generated text
    - Temperature: Higher values (>1.0) make the output more random, lower values make it more focused
    - Top K: Controls diversity by limiting the tokens considered for next-word prediction""")
    
    with gr.Row():
        with gr.Column():
            input_text.render()
            max_tokens_slider.render()
            temperature_slider.render()
            top_k_slider.render()
            generate_btn = gr.Button("Generate Text")
            clear_btn = gr.Button("Clear")
        
        with gr.Column():
            output_text.render()
    
    # Add examples
    gr.Examples(
        examples=examples,
        inputs=[input_text, max_tokens_slider, temperature_slider, top_k_slider],
        outputs=output_text,
        fn=generate_text,
        cache_examples=False,
    )
    
    # Clear output when input changes
    input_text.change(lambda: "", outputs=[output_text])
    
    # Set up button actions
    generate_btn.click(
        generate_text,
        inputs=[input_text, max_tokens_slider, temperature_slider, top_k_slider],
        outputs=[output_text]
    )
    
    clear_btn.click(
        lambda: ("", 100, 0.8, 40, ""),
        outputs=[input_text, max_tokens_slider, temperature_slider, top_k_slider, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True) 