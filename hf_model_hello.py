import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a text generation model from a specified directory.')
    parser.add_argument('--hf_path', type=str, required=True, help='Path to the Hugging Face model directory.')
    parser.add_argument('--run_device', type=str, default='cuda', help='Device to run the model on (e.g., cuda:0 or cpu).')
    parser.add_argument('--tmix_backend', type=str, default='auto', help='Backend for time-mix implementation.')
    return parser.parse_args()

def load_model_and_tokenizer(hf_path):
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    return config, tokenizer

def initialize_model(hf_path, run_device, tmix_backend='auto'):
    model = AutoModelForCausalLM.from_pretrained(
        hf_path, 
        trust_remote_code=True, 
        tmix_backend=tmix_backend, 
        device=run_device
    )
    model.to(run_device)
    return model

def generate_text(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parse_arguments()
    
    # Load model and tokenizer
    config, tokenizer = load_model_and_tokenizer(args.hf_path)
    model = initialize_model(args.hf_path, args.run_device, args.tmix_backend)
    
    print("Model and tokenizer loaded successfully")
    print("Running on device:", model.device)
    
    # Text generation
    prompts = [
        "HELLO WORLD",
        "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
    ]
    
    for prompt in prompts:
        print("\n---------------------------------")
        print(f"Prompt: {prompt}")
        generated_text = generate_text(model, tokenizer, prompt, model.device)
        print("Generated text:", generated_text[len(prompt):])
        print("---------------------------------")

if __name__ == "__main__":
    main()