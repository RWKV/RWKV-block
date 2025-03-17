#!/usr/bin/env python
"""
Test script to verify the train_mask functionality with the MMLU dataset.
This script runs a small number of training steps to ensure the train_mask is working correctly.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hf_trainer.text_dataset_streamer import TextDatasetStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("Testing train_mask functionality with MMLU dataset")
    
    # Create a simplified configuration for testing
    config = {
        "packing_batch_size": 2,
        "packing_context_length": 256,
        "packing_mode": "preserve",
        "shuffle_buffer_size": 10,
        "seed": 42,
        "datasets": [
            {
                "hf_path": "cais/mmlu",
                "hf_args": {
                    "name": "auxiliary_train",
                    "split": "train",
                    "streaming": True
                },
                # Using multipart template with train mask
                "multipart_template": [
                    "The following are multiple choice questions (with answers) about {train.subject}.\n\n{train.question}\nA. {train.choices[0]}\nB. {train.choices[1]}\nC. {train.choices[2]}\nD. {train.choices[3]}\nAnswer: ",
                    "{to_letter(train.answer, case='upper')}"
                ],
                "multipart_train_mask": [0, 1],  # Only train on the answer part
                "weight": 1.0
            }
        ]
    }
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create dataset
    dataset = TextDatasetStreamer(
        dataset_configs=config["datasets"],
        tokenizer=tokenizer,
        packing_mode=config["packing_mode"],
        packing_batch_size=config["packing_batch_size"],
        packing_context_length=config["packing_context_length"],
        shuffle_buffer_size=config["shuffle_buffer_size"],
        seed=config["seed"],
        rank=0,
        world_size=1
    )
    
    # Get a batch from the dataset
    print("Getting a batch from the dataset...")
    batch = next(iter(dataset))
    
    # Print batch information
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Train mask shape: {batch['train_mask'].shape}")
    
    # Print a sample from the batch
    sample_idx = 0
    input_ids = batch["input_ids"][sample_idx]
    train_mask = batch["train_mask"][sample_idx]
    
    # Decode the tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    
    # Decode the full text
    full_text = tokenizer.decode(input_ids.tolist())
    print("\nFull sample text:")
    print(full_text)
    
    # Print tokens and their corresponding train_mask values
    print("\nSample tokens with train_mask values:")
    for i, (token, mask) in enumerate(zip(tokens, train_mask.tolist())):
        if i > 0 and tokens[i-1] == "Answer:" and token != tokenizer.eos_token:
            print(f"Token {i}: '{token}', train_mask: {mask} <-- Answer part")
        else:
            print(f"Token {i}: '{token}', train_mask: {mask}")
    
    # Verify that the train_mask is correctly applied
    # Find the position of "Answer:" in the tokens
    answer_positions = [i for i, t in enumerate(tokens) if "Answer" in t]
    if answer_positions:
        answer_pos = answer_positions[0]
        print(f"\nFound 'Answer:' at position {answer_pos}")
        
        # Check if tokens after "Answer:" have train_mask=1
        answer_part_masks = train_mask[answer_pos+1:].tolist()
        if 1 in answer_part_masks:
            print("✅ Train mask is correctly set to 1 for the answer part")
        else:
            print("❌ Train mask is not set to 1 for the answer part")
        
        # Check if tokens before "Answer:" have train_mask=0
        question_part_masks = train_mask[:answer_pos].tolist()
        if all(m == 0 for m in question_part_masks):
            print("✅ Train mask is correctly set to 0 for the question part")
        else:
            print("❌ Train mask is not set to 0 for the question part")
    else:
        print("Could not find 'Answer:' in the tokens")
    
    print("\nTest completed")

if __name__ == "__main__":
    main()
