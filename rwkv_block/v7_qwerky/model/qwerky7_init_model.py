#!/usr/bin/env python3
"""
Qwerky7 Model Initialization Script

This script initializes a new Qwerky7 model with specified configuration parameters
and saves it to a .pth file. It provides command-line interface for easy model creation.

Example usage:
    python -m rwkv_block.v7_qwerky.model.qwerky7_init_model \
        --num-hidden-layers 32 \
        --hidden-size 4096 \
        --output model.pth
"""

import os, sys
if __name__ == '__main__':
    # Add project root to path
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if proj_path not in sys.path:
        sys.path.insert(0, proj_path)

import argparse
import torch
from typing import Optional

from rwkv_block.v7_qwerky.model.qwerky7_config_map import Qwerky7ConfigMap
from rwkv_block.v7_qwerky.model.qwerky7_model import Qwerky7Model
from rwkv_block.v7_qwerky.model.qwerky7_causal_lm import Qwerky7CausalLM

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Initialize a new Qwerky7 model and save it to a .pth file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required parameters
    parser.add_argument('--num-hidden-layers', '--num_hidden_layers', type=int, required=True,
                      help='Number of hidden layers in the model')
    parser.add_argument('--hidden-size', '--hidden_size', type=int, required=True,
                      help='Size of hidden layers')
    parser.add_argument('--output', type=str, required=True,
                      help='Output path for the model .pth file')
    
    # Optional parameters
    parser.add_argument('--vocab-size', '--vocab_size', type=int, default=152064,
                      help='Size of the vocabulary')
    parser.add_argument('--init-state-wkv', '--init_state_wkv', type=str2bool, default=False,
                      help='Enable WKV state initialization')
    parser.add_argument('--forward-chunk-size', '--forward_chunk_size', type=int, default=4096,
                      help='Size of forward chunks for processing')
    parser.add_argument('--padding-idx', '--padding_idx', type=int, default=151643,
                      help='Padding token index')
    
    # Hybrid model parameters
    parser.add_argument('--num-suffix-hybrid-layers', '--num_suffix_hybrid_layers', type=int, default=0,
                      help='Number of hybrid layers at the end')
    parser.add_argument('--num-prefix-hybrid-layers', '--num_prefix_hybrid_layers', type=int, default=0,
                      help='Number of hybrid layers at the start')
    parser.add_argument('--hybrid-num-attention-heads', '--hybrid_num_attention_heads', type=int, default=1,
                      help='Number of attention heads in hybrid layers')
    parser.add_argument('--hybrid-num-key-value-heads', '--hybrid_num_key_value_heads', type=int, default=1,
                      help='Number of key/value heads in hybrid layers')
    parser.add_argument('--rope-theta', '--rope_theta', type=float, default=1000000.0,
                      help='RoPE theta parameter')
    parser.add_argument('--max-position-embeddings', '--max_position_embeddings', type=int, default=32768,
                      help='Maximum position embeddings')
    
    # Block specific parameters
    parser.add_argument('--head-size', '--head_size', type=int, default=128,
                      help='Size of attention heads')
    parser.add_argument('--rms-norm-eps', '--rms_norm_eps', type=float, default=1e-6,
                      help='RMS normalization epsilon')
    parser.add_argument('--v-first-with-embedding', '--v_first_with_embedding', type=str2bool, default=True,
                      help='Use embedding for v_first')
    
    # Model specific parameters
    parser.add_argument('--use-rotary-pos-emb', '--use_rotary_pos_emb', type=str2bool, default=True,
                      help='Use rotary positional embeddings')
    parser.add_argument('--hybrid-attention-dropout', '--hybrid_attention_dropout', type=float, default=0.0,
                      help='Dropout rate for hybrid attention layers')
    parser.add_argument('--hidden-size-att', '--hidden_size_att', type=int, default=None,
                      help='Size of attention hidden layer (defaults to hidden_size if not set)')
    parser.add_argument('--hidden-size-ffn', '--hidden_size_ffn', type=int, default=None,
                      help='Size of FFN hidden layer (defaults to hidden_size if not set)')
    
    # RWKV7 block parameters
    parser.add_argument('--dropout-rate', '--dropout_rate', type=float, default=0.0,
                      help='Dropout rate (should only be used in training)')
    parser.add_argument('--tmix-backend', '--tmix_backend', type=str, default='auto',
                      help='Implementation backend to use')
    
    # Device and dtype
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to initialize model on (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                      choices=['float32', 'float16', 'bfloat16'],
                      help='Data type for model parameters')
    
    return parser.parse_args()

def init_model(args) -> None:
    """Initialize model with given configuration and save it."""
    print(f"Initializing Qwerky7 model with {args.num_hidden_layers} layers and hidden size {args.hidden_size}")
    
    # Create config map
    config = Qwerky7ConfigMap(
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        init_state_wkv=args.init_state_wkv,
        forward_chunk_size=args.forward_chunk_size,
        padding_idx=args.padding_idx,
        num_suffix_hybrid_layers=args.num_suffix_hybrid_layers,
        num_prefix_hybrid_layers=args.num_prefix_hybrid_layers,
        hybrid_num_attention_heads=args.hybrid_num_attention_heads,
        hybrid_num_key_value_heads=args.hybrid_num_key_value_heads,
        rope_theta=args.rope_theta,
        max_position_embeddings=args.max_position_embeddings,
        head_size=args.head_size,
        rms_norm_eps=args.rms_norm_eps,
        v_first_with_embedding=args.v_first_with_embedding,
        use_rotary_pos_emb=args.use_rotary_pos_emb,
        hybrid_attention_dropout=args.hybrid_attention_dropout,
        hidden_size_att=args.hidden_size_att,
        hidden_size_ffn=args.hidden_size_ffn,
        dropout_rate=args.dropout_rate,
        tmix_backend=args.tmix_backend,
        device=args.device,
        dtype=args.dtype
    )
    
    # Create and initialize model
    print("Creating model...")
    model = Qwerky7CausalLM(config)
    
    print("Resetting parameters to initial values...")
    model.reset_parameters()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Save the model
    print(f"Saving model to {args.output}...")
    torch.save(model.state_dict(), args.output)
    print("Model initialization complete!")

def main():
    args = parse_args()
    init_model(args)

if __name__ == '__main__':
    main()
