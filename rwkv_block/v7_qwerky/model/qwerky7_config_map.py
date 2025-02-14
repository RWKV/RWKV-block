from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

from ..block.qwerky7_block_config_map import Qwerky7BlockConfigMap

@dataclass
class Qwerky7ConfigMap(Qwerky7BlockConfigMap):
    # This is the world tokenizer size
    vocab_size: int = 152064 
    init_state_wkv: bool = False
    forward_chunk_size: int = 4096,

    # Default qwen tokenizer padding_idx
    padding_idx: int = 151643

    # ---
    # Initializer, with excess arg ignore
    # ---
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        vocab_size: int = 152064,
        init_state_wkv: bool = False,
        forward_chunk_size: Optional[int] = 4096,
        padding_idx: int = 151643,
        **kwargs
    ) -> None:
        self.vocab_size = vocab_size
        self.init_state_wkv = init_state_wkv
        self.forward_chunk_size = forward_chunk_size
        self.padding_idx = padding_idx
        super().__init__(num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, **kwargs)
        
    @staticmethod
    def normalize(config_map: any) -> 'Qwerky7ConfigMap':
        '''
        Converts either maps, objs or configmaps
        '''
        if isinstance(config_map, Qwerky7ConfigMap):
            return config_map
        
        if isinstance(config_map, dict):
            return Qwerky7ConfigMap(**config_map)

        if hasattr(config_map, '__dict__'):
            return Qwerky7ConfigMap(**config_map.__dict__)
        
        raise ValueError(f"Unsupported config_map type: {type(config_map)}")

    @staticmethod
    def from_model_state_dict(state_dict: dict, **kwargs) -> 'Qwerky7ConfigMap':
        '''
        Converts the state dict to the config map
        '''

        # Iterate and count the layers
        num_hidden_layers = 0
        for key in state_dict.keys():
            if key.startswith('model.layers.'):
                idx = key.split('.')[2]
                num_hidden_layers = max(num_hidden_layers, int(idx)+1)

        # Enable wkv_state
        if 'init_state.0.wkv' in state_dict:
            kwargs['init_state_wkv'] = True
        
        # Initialize the config map, with the configured values
        return Qwerky7ConfigMap(
            num_hidden_layers=num_hidden_layers,
            hidden_size=state_dict['model.embed_tokens.weight'].shape[1],
            vocab_size=state_dict['model.embed_tokens.weight'].shape[0],
            
            head_size = state_dict['model.layers.0.self_attn.r_k'].shape[1],

            hidden_size_att=state_dict['model.layers.0.self_attn.k_proj.weight'].shape[0],
            hidden_size_ffn=state_dict['model.layers.0.mlp.up_proj.weight'].shape[0],

            v_first_embedding = 'model.layers.0.self_attn.v0' in state_dict,

            **kwargs
        )
        