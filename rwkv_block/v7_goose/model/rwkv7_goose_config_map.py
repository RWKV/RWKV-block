from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

from ..block.rwkv7_block_config_map import RWKV7BlockConfigMap

@dataclass
class RWKV7GooseConfigMap(RWKV7BlockConfigMap):
    # This is the world tokenizer size
    vocab_size: int = 65536 
    forward_chunk_size: int = 4096

    init_wkv_state: bool      = False # Include init WKV state within the model
    freeze_wkv_state: bool    = True  # Freeze the WKV state, require init_wkv_state=True
    freeze_full_weights: bool = False # Freeze the model training weights
    
    # ---
    # Initializer, with excess arg ignore
    # ---
    def __init__(
        self,
        vocab_size: int = 65536,
        forward_chunk_size: Optional[int] = 4096,

        init_wkv_state: bool = False,
        freeze_wkv_state: bool = True,
        freeze_full_weights: bool = False,
        **kwargs
    ) -> None:
        self.vocab_size = vocab_size
        self.forward_chunk_size = forward_chunk_size

        self.init_wkv_state = init_wkv_state
        self.freeze_wkv_state = freeze_wkv_state
        self.freeze_full_weights = freeze_full_weights
        
        super().__init__(**kwargs)
        
    @staticmethod
    def normalize(config_map: any) -> 'RWKV7GooseConfigMap':
        '''
        Converts either maps, objs or RWKV7BlockConfigMap
        '''
        if isinstance(config_map, RWKV7GooseConfigMap):
            return config_map
        
        if isinstance(config_map, dict):
            return RWKV7GooseConfigMap(**config_map)

        if hasattr(config_map, '__dict__'):
            return RWKV7GooseConfigMap(**config_map.__dict__)
        
        raise ValueError(f"Unsupported config_map type: {type(config_map)}")

    @staticmethod
    def from_model_state_dict(state_dict: dict, **kwargs) -> 'RWKV7GooseConfigMap':
        '''
        Converts the state dict to the config map
        '''

        # Iterate and count the layers
        num_hidden_layers = 0
        for key in state_dict.keys():
            if key.startswith('blocks.'):
                idx = key.split('.')[1]
                num_hidden_layers = max(num_hidden_layers, int(idx)+1)

        # Enable wkv_state
        if 'init_state.0.wkv' in state_dict:
            kwargs['init_wkv_state'] = True
        
        # Initialize the config map, with the configured values
        return RWKV7GooseConfigMap(
            num_hidden_layers=num_hidden_layers,
            hidden_size=state_dict['emb.weight'].shape[1],
            vocab_size=state_dict['emb.weight'].shape[0],
            # init_wkv_state=hasattr(state_dict, 'init_state.0.wkv'),

            # n_head=state_dict['blocks.0.att.r_k'].shape[0],
            head_size=state_dict['blocks.0.att.r_k'].shape[1],

            hidden_size_att=state_dict['blocks.0.att.key.weight'].shape[1],
            hidden_size_ffn=state_dict['blocks.0.ffn.key.weight'].shape[0],

            **kwargs
        )
        