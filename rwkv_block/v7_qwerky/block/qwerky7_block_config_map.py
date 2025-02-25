import torch
from torch import nn
from typing import Union, Tuple
from dataclasses import dataclass

from ...v7_goose.block.rwkv7_block_config_map import RWKV7BlockConfigMap

@dataclass
class Qwerky7BlockConfigMap(RWKV7BlockConfigMap):

    # RMS Norm eps
    rms_norm_eps: float = 1e-6
    # Qwerky headsize defaults to 128 (Not 64)
    head_size: int = 128

    # # Attention QKV bias
    # attention_bias: bool = True
    # # Attention output bias
    # attention_output_bias: bool = False

    # Use embedding for v_first
    v_first_with_embedding: bool = True

    def __init__(
        self, 
        num_hidden_layers: int,
        hidden_size: int,
        rms_norm_eps: float = 1e-6,
        v_first_with_embedding: bool = True,
        # attention_bias: bool = True,
        # attention_output_bias: bool = False,
        head_size: int = 128,
        **kargs
    ):
        '''
        Config with RMS Norm eps
        And alias for hidden_size_mlp
        '''
        self.rms_norm_eps = rms_norm_eps
        self.v_first_with_embedding = v_first_with_embedding
        # self.attention_bias = attention_bias
        # self.attention_output_bias = attention_output_bias
        super().__init__(num_hidden_layers, hidden_size, head_size=head_size, **kargs)

    
    def get_hidden_size_mlp(self) -> int:
        '''
        Intermidiate size of the MLP,
        Alias for get_hidden_size_ffn
        '''
        return self.get_hidden_size_ffn()

    # ---
    # Duplicator & Normalizer
    # ---

    def new_block_config_map(self, **kwargs) -> 'Qwerky7BlockConfigMap':
        '''
        Returns a new config map with updated values
        '''

        new_dict = {}
        for key in Qwerky7BlockConfigMap.__dataclass_fields__:
            if key in self.__dict__:
                new_dict[key] = self.__dict__[key]
        new_dict.update(kwargs)

        return Qwerky7BlockConfigMap(**new_dict)

    @staticmethod
    def normalize(config_map: any) -> 'Qwerky7BlockConfigMap':
        '''
        Converts either maps, objs or Qwerky7BlockConfigMap
        '''
        if isinstance(config_map, Qwerky7BlockConfigMap):
            return config_map
        
        dict_obj = None
        if isinstance(config_map, dict):
            dict_obj = config_map
        elif hasattr(config_map, '__dict__'):
            dict_obj = config_map.__dict__
        
        if dict_obj is not None:
            # Filter for only valeus in Qwerky7BlockConfigMap
            new_dict = {}
            for key, value in dict_obj.items():
                if key in Qwerky7BlockConfigMap.__dataclass_fields__:
                    new_dict[key] = value
            return Qwerky7BlockConfigMap(**new_dict)

        raise ValueError(f"Unsupported config_map type: {type(config_map)}")
