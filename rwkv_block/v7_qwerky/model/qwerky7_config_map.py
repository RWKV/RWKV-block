from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

from ..block.qwerky7_block_config_map import Qwerky7BlockConfigMap
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

@dataclass
class Qwerky7ConfigMap(Qwerky7BlockConfigMap):
    # This is the world tokenizer size
    vocab_size: int = 152064
    init_state_wkv: bool = False
    forward_chunk_size: int = 4096,

    # Default qwen tokenizer padding_idx
    padding_idx: int = 151643

    # ---
    # Hybrid models, consist of qwen layers
    # ---

    # Hybrid models, consist of qwen layers
    # on top and/or bottom of qwerky layers, golfinch style
    num_suffix_hybrid_layers: int = 0
    num_prefix_hybrid_layers: int = 0

    # Transformer layers, and rope scaling related config
    hybrid_num_attention_heads: int = 0
    hybrid_num_key_value_heads: int = 0
    hybrid_attention_dropout: float = 0.0
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # Rotary positional embeddings
    use_rotary_pos_emb: bool = True

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
        # ---
        num_suffix_hybrid_layers: int = 0,
        num_prefix_hybrid_layers: int = 0,
        hybrid_num_attention_heads: int = 1,
        hybrid_num_key_value_heads: int = 1,
        rope_theta: float = 1000000.0,
        hybrid_attention_dropout: float = 0.0,
        max_position_embeddings: int = 32768,
        # ---
        use_rotary_pos_emb: bool = True,
        # ---
        **kwargs
    ) -> None:
        self.vocab_size = vocab_size
        self.init_state_wkv = init_state_wkv
        self.forward_chunk_size = forward_chunk_size
        self.padding_idx = padding_idx
        # ---
        self.max_position_embeddings = max_position_embeddings
        self.use_rotary_pos_emb = use_rotary_pos_emb
        # ---
        self.num_suffix_hybrid_layers = num_suffix_hybrid_layers
        self.num_prefix_hybrid_layers = num_prefix_hybrid_layers
        self.hybrid_num_attention_heads = hybrid_num_attention_heads
        self.hybrid_num_key_value_heads = hybrid_num_key_value_heads
        self.hybrid_attention_dropout = hybrid_attention_dropout
        # ---
        self.rope_theta = rope_theta
        # ---
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

    def hybrid_layer_config(self) -> dict:
        '''
        Returns the hybrid config for qwen layers
        '''
        return Qwen2Config(
            vocab_size=self.vocab_size,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.hidden_size_ffn,
            rms_norm_eps=self.rms_norm_eps,
            # ---
            num_attention_heads = self.hybrid_num_attention_heads,
            num_key_value_heads = self.hybrid_num_key_value_heads,
            attention_dropout = self.hybrid_attention_dropout,
            # ---
            rope_theta = self.rope_theta,
            max_position_embeddings = self.max_position_embeddings,
        )

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

        # From the last layer, count the number of layers without r_k
        # which implies they are qwen layers]
        num_suffix_hybrid_layers = 0
        for i in range(num_hidden_layers-1, -1, -1):
            if f'model.layers.{i}.self_attn.r_k' not in state_dict:
                num_suffix_hybrid_layers += 1
            else:
                break

        # Get the number of prefix hybrid layers
        num_prefix_hybrid_layers = 0
        for i in range(0, num_hidden_layers):
            if f'model.layers.{i}.self_attn.r_k' not in state_dict:
                num_prefix_hybrid_layers += 1
            else:
                break
        num_hybrid_layers = num_suffix_hybrid_layers + num_prefix_hybrid_layers

        joint_state_args = { **state_dict, **kwargs }
        if num_hybrid_layers > 0:
            if 'hybrid_num_attention_heads' in joint_state_args:
                num_attention_heads = joint_state_args['hybrid_num_attention_heads']
            else:
                raise ValueError("hybrid model : hybrid_num_attention_heads not found in state_dict or kwargs, unable to guess value")

            if 'hybrid_num_key_value_heads' in joint_state_args:
                num_key_value_heads = joint_state_args['hybrid_num_key_value_heads']
            else:
                raise ValueError("hybrid model : hybrid_num_key_value_heads not found in state_dict or kwargs, unable to guess value")

        # Enable wkv_state
        if 'init_state.0.wkv' in state_dict:
            kwargs['init_state_wkv'] = True

        # Initialize the config map, with the configured values
        return Qwerky7ConfigMap(**{
            **{
                "num_hidden_layers": num_hidden_layers,
                "hidden_size": state_dict['model.embed_tokens.weight'].shape[1],
                "vocab_size":  state_dict['model.embed_tokens.weight'].shape[0],

                "head_size":   state_dict[f'model.layers.{num_prefix_hybrid_layers}.self_attn.r_k'].shape[1],

                "hidden_size_att": state_dict[f'model.layers.{num_prefix_hybrid_layers}.self_attn.k_proj.weight'].shape[0],
                "hidden_size_ffn": state_dict[f'model.layers.{num_prefix_hybrid_layers}.mlp.up_proj.weight'].shape[0],

                "v_first_with_embedding": f'model.layers.{num_prefix_hybrid_layers}.self_attn.v0' in state_dict,

                "num_suffix_hybrid_layers": num_suffix_hybrid_layers,
                "num_prefix_hybrid_layers": num_prefix_hybrid_layers,
            },
            **kwargs
        })

    def num_qwerky_layers(self) -> int:
        """
        Returns the number of qwerky layers in the model
        """
        return self.num_hidden_layers - self.num_suffix_hybrid_layers - self.num_prefix_hybrid_layers

    def num_hybrid_layers(self) -> int:
        """
        Returns the total number of hybrid layers in the model
        """
        return self.num_suffix_hybrid_layers + self.num_prefix_hybrid_layers
