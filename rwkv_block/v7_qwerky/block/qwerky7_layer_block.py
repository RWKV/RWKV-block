import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple

from ...v7_goose.block.rwkv7_layer_block import RWKV7LayerBlock
from ...v7_goose.block.rwkv7_block_config_map import RWKV7BlockConfigMap
from .qwerky7_time_mix import Qwerky7TimeMix
from .qwerky7_block_config_map import Qwerky7BlockConfigMap

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP
from dataclasses import dataclass

@dataclass
class Qwerky7Qwen2MLPConfig:
    '''
    Simple dataclass, to comply with Qwen2MLP config requirements
    '''
    hidden_size: int
    intermediate_size: int
    hidden_act: str ="silu"

class Qwerky7LayerBlock(torch.nn.Module):
    def __init__(
            self, 
            configMap: Union[Qwerky7BlockConfigMap, RWKV7BlockConfigMap, any]
        ):
        super().__init__()

        # The configMap to use
        configMap = Qwerky7BlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        hidden_size = configMap.hidden_size
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')
        rms_norm_eps = configMap.rms_norm_eps

        # Linear module function
        # This is used to replace the linear module, with a custom implementation
        # Might be requried to work around some known deepspeed 3 issues
        self.linear_module_function = None

        # Setup the modules
        with torch.device(device):
            self.input_layernorm = Qwen2RMSNorm(hidden_size, rms_norm_eps).to(dtype=dtype)
            self.self_attn = Qwerky7TimeMix(configMap)

            self.post_attention_layernorm = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps).to(dtype=dtype)
            self.mlp = Qwen2MLP(Qwerky7Qwen2MLPConfig(
                hidden_size = hidden_size,
                intermediate_size = configMap.get_hidden_size_ffn()
            )).to(dtype=dtype)

            # Setup droupout at block level
            dropout_rate = configMap.dropout_rate
            if dropout_rate > 0.0:            
                self.drop0 = nn.Dropout(p = dropout_rate)
                self.drop1 = nn.Dropout(p = dropout_rate)
            else:
                self.drop0 = nn.Identity(device=device)
                self.drop1 = nn.Identity(device=device)
        
    def reset_parameters(self):
        '''
        Reset the parameters of the block, to an initial state used for training a model from scratch
        '''
        # Call the sub blocks reset_parameters
        self.self_attn.reset_parameters()

        # Reset the layernorms
        self.input_layernorm.weight.data.fill_(1.0)
        self.post_attention_layernorm.weight.data.fill_(1.0)
        
        # Update the linear layers
        self.mlp.gate_proj.reset_parameters()
        self.mlp.up_proj.reset_parameters()
        self.mlp.down_proj.reset_parameters()

    def _linear_operation(self, x:torch.Tensor, weight:torch.Tensor, bias:torch.Tensor = None) -> torch.Tensor:
        '''
        Perform the linear operation with the given weight and bias, 
        using linear_module_function if configured
        '''
        if self.linear_module_function is not None:
            return self.linear_module_function(x, weight, bias)
        else:
            return F.linear(x, weight, bias)

    def forward(
        self, 
        x:torch.Tensor, # hidden state
        last_wkv_state: torch.Tensor, 
        v_first:torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        '''
        Forward the block given the input tokens and the last state, and position embeddings
        Last state is a tuple of the following
        - time mix shift state
        - time mix wkv state

        Returns a pair of the output embedding, v_first and the next state
        '''

        # Config dtype
        request_dtype = self.configMap.get_dtype(x.dtype)

        # Ensure everything is in the right device
        x = x.to(self.input_layernorm.weight.device)

        # Forward the time mix, with position embeddings
        att_out, tmix_wkv, v_first = self.self_attn(
            self.input_layernorm(x).to(request_dtype),
            last_wkv_state, # tmix_wkv,
            v_first,
            position_embeddings=position_embeddings
        )

        # x = x + att_out
        x = self.drop0(x + att_out).to(request_dtype)

        # Does the post att layernorm first
        post_att_norm = self.post_attention_layernorm(x)

        # MLP layer handling - due to deepspeed wierd sorcery
        # of hijacking the MLP layer, and casting things to float, 
        # we are reimplementing it here
        # ---
        if self.linear_module_function is None:
            mlp_out = self.mlp(post_att_norm)
        else:
            # Then get the various weights
            mlp_gate_proj = self.mlp.gate_proj
            mlp_up_proj   = self.mlp.up_proj
            mlp_down_proj = self.mlp.down_proj
            mlp_act_fn    = self.mlp.act_fn

            # And perform the linear operations
            # ---
            # `self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))`
            # ---
            gate_out = self._linear_operation(post_att_norm.to(request_dtype), mlp_gate_proj.weight.to(request_dtype)).to(request_dtype)
            act_out  = mlp_act_fn(gate_out).to(request_dtype)
            up_out   = self._linear_operation(post_att_norm.to(request_dtype), mlp_up_proj.weight.to(request_dtype)).to(request_dtype)
            mlp_out  = self._linear_operation(act_out * up_out, mlp_down_proj.weight.to(request_dtype)).to(request_dtype)

        # x = x + ffn_out
        x = self.drop1(x + mlp_out).to(request_dtype)

        # Return the output
        return x, tmix_wkv, v_first
    
    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, 
        in_x:torch.Tensor, 
        in_wkv_state: torch.Tensor,
        in_v_first:torch.Tensor,
        out_x:torch.Tensor, 
        out_wkv_state: torch.Tensor,
        out_v_first:torch.Tensor
    ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], out_wkv_state[:], out_v_first[:] = self.forward(in_x, in_wkv_state, in_v_first)
        return out_x, out_wkv_state, out_v_first

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(self, in_x: torch.Tensor, in_wkv_state:torch.Tensor, in_v_first:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        '''
        Compiled varient of the forward function
        '''
        return self.forward(in_x, in_wkv_state, in_v_first)
    
    def load_from_model_state_dict(self, model_state_dict:dict, layer_id:int=-1, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, load the block weights accordingly
        '''
        if layer_id <= -1:
            layer_id = self.configMap.get_layer_id(-1)
        assert layer_id >= 0, f'layer_id must be >= 0, got {layer_id}'
            
        # Get the current state_dict
        current_state_dict = self.state_dict()

        # Iterate each parameter in the state_dict, and compare from the model
        for n in current_state_dict:
            model_key = f"model.layers.{layer_id}.{n}"
            if model_key not in model_state_dict:
                continue

            # Copy the values from the state_dict
            try:
                current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
            except Exception as e:
                print(f"[ERROR] loading: {model_key} | model shape: {current_state_dict[n].shape} | weight shape: {model_state_dict[model_key].shape}")
                raise e