import torch
from torch import nn
from torch import Tensor
from typing import Optional, Union
from torch.nn import functional as F

from .rwkv7_block_config_map import RWKV7BlockConfigMap
from .rwkv7_channel_mix import RWKV7ChannelMix
from .rwkv7_time_mix import RWKV7TimeMix

class RWKV7LayerBlock(torch.nn.Module):
    '''
    layer block for RWKV V7
    '''

    def __init__(self, configMap: Union[RWKV7BlockConfigMap, any]):
        super().__init__()

        configMap:RWKV7BlockConfigMap = RWKV7BlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        hidden_size = configMap.hidden_size
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')
        dropout_rate = configMap.dropout_rate

        # Get valid layer_id
        layer_id = configMap.get_layer_id(-1)
        assert layer_id >= 0, f'layer_id must be >= 0, got {layer_id}'

        with torch.device(device):
            # Setup the layernorms, and mixes
            self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
            self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)

            if layer_id == 0:
                self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype)
            else:
                self.ln0 = nn.Identity()

            self.att = RWKV7TimeMix(configMap)
            self.ffn = RWKV7ChannelMix(configMap)

            # Setup droupout at block level
            if dropout_rate > 0.0:            
                self.drop0 = nn.Dropout(p = dropout_rate)
                self.drop1 = nn.Dropout(p = dropout_rate)
            else:
                self.drop0 = nn.Identity()
                self.drop1 = nn.Identity()
        
    def reset_parameters(self):
        '''
        Reset the parameters of the block, to an initial state used for training a model from scratch
        '''
        # configMap = self.configMap

        # # Get required props
        # hidden_size = configMap.hidden_size
        # device = configMap.get_device(None)
        # dtype = configMap.get_dtype('bfloat16')
        # dropout_rate = configMap.dropout_rate

        # # Get valid layer_id
        # layer_id = configMap.get_layer_id(-1)
        # assert layer_id >= 0, f'layer_id must be >= 0, got {layer_id}'

        # Redo the Setup for the layernorms, and mixes
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()

        # if layer_id == 0:
        #     self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype)
        # else:
        #     self.ln0 = nn.Identity()
        self.ln0.reset_parameters()

        # Call the sub blocks reset_parameters
        self.att.reset_parameters()
        self.ffn.reset_parameters()

    def forward(
        self, x:torch.Tensor,
        last_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor], 
        v_first:torch.Tensor
        ) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Forward the block given the input tokens and the last state
        Last state is a tuple of the following
        - time mix shift state
        - time mix wkv state
        - channel mix shift state

        Returns a pair of the output embedding, v_first and the next state
        '''

        # # Ensure everything is in the right device
        # x = x.to(self.ln1.weight.device)
        # last_state = [ s.to(self.ln1.weight.device) for s in last_state ]

        # Note, that this only applies for layer 0
        ln0_out = self.ln0(x)

        # assert self.ln1(x) is not None
        # assert last_state.tmix_shift is not None
        # assert last_state.tmix_wkv is not None

        att_out, tmix_shift, tmix_wkv, v_first = self.att(
            self.ln1(ln0_out),
            last_state[0], # tmix_shift,
            last_state[1], # tmix_wkv
            v_first
        )

        # x = x + att_out
        x = self.drop0(ln0_out + att_out)

        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state[2] # cmix_shift,
        )

        # x = x + ffn_out
        x = self.drop1(x + ffn_out)

        # # Debugging for NaN
        # layer_id = self.configMap.get_layer_id(-1)
        # assert torch.isnan(att_out).sum() == 0, f'NaN detected att_out @ layer {layer_id}'
        # assert torch.isnan(ffn_out).sum() == 0, f'NaN detected ffn_out @ layer {layer_id}'
        # assert torch.isnan(v_first).sum() == 0, f'NaN detected v_first @ layer {layer_id}'
        # assert torch.isnan(tmix_shift).sum() == 0, f'NaN detected tmix_shift @ layer {layer_id}'
        # assert torch.isnan(tmix_wkv).sum() == 0, f'NaN detected tmix_wkv @ layer {layer_id}'
        # assert torch.isnan(ffn_state).sum() == 0, f'NaN detected ffn_state @ layer {layer_id}'
        # assert torch.isnan(x).sum() == 0, f'NaN detected block out @ layer {layer_id}'

        return x, (tmix_shift, tmix_wkv, ffn_state), v_first
    
    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, 
        in_x:torch.Tensor, 
        in_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor],
        in_v_first:torch.Tensor,
        out_x:torch.Tensor, 
        out_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor],
        out_v_first:torch.Tensor
        ) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], tmp_state, out_v_first[:] = self.forward(in_x, in_state, in_v_first)
        out_state[0][:], out_state[1][:], out_state[2][:] = tmp_state
        return out_x, out_state, out_v_first

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(self, in_x: torch.Tensor, in_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor], in_v_first:torch.Tensor) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Compiled varient of the forward function
        '''
        return self.forward(in_x, in_state, in_v_first)
    
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
            model_key = f"blocks.{layer_id}.{n}"
            if model_key not in model_state_dict:
                continue

            # Copy the values from the state_dict
            try:
                current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
            except Exception as e:
                print(f"[ERROR] loading: {model_key} | model shape: {current_state_dict[n].shape} | weight shape: {model_state_dict[model_key].shape}")
                raise e