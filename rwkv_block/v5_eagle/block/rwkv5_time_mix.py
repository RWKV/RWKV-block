import torch
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

from .rwkv5_block_config_map import RWKV5BlockConfigMap
from .rwkv5_optimized_ops import modified_lerp, RWKVx050_chunk

class RWKV5TimeMix(torch.nn.Module):
    '''
    Time Mix block for RWKV V5
    '''

    def __init__(self, configMap: Union[RWKV5BlockConfigMap, any]):
        super().__init__()

        configMap:RWKV5BlockConfigMap = RWKV5BlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        hidden_size = configMap.hidden_size
        num_hidden_layers = configMap.num_hidden_layers

        # Get optional props
        hidden_size_att = configMap.get_hidden_size_att()
        layer_id = configMap.get_layer_id(0)
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')

        n_head = configMap.get_n_head()
        head_size = configMap.head_size
        head_size_divisor = configMap.head_size_divisor

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = head_size_divisor
        self.tmix_backend = configMap.tmix_backend

        # Build the various params
        # ---

        # V5-R4 changes
        # https://github.com/BlinkDL/RWKV-LM/commit/5aab658f945ba80745d36c2ab411fb43df3a74f9    
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, hidden_size, device=device, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(hidden_size_att, device=device, dtype=dtype)
            for n in range(hidden_size_att):
                decay_speed[n] = -6 + 5 * (n / (hidden_size_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(n_head, head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(hidden_size_att, device=device, dtype=dtype)
            for n in range(hidden_size_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (hidden_size_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(n_head, head_size))

        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)

        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, device=device, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)
        self.ln_x = nn.GroupNorm(n_head, hidden_size_att, device=device, dtype=dtype)
        
    def forward(self, x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor) -> tuple[Tensor,Tensor,Tensor]:
        '''
        forwarding time mix given the model weights and the input tokens and states.
        
        Given:
        - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
        - Incoming states containing of shapes:
            [batch_size, state_size] ## Token Shift state,
            [batch_size, n_head, head_size, head_size] ## WKV state
        
        
        Returns a pair 
        - output embedding of shape [batch_size, seq_len, embedding_size]
        - output state of shapes:
            [batch_size, state_size] ## Token Shift state,
            [batch_size, n_head, head_size, head_size] ## WKV state
        
        '''

        # Get the shift state
        shift_state_out = x[:,-1]

        # x_chunk_len = x.size(-2)
        # assert x_chunk_len % wkv_chunk_len == 0 or x_chunk_len == 1, "optimized nocuda rwkv requires data len supplied to be an exact multiple of the chunk len"

        # Get the x sizing
        B, T, C = x.size()
        H = self.n_head
        K = self.head_size
        head_size_divisor = self.head_size_divisor

        # Perform the tokenshift, and get the respective state
        xx = torch.concat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1)

        # Get the xk, xv, xr, xg, and rkvg
        xk = modified_lerp(x, self.time_mix_k, xx)
        xv = modified_lerp(x, self.time_mix_v, xx)
        xr = modified_lerp(x, self.time_mix_r, xx)
        xg = modified_lerp(x, self.time_mix_g, xx)

        r = self.receptance(xr).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(xk).view(B, T, H, K).transpose(1, 2)      # BHTK
        v = self.value(xv).view(B, T, H, K).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(xg))

        w = torch.exp(-torch.exp(self.time_decay.float())).view(1,H,1,K).expand(1,H,T,K)

        u = self.time_faaaa.view(1,H,1,K).to(r.dtype)

        # Logits and state
        wkv_state_out = wkv_state_in.to(r.dtype)

        # RWKVx060 optimized kernels (backward compatible with RWKVx050)
        x_logits, wkv_state_out = RWKVx050_chunk(r, k, v, w, u, wkv_state_out, backend=self.tmix_backend) 
        x_logits = x_logits.transpose(1,2).reshape(B,T,C)

        # Reshape and normalize the logits
        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits / head_size_divisor).view(B, T, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return (x_logits, shift_state_out, wkv_state_out)

    @torch.compile(mode="default")
    def forward_with_default_compile(self, in_x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor, out_x:Tensor, shift_state_out:Tensor, wkv_state_out:Tensor) -> tuple[Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], shift_state_out[:], wkv_state_out[:] = self.forward(in_x, shift_state_in, wkv_state_in)
        return out_x, shift_state_out, wkv_state_out
    
    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(self, in_x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor) -> tuple[Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no input tensor being modified. 
        Useful for reduce-overhead compile mode
        '''
        return self.forward(in_x, shift_state_in, wkv_state_in)
    
    # ---------------------------------
    #
    #  Model state handling
    #
    # ---------------------------------
    
    def load_from_model_state_dict(self, model_state_dict: dict, layer_id:int, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the the current module weights, using the layer_id
        '''
        # Get the current state_dict
        current_state_dict = self.state_dict()

        # Iterate each parameter in the state_dict, and compare from the model
        for n in current_state_dict:
            model_key = f"blocks.{layer_id}.att.{n}"
            if model_key not in model_state_dict:
                continue

            # Copy the values from the state_dict
            current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
