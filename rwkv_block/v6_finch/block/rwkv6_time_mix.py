import torch
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

from .rwkv6_block_config_map import RWKV6BlockConfigMap
from .rwkv6_optimized_ops import RWKVx060_reshape_run

class RWKV6TimeMix(torch.nn.Module):
    '''
    Time Mix block for RWKV V6
    '''

    def __init__(self, configMap: Union[RWKV6BlockConfigMap, any]):
        super().__init__()

        configMap:RWKV6BlockConfigMap = RWKV6BlockConfigMap.normalize(configMap)
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

        tmix_maa_dim = configMap.tmix_maa_dim
        tmix_decay_dim = configMap.tmix_decay_dim

        # Build the various params
        # ---

        # Ref:
        # https://github.com/SmerkyG/LinearAttentionArena/blob/f700ff6ae1c834cd0d4110e90d24ae90d69056c0/src/tmix_x060.py
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, hidden_size, device=device, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_DIM = tmix_maa_dim # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(hidden_size, D_MIX_DIM*5, device=device, dtype=dtype))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_DIM, hidden_size, device=device, dtype=dtype).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(hidden_size_att, device=device, dtype=dtype)
            for n in range(hidden_size_att):
                decay_speed[n] = -6 + 5 * (n / (hidden_size_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,hidden_size_att))

            D_DECAY_DIM = tmix_decay_dim
            self.time_decay_w1 = nn.Parameter(torch.zeros(hidden_size, D_DECAY_DIM, device=device, dtype=dtype))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_DIM, hidden_size_att, device=device, dtype=dtype).uniform_(-0.01, 0.01))
            
            tmp = torch.zeros(hidden_size_att, device=device, dtype=dtype)
            for n in range(hidden_size_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (hidden_size_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)

        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, device=device, dtype=dtype)
        self.ln_x = nn.GroupNorm(n_head, hidden_size_att, device=device, dtype=dtype, eps=(1e-5)*(self.head_size_divisor**2))
        
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
        # Get the sizing
        BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE = x.size()
        N_HEAD = self.n_head
        HEAD_SIZE = self.head_size

        ##########
        ## x060
        ##########

        shift_state_out = x[:, -1]
        dxprev = torch.concat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(BATCH_SIZE*SEQ_LEN, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)

        mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        xw = x + dxprev * (self.time_maa_w + mw)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        w = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(r.dtype)
        u = self.time_faaaa

        x, wkv_state_out = RWKVx060_reshape_run(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE, N_HEAD, HEAD_SIZE, r, k, v, w, u, wkv_state_in, backend=self.tmix_backend)
        x = x.view(BATCH_SIZE * SEQ_LEN, IN_EMB_SIZE)

        x = self.ln_x(x).view(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)
        x = self.output(x * g)

        return x, shift_state_out, wkv_state_out

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