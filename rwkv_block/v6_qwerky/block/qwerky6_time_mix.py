import torch, math
from torch import Tensor
from typing import Union, Tuple
from torch.nn import functional as F
from torch import nn

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from ...v6_finch.block.rwkv6_time_mix import RWKV6TimeMix, _run_tmix_backend, _has_fla, _has_triton, _has_cuda
from .qwerky6_block_config_map import Qwerky6BlockConfigMap

class Qwerky6TimeMix(torch.nn.Module):
    '''
    Time Mix block for QWERKY V6
    '''

    def __init__(self, configMap: Union[Qwerky6BlockConfigMap, any]):
        super().__init__()

        configMap:Qwerky6BlockConfigMap = Qwerky6BlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        hidden_size = configMap.hidden_size
        # num_hidden_layers = configMap.num_hidden_layers

        # Get the layer id
        layer_id = configMap.get_layer_id(0)
        self.layer_id = layer_id

        # Get optional props
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')

        # By default, hidden_size_ffn = hidden_size
        hidden_size_att = configMap.get_hidden_size_att()

        # Head size settings
        head_size = configMap.head_size
        self.head_size = head_size

        # Number of heads
        n_head = hidden_size // head_size
        assert hidden_size % head_size == 0, "hidden_size should be divisible by head_size"
        self.n_head = n_head

        # Number of GQA heads
        n_gqa_head = hidden_size_att // head_size
        assert hidden_size_att % head_size == 0, "hidden_size_att should be divisible by head_size"
        self.n_gqa_head = n_gqa_head

        # Number of GQA head groups
        n_gqa_head_group = n_head // n_gqa_head
        assert n_head % n_gqa_head == 0, "n_head should be divisible by n_gqa_head"
        self.n_gqa_head_group = n_gqa_head_group

        # Backend
        self.tmix_backend = configMap.tmix_backend

        # Linear module function
        # This is used to replace the linear module, with a custom implementation
        # Might be requried to work around some known deepspeed 3 issues
        self.linear_module_function = None

        # Build the various params
        # ---

        with torch.no_grad():
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_MIX_LORA = 128
            D_DECAY_LORA = 128

            
            self.x_r = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.x_w = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.x_k = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.x_v = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.x_a = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.x_g = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.x_x = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            
            self.x_w2 = nn.Parameter(torch.empty(5, D_MIX_LORA, hidden_size, device=device, dtype=dtype))
            self.x_w1 = nn.Parameter(torch.empty(hidden_size, D_MIX_LORA * self.x_w2.size(0), device=device, dtype=dtype))

            self.w_w0 = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.w_w1 = nn.Parameter(torch.empty(hidden_size, D_DECAY_LORA, device=device, dtype=dtype))
            self.w_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, hidden_size, device=device, dtype=dtype))

        # Renamed to q,k,v,o_proj : in line with transformers naming
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size_att, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size_att, bias=True, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)

        self.ln_x = nn.GroupNorm(n_head, hidden_size, device=device, dtype=dtype, eps=(1e-5)*head_size)
        
    def reset_parameters(self):
        '''
        Reset the parameters of the block, to an initial state used for training a model from scratch
        '''
        configMap = self.configMap

        # Get required props
        hidden_size = configMap.hidden_size
        num_hidden_layers = configMap.num_hidden_layers

        # Get the layer id
        layer_id = self.layer_id

        # Get optional props
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')

        # Head size settings
        head_size = self.head_size
        n_head = self.n_head

        # Reset the various params
        # ---
        with torch.device(device), torch.no_grad():
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, hidden_size, device=device, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            self.x_r.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)
            self.x_w.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)
            self.x_k.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)
            self.x_v.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)
            self.x_a.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)

            # idk what goes here TODO

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
        x:Tensor, 
        wkv_state_in:Tensor = None, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[Tensor,Tensor,Tensor]:
        '''
        forwarding time mix given the model weights and the input tokens and states.
        
        Given:
        - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
        - Incoming wkv_state containing of shapes [batch_size, n_head, head_size, head_size]
        - Incoming v_first_val of shape [batch_size, seq_len, embedding_size]
        
        Returns a pair 
        - output embedding of shape [batch_size, seq_len, embedding_size]
        - output wkv_state of shape [batch_size, n_head, head_size, head_size] 
        - output v_first_val of shape [batch_size, seq_len, embedding_size]
        '''

        # x dtype
        x_dtype = x.dtype

        # Get the sizing
        BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE = x.size()
        N_HEAD = self.n_head
        HEAD_SIZE = self.head_size

        # Ensure wkv_state_in is initialized
        if wkv_state_in is None:
            wkv_state_in = torch.zeros(BATCH_SIZE,N_HEAD,HEAD_SIZE,HEAD_SIZE, dtype=torch.float, device=self.w0.device)
        else:
            wkv_state_in = wkv_state_in.clone()

        ##########
        ## qwerky6
        ##########

        ## ---
        ## No token shift
        ## ---
        
        # if shift_state_in is None:
        #     shift_state_in = torch.zeros(BATCH_SIZE, IN_EMB_SIZE, dtype=x.dtype, device=x.device)

        # shift_state_out = x[:, -1]
        # dxprev = torch.cat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        ## ---
        ## Normalize xr-xg values to x
        ## ---
        # xr = x + dxprev * self.x_r
        # xw = x + dxprev * self.x_w
        # xk = x + dxprev * self.x_k
        # xv = x + dxprev * self.x_v
        # xa = x + dxprev * self.x_a
        # xg = x + dxprev * self.x_g
        # xx = dxprev

        xr = xw = xk = xv = xa = xg = x.to(self.q_proj.weight.device)

        r = self._linear_operation(xr, self.q_proj.weight, self.q_proj.bias) # self.q_proj(xr.to(self.q_proj.weight.dtype))
        w_lora_result = self.w0.float() + (torch.tanh(xw.float() @ self.w1.float()) @ self.w2.float()).float()
        k = self._linear_operation(xk, self.k_proj.weight, self.k_proj.bias) # self.k_proj(xk.to(self.k_proj.weight.dtype))
        v = self._linear_operation(xv, self.v_proj.weight, self.v_proj.bias) # self.v_proj(xv.to(self.v_proj.weight.dtype))
        g = torch.sigmoid(xg.float() @ self.g1.float()) @ self.g2.float()
        iclr = torch.sigmoid(self.a0.float() + (xa.float() @ self.a1.float()) @ self.a2.float()) # a is "in-context learning rate"

        ##########
        # Apply rotary pos emb
        ##########
        if position_embeddings is not None:
            # Debug prints
            # print(f"r shape before view: {r.shape}")  # Should be [B, T, hidden_size]
            # print(f"k shape before view: {k.shape}")  # Should be [B, T, hidden_size_att]
            # print(f"N_HEAD: {N_HEAD}, n_gqa_head: {self.n_gqa_head}, HEAD_SIZE: {HEAD_SIZE}")
            
            r = r.view(BATCH_SIZE, SEQ_LEN, -1, HEAD_SIZE)
            k = k.view(BATCH_SIZE, SEQ_LEN, -1, HEAD_SIZE)
            
            # print(f"r shape after view: {r.shape}")  # Should be [B, T, N_HEAD, HEAD_SIZE]
            # print(f"k shape after view: {k.shape}")  # Should be [B, T, n_gqa_head, HEAD_SIZE]
            
            cos, sin = position_embeddings
            r, k = apply_rotary_pos_emb(r, k, cos, sin, unsqueeze_dim=2)
            r = r.view(BATCH_SIZE, SEQ_LEN, -1)
            k = k.view(BATCH_SIZE, SEQ_LEN, -1)
            # r = r.transpose(1,2).view(B,T,-1).to(v.dtype)
            # k = k.transpose(1,2).view(B,T,-1).to(v.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(BATCH_SIZE, SEQ_LEN, -1, 1, HEAD_SIZE).expand(-1, -1, -1, self.n_gqa_head_group, -1).reshape(BATCH_SIZE, SEQ_LEN, -1)
        v = v.view(BATCH_SIZE, SEQ_LEN, -1, 1, HEAD_SIZE).expand(-1, -1, -1, self.n_gqa_head_group, -1).reshape(BATCH_SIZE, SEQ_LEN, -1)

        ##########
        # qwerky6
        ##########
        
        # kk = F.normalize((k * self.k_k).view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1), dim=-1, p=2.0).view(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)
        kk = F.normalize((k * self.k_k).view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1), dim=-1, p=2.0).view(BATCH_SIZE,SEQ_LEN,-1)

        # ---
        # Note the change to ICLR value is intentional here
        # as a means to normalize the value without layernorm
        # commented is the original code
        # ---
        # k = k * (1 + (iclr-1) * self.k_a)
        # ---
        iclr = 1 + (iclr-1) * self.k_a
        k = k * iclr

        ##########
        # x060
        ##########
        if v_first_val is None:
            v_first_val = v # store the v of the first layer
        else:
            v = v.float() + (v_first_val.float() - v.float()) * torch.sigmoid(self.v0.float() + (xv.float() @ self.v1.float()) @ self.v2.float()) # add value residual
            
        ##########
        # Auto select the backend if not specified
        tmix_backend = self.tmix_backend.lower()
        if tmix_backend == "auto":
            if r.device.type == "cpu":
                tmix_backend = "pytorch"
            elif _has_triton is True:
                tmix_backend = "triton_bighead"
            elif _has_fla is True:
                tmix_backend = "fla"
            elif _has_cuda is True:
                tmix_backend = "cuda"
            else:
                tmix_backend = "pytorch"

        # # Warn against CUDA backend
        # if tmix_backend == "cuda" and HEAD_SIZE != 64:
        #     print(f"[WARNING] !!! CUDA backend has potential memory safety issues for qwerky for non-64 head sizes !!!")

        # Contigous safety
        xi = torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() 
        xi, r, k, v, kk, iclr = [i.bfloat16().contiguous() for i in [xi, r, k, v, kk, iclr]]
        w_lora_result, wkv_state_in = [i.float().contiguous() for i in [w_lora_result, wkv_state_in]]

        # Apply the time mix backend
        xx, wkv_state_out = _run_tmix_backend(tmix_backend, r, w_lora_result, k, v, kk, iclr, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xi, wkv_state_in)
        ##########

        # xx = self.ln_x(xx.view(BATCH_SIZE * SEQ_LEN, IN_EMB_SIZE)).view(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)
        xn = torch.nn.functional.group_norm(xx.view(BATCH_SIZE * SEQ_LEN, IN_EMB_SIZE).float(), num_groups=N_HEAD, weight=self.ln_x.weight.float(), bias=self.ln_x.bias.float(), eps = self.ln_x.eps).view(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE).to(dtype=x_dtype)

        # ---
        # Intentionally removed for qwerky6
        # ---
        # xx = xx + ((r.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)*k.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)).view(BATCH_SIZE,SEQ_LEN,IN_EMB_SIZE)
        
        # xo = self.o_proj(xn * g).to(dtype=x_dtype)
        # F.linear((xn * g).float(), self.o_proj.weight.float()).to(dtype=x_dtype)
        xo = self._linear_operation((xn * g).float(), self.o_proj.weight.float()).to(x_dtype)

        # Return the results
        return xo, wkv_state_out
    
    @torch.compile(mode="default")
    def forward_with_default_compile(self, in_x:Tensor, wkv_state_in:Tensor, out_x:Tensor, wkv_state_out:Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]=None) -> tuple[Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], wkv_state_out[:], = self.forward(in_x, wkv_state_in, position_embeddings=position_embeddings)
        return out_x, wkv_state_out

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(self, in_x:Tensor, wkv_state_in:Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None) -> tuple[Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no input tensor being modified. 
        Useful for reduce-overhead compile mode
        '''
        return self.forward(in_x, wkv_state_in, position_embeddings=position_embeddings)
    
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
            model_key = f"model.layers.{layer_id}.self_attn.{n}"
            if model_key not in model_state_dict:
                continue

            # Copy the values from the state_dict
            try:
                current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
            except Exception as e:
                print(f"[ERROR] loading: {model_key} | model shape: {current_state_dict[n].shape} | weight shape: {model_state_dict[model_key].shape}")
                raise e
