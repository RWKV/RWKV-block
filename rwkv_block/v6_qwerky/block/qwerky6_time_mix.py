import torch, math
from torch import Tensor
from typing import Optional, Union, Tuple
from torch.nn import functional as F
from torch import nn

from transformers.models.qwen2.modeling_qwen2 import repeat_kv

from .qwerky6_block_config_map import Qwerky6BlockConfigMap
from fla.ops.gla import fused_recurrent_gla
from fla.ops.gla.naive import naive_recurrent_gla

_time_mix_backends = {
    'naive': naive_recurrent_gla,
    'fused': fused_recurrent_gla,
    "auto": fused_recurrent_gla if torch.cuda.is_available() else naive_recurrent_gla
}

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
        self.tmix_backend = _time_mix_backends[configMap.tmix_backend]

        # Linear module function
        # This is used to replace the linear module, with a custom implementation
        # Might be requried to work around some known deepspeed 3 issues
        self.linear_module_function = None

        # Build the various params
        # ---

        with torch.no_grad():
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_MIX_LORA = configMap.d_mix_lora
            D_DECAY_LORA = configMap.d_decay_lora

            
            self.time_maa_r = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.time_maa_w = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.time_maa_k = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.time_maa_v = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.time_maa_a = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.time_maa_g = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.time_maa_x = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            
            self.time_maa_w2 = nn.Parameter(torch.empty(5, D_MIX_LORA, hidden_size, device=device, dtype=dtype))
            self.time_maa_w1 = nn.Parameter(torch.empty(hidden_size, D_MIX_LORA * self.time_maa_w2.size(0), device=device, dtype=dtype))

            self.time_decay = nn.Parameter(torch.empty(1,1,hidden_size, device=device, dtype=dtype))
            self.time_decay_w1 = nn.Parameter(torch.empty(hidden_size, D_DECAY_LORA, device=device, dtype=dtype))
            self.time_decay_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, hidden_size, device=device, dtype=dtype))

        # Renamed to q,k,v,o_proj : in line with transformers naming
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size_att, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size_att, bias=True, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)

        
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

            self.time_maa_r.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)
            self.time_maa_w.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)
            self.time_maa_k.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)
            self.time_maa_v.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)
            self.time_maa_a.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0)

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
        shift_state_in:Tensor = None, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[Tensor,Tensor,Tensor]:
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
        decay_states = (
            self.time_decay +
            torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)

        r = self.q_proj(xr)
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        g = (self.gate(xg))

        gate_states = F.sigmoid(g)

        query_states = r.view(BATCH_SIZE, SEQ_LEN, -1,
                                         self.head_size).transpose(1, 2)
        key_states = k.view(BATCH_SIZE, SEQ_LEN, -1,
                                     self.head_size).transpose(1, 2)
        value_states = v.view(BATCH_SIZE, SEQ_LEN, -1,
                                         self.head_size).transpose(1, 2)
        decay_states = decay_states.view(BATCH_SIZE, SEQ_LEN, -1,
                                         self.head_size).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.n_gqa_head_group)
        value_states = repeat_kv(value_states, self.n_gqa_head_group)

        decay_states_log = -decay_states.float().exp()
        decay_states_log = decay_states_log.clamp(-5)
        key_states = (key_states * (1 - decay_states_log.exp()))

        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)

        output_final_state = True
        attn_output, output_kv_state = self.tmix_backend(
            q=query_states, k=key_states, v=value_states, gk=decay_states_log.float(),
            initial_state=wkv_state_in, output_final_state=output_final_state)

        attn_output = attn_output.transpose(1, 2).reshape(BATCH_SIZE, SEQ_LEN, -1)
        x = self.o_proj(attn_output * gate_states)

        return x, shift_state_out, output_kv_state
    
    @torch.compile(mode="default")
    def forward_with_default_compile(self, in_x:Tensor, wkv_state_in:Tensor,shift_state_in:Tensor, out_x:Tensor, wkv_state_out:Tensor, shift_state_out:Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]=None) -> tuple[Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], wkv_state_out[:], shift_state_out[:] = self.forward(in_x, wkv_state_in,shift_state_in, position_embeddings=position_embeddings)
        return out_x, wkv_state_out, shift_state_out

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(self, in_x:Tensor, wkv_state_in:Tensor, shift_state_in:Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None) -> tuple[Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no input tensor being modified. 
        Useful for reduce-overhead compile mode
        '''
        return self.forward(in_x, wkv_state_in, shift_state_in, position_embeddings=position_embeddings)
    
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
