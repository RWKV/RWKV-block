import torch, math
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

from .rwkv7_block_config_map import RWKV7BlockConfigMap

# Checks if a package is installed and importable
import importlib.util
import sys

def has_python_package(name):
    '''
    Checks if a package is installed and importable
    '''
    if name in sys.modules:
        return True
    if importlib.util.find_spec(name) is not None:
        return True
    return False

# Check for triton and fla packages
_has_triton = has_python_package("triton")
_has_fla = has_python_package("fla")
_has_cuda = torch.cuda.is_available()

# Warning if FLA is not available
if not _has_fla:
    print("[WARNING] fla not available, falling back to triton, cuda or pytorch mode - install fla from `https://github.com/fla-org/flash-linear-attention`")

# Check if the FLA package is available
class RWKV7TimeMix(torch.nn.Module):
    '''
    Time Mix block for RWKV V7
    '''

    def __init__(self, configMap: Union[RWKV7BlockConfigMap, any]):
        '''
        Initialize the TimeMix block.
        
        Note: this does not initialize the parameter weights itself
        which would depend on the `reset_parameters()` method
        '''
        super().__init__()

        configMap:RWKV7BlockConfigMap = RWKV7BlockConfigMap.normalize(configMap)
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

        # Assert hidden_size == hidden_size_att, until we support different hidden_size and hidden_size_att
        assert hidden_size == hidden_size_att, "hidden_size should be equal to hidden_size_att (@TODO: support different hidden_size and hidden_size_att)"

        # Head size settings
        head_size = configMap.head_size
        self.head_size = head_size

        # Number of heads
        n_head = hidden_size_att // head_size
        assert hidden_size_att % head_size == 0, "hidden_size_att should be divisible by head_size"
        self.n_head = n_head

        # Backend
        self.tmix_backend = configMap.tmix_backend

        # Build the various params
        # ---
        with torch.device(device):
            with torch.no_grad():
                # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
                def calc_lora_rank(exponent, multiplier):
                    return max(1, round(hidden_size_att ** exponent * multiplier / 32)) * 32
                D_DECAY_LORA = calc_lora_rank(0.5, 1.8)
                D_AAA_LORA   = calc_lora_rank(0.5, 1.8)
                D_MV_LORA    = calc_lora_rank(0.5, 1.3)
                D_GATE_LORA  = calc_lora_rank(0.8, 0.6)

                self.x_r = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.x_w = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.x_k = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.x_v = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.x_a = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.x_g = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))

                self.w0 = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.w1 = nn.Parameter(torch.empty(hidden_size_att, D_DECAY_LORA, dtype=dtype))
                self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, hidden_size_att, dtype=dtype))

                self.a0 = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.a1 = nn.Parameter(torch.empty(hidden_size_att,D_AAA_LORA, dtype=dtype))
                self.a2 = nn.Parameter(torch.empty(D_AAA_LORA,hidden_size_att, dtype=dtype))
                
                if layer_id > 0:
                    self.v0 = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                    self.v1 = nn.Parameter(torch.empty(hidden_size_att,D_MV_LORA, dtype=dtype))
                    self.v2 = nn.Parameter(torch.empty(D_MV_LORA,hidden_size_att, dtype=dtype))
                    
                self.g1 = nn.Parameter(torch.empty(hidden_size_att, D_GATE_LORA, dtype=dtype))
                self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, hidden_size_att, dtype=dtype))

                self.k_k = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.k_a = nn.Parameter(torch.empty(1,1,hidden_size_att, dtype=dtype))
                self.r_k = nn.Parameter(torch.empty(n_head, head_size, dtype=dtype))

            self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
            self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
            self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
            self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
            
            self.ln_x = nn.GroupNorm(n_head, hidden_size_att, dtype=dtype, eps=(1e-5)*head_size)
        
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

        # By default, hidden_size_ffn = hidden_size
        hidden_size_att = configMap.get_hidden_size_att()

        # Assert hidden_size == hidden_size_att, until we support different hidden_size and hidden_size_att
        assert hidden_size == hidden_size_att, "hidden_size should be equal to hidden_size_att (@TODO: support different hidden_size and hidden_size_att)"

        # Head size settings
        head_size = self.head_size

        # Number of heads
        n_head = hidden_size_att // head_size
        assert hidden_size_att % head_size == 0, "hidden_size_att should be divisible by head_size"
        
        # Reset the various params
        # ---
        with torch.device(device), torch.no_grad():
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, hidden_size, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            def calc_lora_rank(exponent, multiplier):
                return max(1, round(hidden_size ** exponent * multiplier / 32)) * 32
            D_DECAY_LORA = calc_lora_rank(0.5, 1.8)
            D_AAA_LORA   = calc_lora_rank(0.5, 1.8)
            D_MV_LORA    = calc_lora_rank(0.5, 1.3)
            D_GATE_LORA  = calc_lora_rank(0.8, 0.6)

            self.x_r.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k.copy_(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v.copy_(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x[i], gain=gain * scale)
                else:
                    assert False
                return x.to(dtype=dtype)

            # D_DECAY_LORA = max(32, int(round(  (1.8*(hidden_size**0.5))  /32)*32))
            decay_speed = torch.ones(hidden_size_att, dtype=dtype)
            for n in range(hidden_size_att):
                decay_speed[n] = -7 + 5 * (n / (hidden_size_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            
            self.w0.copy_(decay_speed.reshape(1,1,hidden_size_att).to(dtype=dtype) + 0.5)  # !!! 0.5 comes from F.softplus !!!
            self.w1.copy_(torch.zeros(hidden_size_att, D_DECAY_LORA, dtype=dtype))
            self.w2.copy_(ortho_init(torch.zeros(D_DECAY_LORA, hidden_size_att), 0.1))

            # D_AAA_LORA = max(32, int(round(  (1.8*(hidden_size**0.5))  /32)*32)) # suggestion
            self.a0.copy_(torch.zeros(1,1,hidden_size_att, dtype=dtype))
            self.a1.copy_(torch.zeros(hidden_size_att, D_AAA_LORA, dtype=dtype))
            self.a2.copy_(ortho_init(torch.zeros(D_AAA_LORA, hidden_size_att), 0.1))

            # D_MV_LORA = max(32, int(round(  (1.3*(hidden_size**0.5))  /32)*32)) # suggestion
            if layer_id > 0:
                self.v0.copy_(torch.zeros(1,1,hidden_size_att, dtype=dtype)+1.0)
                self.v1.copy_(torch.zeros(hidden_size_att, D_MV_LORA, dtype=dtype))
                self.v2.copy_(ortho_init(torch.zeros(D_MV_LORA, hidden_size_att), 0.1))

            # D_GATE_LORA = max(32, int(round(  (0.6*(hidden_size**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1.copy_(torch.zeros(hidden_size_att, D_GATE_LORA, dtype=dtype))
            self.g2.copy_(ortho_init(torch.zeros(D_GATE_LORA, hidden_size_att), 0.1))

            self.k_k.copy_(torch.ones(1,1,hidden_size_att, dtype=dtype)*0.85)
            self.k_a.copy_(torch.ones(1,1,hidden_size_att, dtype=dtype))
            self.r_k.copy_(torch.zeros(n_head,head_size, dtype=dtype))
            
        self.receptance.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        self.output.reset_parameters()
        self.ln_x.reset_parameters()

    def forward(self, x:Tensor, shift_state_in:Tensor=None, wkv_state_in:Tensor=None, v_first_val:Tensor=None) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        '''
        forwarding time mix given the model weights and the input tokens and states.
        
        Given:
        - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
        - Incoming states containing of shapes:
            [batch_size, state_size] ## Token Shift state,
            [batch_size, n_head, head_size, head_size] ## WKV state
        - Incoming v_first_val of shape [batch_size, seq_len, embedding_size]
        
        Returns a pair 
        - output embedding of shape [batch_size, seq_len, embedding_size]
        - output state of shapes:
            [batch_size, state_size] ## Token Shift state,
            [batch_size, n_head, head_size, head_size] ## WKV state
        - output v_first_val of shape [batch_size, seq_len, embedding_size]
        
        '''
        # Get the sizing
        BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE = x.size()
        N_HEAD = self.n_head
        HEAD_SIZE = self.head_size

        # Ensure wkv_state_in is initialized
        if wkv_state_in is None:
            wkv_state_in = torch.zeros(BATCH_SIZE,N_HEAD,HEAD_SIZE,HEAD_SIZE, dtype=torch.float, device=self.w0.device)
        else:
            wkv_state_in = wkv_state_in.clone()

        # Ensure shift_state_in is initialized
        if shift_state_in is None:
            shift_state_in = torch.zeros(BATCH_SIZE, IN_EMB_SIZE, dtype=x.dtype, device=x.device)

        ##########
        ## x070
        ##########

        shift_state_out = x[:, -1]
        dxprev = torch.cat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xr = x + dxprev * self.x_r
        xw = x + dxprev * self.x_w
        xk = x + dxprev * self.x_k
        xv = x + dxprev * self.x_v
        xa = x + dxprev * self.x_a
        xg = x + dxprev * self.x_g
        xx = dxprev

        r = self.receptance(xr)
        w_lora_result = self.w0 + (torch.tanh(xw @ self.w1) @ self.w2).float()
        k = self.key(xk)
        v = self.value(xv)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        iclr = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"

        kk = F.normalize((k * self.k_k).view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1), dim=-1, p=2.0).view(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)
        k = k * (1 + (iclr-1) * self.k_a)

        if self.layer_id == 0 or v_first_val is None:
            v_first_val = v # store the v of the first layer
        else:
            v = v + (v_first_val - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        
        ##########
        # Apply the time mix backend
        xx, wkv_state_out = _run_tmix_backend(self.tmix_backend.lower(), r, w_lora_result, k, v, kk, iclr, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xx, wkv_state_in)
        ##########

        xx = self.ln_x(xx.view(BATCH_SIZE * SEQ_LEN, IN_EMB_SIZE)).view(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)
        xx = xx + ((r.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)*k.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)).view(BATCH_SIZE,SEQ_LEN,IN_EMB_SIZE)
        xx = self.output(xx * g)

        return xx, shift_state_out, wkv_state_out, v_first_val
    
    @torch.compile(mode="default")
    def forward_with_default_compile(self, in_x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor, v_first_val_in:Tensor, out_x:Tensor, shift_state_out:Tensor, wkv_state_out:Tensor, v_first_val_out:Tensor) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], shift_state_out[:], wkv_state_out[:], v_first_val_out[:] = self.forward(in_x, shift_state_in, wkv_state_in, v_first_val_in)
        return out_x, shift_state_out, wkv_state_out, v_first_val_out

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(self, in_x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor, v_first_val:Tensor) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no input tensor being modified. 
        Useful for reduce-overhead compile mode
        '''
        return self.forward(in_x, shift_state_in, wkv_state_in, v_first_val)
    
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
            try:
                current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
            except Exception as e:
                print(f"[ERROR] loading: {model_key} | model shape: {current_state_dict[n].shape} | weight shape: {model_state_dict[model_key].shape}")
                raise e

@staticmethod
def _run_tmix_backend(
    tmix_backend,
    r, w_lora_result, k, v, kk, iclr, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xx, wkv_state_in
):
    # Auto select the backend if not specified
    if tmix_backend == "auto":
        if r.device.type == "cpu":
            tmix_backend = "pytorch"
        elif _has_fla is True:
            tmix_backend = "fla"
        elif _has_triton is True:
            tmix_backend = "triton"
        elif _has_cuda is True:
            tmix_backend = "cuda"
        else:
            tmix_backend = "pytorch"

    # Set the default device, if not pytorch
    # this mitigates with mismatched default device issues
    # known to occur with triton, fla, and cuda
    # 
    # This unfortunately will not work with single thread
    # multi-gpu setups. Sadly
    if tmix_backend != "pytorch":
        device = r.device
        if torch.get_default_device() != device:
            torch.set_default_device(device)
            torch.cuda.set_device(device)

    # Tracking the dtype
    xx_dtype = xx.dtype

    ######## cuda-based method 
    # wkv_state_out = wkv_state_in.clone()
    # w = -F.softplus(-(self.w0 + w)) - 0.5 # soft-clamp to (-inf, -0.5)
    # xx = RWKV7_OP(wkv_state_out, r, w, k, v, -kk, kk*a)
    ######## cuda-based method 

    if tmix_backend == "pytorch_ref" or tmix_backend == "pytorch_ref_ori":
        # Pure pytorch mode for rwkv attention
        from .kernel.rwkv7_attn_pytorch import rwkv7_attn_pytorch_ref
        # Reference minimal compilation version
        w = torch.exp(-0.606531 * torch.sigmoid(w_lora_result)) # 0.606531 = exp(-0.5)
        xx, wkv_state_out = rwkv7_attn_pytorch_ref(r, w, k, v, kk, iclr, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xx, wkv_state_in) 
    elif tmix_backend == "pytorch_ref_fp32":
        # Pure pytorch mode for rwkv attention
        from .kernel.rwkv7_attn_pytorch import rwkv7_attn_pytorch_ref_fp32
        # Modified to follow the same logic as "cuda" version
        # w = torch.exp(-0.606531 * torch.sigmoid(w_lora_result)) # 0.606531 = exp(-0.5)
        w = -F.softplus(-w_lora_result) - 0.5 # ref_fp32, follows the cuda style (and does its own -exp * exp)
        xx, wkv_state_out = rwkv7_attn_pytorch_ref_fp32(r, w, k, v, kk, iclr, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xx, wkv_state_in) 
    elif tmix_backend == "pytorch":
        # Pure pytorch mode for rwkv attention
        from .kernel.rwkv7_attn_pytorch import rwkv7_attn_pytorch
        # Tweaked pytorch compile varient
        w = torch.exp(-0.606531 * torch.sigmoid(w_lora_result)) # 0.606531 = exp(-0.5)
        xx, wkv_state_out = rwkv7_attn_pytorch(r, w, k, v, kk, iclr, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xx, wkv_state_in) 
    elif tmix_backend in ["triton", "triton_smallhead", "triton_small"]:
        from .kernel.rwkv7_attn_triton import rwkv7_attn_triton
        w = -F.softplus(-w_lora_result) - 0.5
        xx, wkv_state_out = rwkv7_attn_triton(r, w, k, v, kk, iclr, s0=wkv_state_in, HEAD_SIZE=HEAD_SIZE)
    elif tmix_backend in ["triton_bighead", "triton_big"]:
        from .kernel.rwkv7_attn_triton import rwkv7_attn_triton_bighead
        w = -F.softplus(-w_lora_result) - 0.5
        xx, wkv_state_out = rwkv7_attn_triton_bighead(r, w, k, v, kk, iclr, s0=wkv_state_in, HEAD_SIZE=HEAD_SIZE)
    elif tmix_backend == "cuda_ref":
        # Cuda based method for rwkv attention
        from .kernel.rwkv7_attn_cuda import rwkv7_attn_cuda_ref
        # Reference cuda version (no state output)
        w = -F.softplus(-w_lora_result) - 0.5
        xx, wkv_state_out = rwkv7_attn_cuda_ref(r, w, k, v, kk, iclr, s0=wkv_state_in, HEAD_SIZE=HEAD_SIZE)
    elif tmix_backend == "cuda":
        # Cuda based method for rwkv attention
        from .kernel.rwkv7_attn_cuda import rwkv7_attn_cuda
        # Modified cuda version (with state output)
        w = -F.softplus(-w_lora_result) - 0.5
        xx, wkv_state_out = rwkv7_attn_cuda(r, w, k, v, kk, iclr, s0=wkv_state_in, HEAD_SIZE=HEAD_SIZE)
    elif tmix_backend == "fla":
        # FLA based method for rwkv attention
        from .kernel.rwkv7_attn_fla import rwkv7_attn_fla
        # FLA runs with the softplus w
        w = -F.softplus(-w_lora_result) - 0.5
        xx, wkv_state_out = rwkv7_attn_fla(r, w, k, v, kk, iclr, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xx, wkv_state_in) 
    elif tmix_backend == "fla_fused" or tmix_backend == "fused_fla":
        # FLA based method for rwkv attention
        from .kernel.rwkv7_attn_fla import rwkv7_attn_fused_reccurent_fla
        # FLA runs with the softplus w
        w = -F.softplus(-w_lora_result) - 0.5
        xx, wkv_state_out = rwkv7_attn_fused_reccurent_fla(r, w, k, v, kk, iclr, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xx, wkv_state_in) 
    else:
        raise ValueError(f"Unknown tmix_backend: {tmix_backend}")
    return xx.to(dtype=xx_dtype), wkv_state_out.to(dtype=wkv_state_in.dtype)
