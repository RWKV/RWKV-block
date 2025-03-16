# ---
# Collection of optimized operations used within the RWKV v5/v6 implementation.
# ---

import torch, sys, importlib
import torch.nn.functional as F
from torch import Tensor

def modified_lerp(start_mul, start, weight):
    '''
    Modified LERP operation, which is used to compute the 
    time mixing and channel mixing components. 

    This is slightly different from the standard LERP operation
    due to the presence of the start_mul parameter.
    '''
    return start_mul * start + weight * (1 - start)

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
# _has_triton = has_python_package("triton")
_has_fla = has_python_package("fla")
_has_cuda = torch.cuda.is_available()

# ---
# RWKVx060_chunk implementation, with support for different backends
# ---

def RWKVx060_chunk(
        # Inbound request tensors
        r:Tensor,k:Tensor,v:Tensor,w:Tensor,u:Tensor,wkv_state:Tensor,
        # Operator backend type to use
        backend:str='auto'
    )->tuple[Tensor,Tensor]:
    '''
    Highly optimized RWKV inner opperations, used within the TimeMix module
    With support for different implementations (torch, fla, etc)
    '''
    if backend == 'auto':
        if r.device.type == 'cpu' or r.device.type == 'mps':
            backend = 'pytorch'
        elif _has_fla == True:
            backend = 'fla'
        else:
            backend = 'pytorch'

    # Set the default device, if not pytorch
    # this mitigates with mismatched default device issues
    # known to occur with triton, fla, and cuda
    # 
    # This unfortunately will not work with single thread
    # multi-gpu setups. Sadly
    if backend == "pytorch" or backend == "torch":
        pass
    else:
        device = r.device
        if torch.get_default_device() != device:
            torch.set_default_device(device)
            torch.cuda.set_device(device)

    # Reshape the tensors to the correct format
    with torch.device(device):
        if backend == 'fla':
            x, s = RWKVx060_chunk_fla(r.float(),k.float(),v.float(),w.float(),u.float(),wkv_state.float())
            return x.bfloat16(), s.bfloat16()
        elif backend == "pytorch" or backend == "torch":
            return RWKVx060_chunk_torch(r,k,v,w,u,wkv_state)
        else:
            raise ValueError(f"Unsupported backend type: {backend}")

def RWKVx060_reshape_run(
        # Request shapes
        B:int, T:int, C:int, H:int, K:int,
        # Inbound request tensors
        r:Tensor,k:Tensor,v:Tensor,w:Tensor,u:Tensor,in_wkv_state:Tensor,
        # Operator backend type to use
        backend:str='auto'
    ):
    '''
    Highly optimized RWKV inner opperations, used within the TimeMix module
    When being passed while reshaping the format tensors 
    (with the B, T, C, H values: Batch size, token/time, C, H)
    '''
    if backend == 'auto':
        if r.device.type == 'cpu' or r.device.type == 'mps':
            backend = 'pytorch'
        elif _has_fla == True:
            backend = 'fla'
        else:
            backend = 'pytorch'

    # Set the default device, if not pytorch
    # this mitigates with mismatched default device issues
    # known to occur with triton, fla, and cuda
    # 
    # This unfortunately will not work with single thread
    # multi-gpu setups. Sadly
    if backend == "pytorch" or backend == "torch":
        pass
    else:
        device = r.device
        if torch.get_default_device() != device:
            torch.set_default_device(device)
            torch.cuda.set_device(device)

    # Reshape the tensors to the correct format
    with torch.device(device):
        # Reshape the tensors to the correct format, for the respective backend
        if backend == 'fla':
            r = r.view(B,T,H,-1).transpose(1,2).float()
            k = k.view(B,T,H,-1).transpose(1,2).float()
            v = v.view(B,T,H,-1).transpose(1,2).float()
            w = -torch.exp(w.view(B,T,H,-1).transpose(1,2).float())
            o, out_wkv_state = RWKVx060_chunk_fla(r, k, v, w, u=u.float(), wkv_state=in_wkv_state.float(), scale=1, output_final_state=True)
            return o.bfloat16().transpose(1,2).reshape(B,T,C), out_wkv_state.bfloat16()
        elif backend == 'pytorch' or backend == 'torch':
            u = u.view(1,H,1,K).to(r.dtype)
            r = r.view(B,T,H,K).transpose(1,2)
            k = k.view(B,T,H,K).transpose(1,2)
            v = v.view(B,T,H,K).transpose(1,2)
            w = -torch.exp(w.view(B,T,H,K).transpose(1,2))
            o, out_wkv_state = RWKVx060_chunk_torch(r, k, v, w, u, in_wkv_state)
            return o.transpose(1,2).reshape(B,T,C), out_wkv_state
        else:
            raise ValueError(f"Unsupported backend type: {backend}")


# ---
# RWKVx060_chunk pytorch backend
# ---

def RWKVx060_chunk_torch(
        # Inbound request tensors
        r:Tensor,k:Tensor,v:Tensor,w:Tensor,u:Tensor,wkv_state:Tensor
    )->tuple[Tensor,Tensor]:
    '''
    Highly optimized RWKV inner opperations, used within the TimeMix module
    This was made by @SmerkyG for supporting RWKV v5 in a pure pytorch manner
    And was able to match the cuda implementation in terms of performance (when compiled)
    '''
    B,H,L,K = k.size()

    # """
    # expects
    # r : (B,H,L,K)
    # k : (B,H,L,K)
    # v : (B,H,L,V)
    # w : (B,H,L,K) or (1,H,L,K)
    # u : (1,H,1,K)
    # wkv_state : (B,H,K,V)

    # outputs
    # out : (B,H,L,V)
    # """

    # Length management
    processed_len = 0
    remaining_len = L

    # State management
    nxt_wkv_state = wkv_state
    out_arr = []

    # Chunk sizes to loop, and support
    bChunkSize_arr = [128, 16, 2, 1]

    # Iterate the bChunk
    for bChunkSize in bChunkSize_arr:
        
        # Check if the remaining length is less than the chunk size
        while remaining_len >= bChunkSize:
            # Get the multiple of bChunkSize
            mul = remaining_len // bChunkSize
            chunk_len = bChunkSize * mul

            # Call the subchunk operation
            out, nxt_wkv_state = RWKVx060_subchunk_torch(
                r[:, :, processed_len:processed_len+chunk_len, :],
                k[:, :, processed_len:processed_len+chunk_len, :],
                v[:, :, processed_len:processed_len+chunk_len, :],
                w[:, :, processed_len:processed_len+chunk_len, :],
                u,
                nxt_wkv_state,
                chunk_len=bChunkSize
            )

            # Append the output to the out_arr
            out_arr.append(out)

            # Update the processed length and remaining length
            processed_len += chunk_len
            remaining_len -= chunk_len

    # Concatenate the out_arr, and return
    out = torch.cat(out_arr, dim=2)
    return out, nxt_wkv_state

def RWKVx060_subchunk_torch(
        # Inbound request tensors
        r:Tensor,k:Tensor,v:Tensor,w:Tensor,u:Tensor,wkv_state:Tensor,
        # Chunk size and precision
        chunk_len:int=128,precision:int=64
    )->tuple[Tensor,Tensor]:
    ''' 
    Highly optimized RWKV inner opperations, used within the TimeMix module
    This was made by @SmerkyG for supporting RWKV v5 in a pure pytorch manner
    And was able to match the cuda implementation in terms of performance (when compiled)

    The subchunk varient, requires the time component (L) to be an exact multiple of the chunk length
    And that the chunk length needs to be a multiple of 2
    '''
    B,H,L,K = k.size()
    return RWKVx060_subchunk_torch_inner(B,H,L,K, r,k,v,w,u,wkv_state,chunk_len,precision)

def RWKVx060_subchunk_torch_inner(
        # Inbound request shapes
        B:int,H:int,L:int,K:int, 
        # Inbound request tensors
        r:Tensor,k:Tensor,v:Tensor,w:Tensor,u:Tensor,wkv_state:Tensor,
        # Chunk size and precision
        chunk_len:int=128,precision:int=64
    )->tuple[Tensor,Tensor]:
    """
    expects
    r : (B,H,L,K)
    k : (B,H,L,K)
    v : (B,H,L,V)
    w : (B,H,L,K) or (1,H,L,K)
    u : (1,H,1,K)
    wkv_state : (B,H,K,V)

    outputs
    out : (B,H,L,V)
    """

    # # Log all the shapes for debugging
    # print(f"r.shape: {r.shape}")
    # print(f"k.shape: {k.shape}")
    # print(f"v.shape: {v.shape}")
    # print(f"w.shape: {w.shape}")
    # print(f"u.shape: {u.shape}")
    # print(f"wkv_state.shape: {wkv_state.shape}")

    # 24 is optimal chunk length for fp32 
    # (longer will use too much memory and cause precision problems or even numerical instability, shorter is inefficient)
    # otherwise fp64 is recommended instead
    assert(chunk_len <= 24 or precision == 64)

    B,H,L,K = k.size()
    V = v.size(-1)
    T = chunk_len

    if L == 1:
        kv = k.mT @ v
        out = r @ (wkv_state + u.mT * kv)
        wkv_state = w.mT * wkv_state + kv
        return out, wkv_state
    else:
        # FIXME - support fast path for non-exact multiples
        # ensure it's an exact multiple
        assert L%T == 0, "fast non-cuda rwkv5.2+ requires ctxlen to be an exact multiple of chunk_len"

        N = L // T

        # this has to be done to avoid numerical instability (inf/NaN) when w is used as a divisor up to chunk_length//2 places away (so precision_min_val^(T//2) has to be in fp range)
        # NOTE - this does not account for the impact of the size of R, K so we currently use the chunk_len=32 numbers for chunk_len=24
        assert(precision == 32 or precision == 64)
        precision_min_val = 0.005 # good for fp32 (1.175e-38 ^ (1/16.0) < 0.00426)
        if precision == 32:
            precision_dtype = torch.float32
        else: #elif precision_dtype == torch.float64:
            precision_dtype = torch.float64
        w = w.clamp(precision_min_val)

        # calculate cumulative decay in log space where it won't overflow
        w_log = w.float().log() # (1,H,L,K) or (B,H,L,K)

        # chunked view of w_log
        wc_log = w_log.view(w.size(0),H,N,T,K)
        wc_log_cum = wc_log.cumsum(dim=-2)

        # chunked view of shifted_w_log
        shifted_wc_log_cum = F.pad(wc_log_cum, (0, 0, 1, -1))

        # NOTE - we have to apply the decay weight from TWO ahead.. ONE ahead gets no decay (log==0)
        # pre-applied weights
        # left side is prior chunk (w_inter), right side is current chunk (w_intra)
        # without u...
        # w0   w1   w2   w3   | w4   w5   w6   w7          
        # w1:4 w2:4 w3:4 w4:4 | w4:5 w4:6 w4:7 w4:8
        # with u...
        # w0   w1   w2   w3   | w4   w5   w6   w7          
        # w1:4 w2:4 w3:4 w4:4 | w4:4 w4:5 w4:6 w4:7

        # ws decays the entire current state (representing t-1) to the prior block (t-2)
        ws = wc_log.sum(dim=-2, keepdim=True) # 1HN1K or BHN1K
        # w_inter is the decay to the end of the current block, since it will be applied at the next iteration when current (t) becomes prior (t-1)
        # this formula because e.g. w1:4 = w0:4 - w0:1
        w_inter = ws - wc_log_cum # 1HNTK or BHNTK (w^(T-1) ... w^0)
        # w_intra is the decay from the beginning of the current block (t), since it will be applied to current queries (t) against prior state (representing keys+values up to but not including block t)
        # this formula because e.g. w1:3 = w0:3 - w0
        w_intra = wc_log_cum - wc_log # 1HNTK or BHNTK (w^0 ... w^(T-2))

        ws = list(ws.mT.exp().to(r.dtype).unbind(dim=-3)) # N x 1HK1 or BHK1 !!NOTE THE .mT HERE!!
        w_inter = w_inter.exp().to(r.dtype) # 1HNTK or BHNTK
        w_intra = w_intra.exp().to(r.dtype) # 1HNTK or BHNTK

        # chunked view of r, k, v
        r = r.view(B,H,N,T,K) 
        k = k.view(B,H,N,T,K) 
        v = v.view(B,H,N,T,V)
        u = u.unsqueeze(2).to(r.dtype) # (1,H,1,1,K)

        # parallel calculation of all intra-chunk attention contributions
        wc_log_offset = shifted_wc_log_cum[...,T//2:T//2+1,:] # B,H,N,1,K
        r_decay = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype).exp() # B,H,N,T,K
        k_inv_decay = (wc_log_offset - wc_log_cum).to(precision_dtype).exp() # B,H,N,T,K
        a = ((r*r_decay) @ (k*k_inv_decay).mT).to(r.dtype).tril(-1) # B,H,N,T,T
        # add u term to attention (NOTE - the tril(-1) above zeroed the diagonal)
        a = a + torch.einsum('bhntk,bhntk->bhnt', r, u * k).diag_embed()
        out = a @ v # BHNTV
        # alternate way of adding in u
        # out = out + torch.einsum('bhntk,bhntk,bhntv->bhntv', r, u * k, v) 

        # parallel precalculation of chunked (k*wk).mT@v for use in recurrent state calc below
        wkv = (k * w_inter).mT @ v # BHNKV
        wkv = list(wkv.unbind(dim=-3)) # N x BHKV

        # recurrent calculation of all states
        states = []
        for i in range(N):
            states.append(wkv_state)
            wkv_state = wkv_state * ws[i] + wkv[i] # BHKV
            # equivalent non-precalced version
            #wkv = (k[...,i,:,:] * wk[...,i,:,:]).mT @ v[...,i,:,:]
            #wkv_state = wkv_state * ws[i] + wkv
        states = torch.stack(states, dim=2) # BHNKV       

        # parallel application of all r to states
        out = out + (r * w_intra) @ states # BHNTV
        out = out.view(B,H,L,V)

        # # Log the output shapes fpr debugging
        # print(f"out.shape: {out.shape}")
        # print(f"(out) wkv_state.shape: {wkv_state.shape}")

        return out, wkv_state

# ---
# RWKVx060_chunk FLA backend
# ---

# The empty fla_chunk_rwkv6 operator cache
global _RWKVx060_chunk_fla_operator
_RWKVx060_chunk_fla_operator = None

@torch.compiler.disable
def RWKVx060_chunk_fla(
        # Inbound request tensors
        r:Tensor,k:Tensor,v:Tensor,w:Tensor,u:Tensor,wkv_state:Tensor,
        # Optional parameters
        scale=1,output_final_state=True
    ):
    '''
    Run the RWKVx060 chunk operation.
    Note this is currently not pytorch compiler friendly sadly
    '''
    global _RWKVx060_chunk_fla_operator
    if _RWKVx060_chunk_fla_operator is None:
        from fla.ops.rwkv6 import chunk_rwkv6
        _RWKVx060_chunk_fla_operator = chunk_rwkv6

    return _RWKVx060_chunk_fla_operator(r, k, v, w, u=u, scale=scale, initial_state=wkv_state, output_final_state=output_final_state)
