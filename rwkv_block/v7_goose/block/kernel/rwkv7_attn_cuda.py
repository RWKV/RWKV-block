import torch, os, time
from .rwkv7_attn_pytorch import rwkv7_attn_pytorch_chunk, rwkv7_attn_pytorch_ref_fp32

####################################################################################################
# Stateless reference implementation
####################################################################################################

def load_ref_wkv_cuda_kernel(CHUNK_LEN = 16, HEAD_SIZE = 64):
    from torch.utils.cpp_extension import load

    # load_name = f"wind_backstepping_C{HEAD_SIZE}_L{CHUNK_LEN}"
    load_name = "wind_backstepping"
    load_file = "wkv7"

    # Check if the load_name is already loaded
    if load_name in torch.ops:
        if torch.ops.wind_backstepping._head_size != HEAD_SIZE:
            raise ValueError(f"Loaded kernel has different head size {torch.ops.wind_backstepping._head_size} than expected {HEAD_SIZE} - we currently do not support multiple head sizes in the same process")
        return torch.ops.wind_backstepping

    # Logging of warning usage for reference implementation
    print("[WARNING] Reference CUDA kernel does not support input RWKV state, and is used only for training/validaiton purposes")
    
    # Get the this script file path, to cmpute the cuda path
    this_file_path = os.path.dirname(os.path.abspath(__file__))

    # # Get the device compute capability
    # cuda_device = torch.cuda.current_device()
    # compute_capability = torch.cuda.get_device_capability(cuda_device)
    # compute_capability_str = f"{compute_capability[0]}{compute_capability[1]}"
    # print("[INFO] Using compute capability:", compute_capability_str)

    # Load the kernel, there is some wierd edge condition in compilation,
    # that try catching.... and trying again.... sometimes work?
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"] # 
    try:
        load(name=load_name, sources=[f'{this_file_path}/cuda/{load_file}_cuda.cu', f'{this_file_path}/cuda/{load_file}_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    except Exception as e:
        print("[WARNING] Failed to load the kernel, trying again (sometimes the compiler has wierd race condition)...")
        time.sleep(2) # Somehow this works, with minor compilation error, that passes on subsequent reruns
        load(name=load_name, sources=[f'{this_file_path}/cuda/{load_file}_cuda.cu', f'{this_file_path}/cuda/{load_file}_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    # Save the head size (for error checking)
    torch.ops.wind_backstepping._head_size = HEAD_SIZE

    # Return the loaded kernel
    return torch.ops.wind_backstepping

@torch.compiler.disable()
def ref_wkv_cuda_forward(w,q,k,v,z,b, y,s,sa):
    torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)

@torch.compiler.disable()
def ref_wkv_cuda_backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db):
    torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)

class RefCudaWindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        CHUNK_LEN=16
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        ref_wkv_cuda_forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        ref_wkv_cuda_backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

@torch.compiler.disable()
def rwkv7_attn_cuda_ref(q,w,k,v, kk,iclr, HEAD_SIZE=64, s0=None):
    q,w,k,v, kk,iclr = [i.bfloat16().contiguous() for i in [q,w,k,v,kk,iclr]]
    return rwkv7_attn_cuda_ref_no_compile(q,w,k,v, kk,iclr, HEAD_SIZE, s0)

@torch.compiler.disable()
def rwkv7_attn_cuda_ref_no_compile(q,w,k,v, kk,iclr, HEAD_SIZE=64, s0=None):
    # Preload the kernel
    load_ref_wkv_cuda_kernel()

    # Get the shape
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C

    # Assert that the chunk is multiple of 16
    assert T % 16 == 0, 'reference cuda, only works in multiple of 16'

    # Initialize the state, if not provided - for compatibility (THE STATE IS NOT UPDATED)
    s0 = torch.zeros(B,H,C,C, dtype=torch.float,device=w.device) if s0 is None else s0
    s0 = s0.contiguous()
    
    # Handling the cuda kernel
    q,w,k,v,a,b = [i.view(B,T,H,C).contiguous() for i in [q,w,k,v,(-kk),(kk*iclr)]]

    # Forward with backprop
    xx = RefCudaWindBackstepping.apply(w,q,k,v,a,b)
    return xx.view(B,T,HC), s0.view(B,H,C,C)

####################################################################################################
# State based cuda code
####################################################################################################

def load_wkv_cuda_kernel(CHUNK_LEN = 16, HEAD_SIZE = 64):
    from torch.utils.cpp_extension import load

    # load_name = f"wind_backstepping_C{HEAD_SIZE}_L{CHUNK_LEN}"
    load_name = "state_wind_backstepping"
    load_file = "state_wkv7"

    # Check if the load_name is already loaded
    if load_name in torch.ops:
        if torch.ops.state_wind_backstepping._head_size != HEAD_SIZE:
            raise ValueError(f"Loaded kernel has different head size {torch.ops.state_wind_backstepping._head_size} than expected {HEAD_SIZE} - we currently do not support multiple head sizes in the same process")
        return torch.ops.state_wind_backstepping
    
    # Get the this script file path, to cmpute the cuda path
    this_file_path = os.path.dirname(os.path.abspath(__file__))

    # Load the kernel, there is some wierd edge condition in compilation,
    # that try catching.... and trying again.... sometimes work?
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"] # 
    try:
        load(name=load_name, sources=[f'{this_file_path}/cuda/{load_file}_cuda.cu', f'{this_file_path}/cuda/{load_file}_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    except Exception as e:
        print("[WARNING] Failed to load the kernel, trying again (sometimes the compiler has wierd race condition)...")
        time.sleep(2) # Somehow this works, with minor compilation error, that passes on subsequent reruns
        load(name=load_name, sources=[f'{this_file_path}/cuda/{load_file}_cuda.cu', f'{this_file_path}/cuda/{load_file}_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    # Save the head size (for error checking)
    torch.ops.state_wind_backstepping._head_size = HEAD_SIZE

    # Return the loaded kernel
    return torch.ops.state_wind_backstepping

@torch.compiler.disable()
def wkv_cuda_forward(state, w,q,k,v,z,b, y,s,sa):
    torch.ops.state_wind_backstepping.forward(state, w,q,k,v,z,b, y,s,sa)

@torch.compiler.disable()
def wkv_cuda_backward(state, w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db):
    torch.ops.state_wind_backstepping.backward(state, w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)

class CudaWindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s0, w,q,k,v,z,b):
        CHUNK_LEN=16
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        sOri = s0.clone()
        wkv_cuda_forward(s0, w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(sOri, w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        state,w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dS0,dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [state,w,q,k,v,z,b]]
        wkv_cuda_backward(state, w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dS0,dw,dq,dk,dv,dz,db

def rwkv7_attn_cuda(r,w,k,v, kk,iclr, HEAD_SIZE=64, s0=None):
    r,w,k,v, kk,iclr = [i.bfloat16().contiguous() for i in [r,w,k,v,kk,iclr]]
    s0 = s0.float().contiguous() if s0 is not None else None
    return rwkv7_attn_cuda_no_compile(r,w,k,v, kk,iclr, HEAD_SIZE, s0)

@torch.compiler.disable()
def rwkv7_attn_cuda_no_compile(r,w,k,v, kk,iclr, HEAD_SIZE=64, s0=None):
    # Preload the kernel
    load_wkv_cuda_kernel(HEAD_SIZE=HEAD_SIZE)

    # Get the shape
    B,T,HC = w.shape

    # Check if the chunk is multiple of 16
    chunk_remainder = T % 16

    # Initialize the state
    C = HEAD_SIZE
    H = HC//C

    # Initialize the state
    s0 = torch.zeros(B,H,C,C, dtype=torch.float,device=w.device) if s0 is None else s0
    sT = s0.to(dtype=torch.float).contiguous()

    # If its smaller then a chunk, use the pytorch implementation
    if T < 16: 
        chunk_xx, chunk_sT = rwkv7_attn_pytorch_chunk(
            r,(-w.float().exp()).exp(),
            k,v, 
            kk,iclr, 
            B, T, H, C, 
            torch.empty(B, T, HC, device=w.device, dtype=w.dtype),
            sT
        )
        # chunk_xx, chunk_sT = rwkv7_attn_pytorch_ref_fp32(
        #     r,w,k,v, 
        #     kk,iclr, 
        #     B, T, H, C, 
        #     torch.empty(B, T, HC, device=w.device, dtype=w.dtype),
        #     sT
        # )
        return chunk_xx, chunk_sT.to(dtype=s0.dtype)

    # Optimize the call, if chunk is multiple of 16
    if chunk_remainder == 0:
        chunk_xx, chunk_sT = rwkv7_attn_cuda_chunk(r,w,k,v, kk,iclr, HEAD_SIZE, sT)
        return chunk_xx, chunk_sT.to(dtype=s0.dtype)

    # Compute the number of chunks
    chunks = T // 16
    si = chunks * 16

    # Get the chunked output
    chunk_xx, chunk_sT = rwkv7_attn_cuda_chunk(
        r[:,:si],w[:,:si],k[:,:si],v[:,:si], kk[:,:si],iclr[:,:si],
        HEAD_SIZE, s0
    )

    # Get the remainder
    # ---
    remain_xx, last_sT = rwkv7_attn_pytorch_chunk(
        r[:,si:],(-w[:,si:].float().exp()).exp(),
        k[:,si:],v[:,si:], 
        kk[:,si:],iclr[:,si:], 
        B, chunk_remainder, H, C, 
        torch.zeros(B, chunk_remainder, HC, device=w.device, dtype=w.dtype), 
        chunk_sT
    ) 
    # remain_xx, last_sT = rwkv7_attn_pytorch_ref_fp32(
    #     r[:,si:],w[:,si:],k[:,si:],v[:,si:], 
    #     kk[:,si:],iclr[:,si:], 
    #     B, chunk_remainder, H, C, 
    #     torch.empty(B, chunk_remainder, HC, device=w.device, dtype=w.dtype), 
    #     chunk_sT
    # )

    # Concatenate and return results
    return torch.cat([chunk_xx.to(dtype=w.dtype), remain_xx.to(dtype=w.dtype)], dim=1), last_sT.to(dtype=s0.dtype)


def rwkv7_attn_cuda_chunk(r,w,k,v, kk,iclr, HEAD_SIZE=64, s0=None):
    '''
    Triton implementation running in blocks of 16 (hardcoded requirement for the kernel)
    '''
    B,T,HC = w.shape
    assert T % 16 == 0, 'pure cuda, only works in multiple of 16'
    C = HEAD_SIZE
    H = HC//C

    # Handling the cuda kernel
    a,b = -kk, (kk.to(dtype=iclr.dtype)*iclr)
    r,w,k,v,a,b = [i.view(B,T,H,C).bfloat16().contiguous() for i in [r,w,k,v,a,b]]

    if s0 is None:
        s1 = torch.zeros(B,H,C,C, dtype=torch.float,device=w.device).contiguous()
    else:
        s1 = s0.float().clone().contiguous()

    # Forward with backprop
    xx = CudaWindBackstepping.apply(s1,w,r,k,v,a,b)
    return xx.view(B,T,HC), s1.view(B,H,C,C)