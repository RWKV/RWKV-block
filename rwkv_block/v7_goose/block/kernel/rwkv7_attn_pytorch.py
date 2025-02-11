import torch

# Enable tensorfloat 32 
torch.set_float32_matmul_precision('high')

# Handles the RWKV v7 attention mechanic, in pure pytorch
def rwkv7_attn_pytorch(
    r,w,k,v, kk,a, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in
):

    ### Reference implement
    # return rwkv7_attn_pytorch_ref(
    #     r,w,k,v, kk,a,
    #     BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    #     xx, wkv_state_in
    # )
    ###

    # # This works, but it has too much of a vram overhead
    ###
    # return rwkv7_attn_pytorch_v2_chunk_w_compile_break(
    #     r,w,k,v, kk,a,
    #     BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    #     xx, wkv_state_in
    # )
    ###

    # > per 9k chunk, per block, on a 4090 ...
    # > with forward_with_reduce_compile on the timemix ...
    #
    # Somehow...
    # The reference implement takes: 2281ms
    # The chunked version takes:     389ms  (chunksize 256)

    # Get the shape
    B,T,HC = w.shape

    # Compute the chunks
    chunk_size = 256
    chunk_count = SEQ_LEN // chunk_size
    chunk_remainder = SEQ_LEN % chunk_size

    # The wkv_state_out
    wkv_state_out = wkv_state_in.float()

    # # List of tensor to build
    # xlist = []
    xx = xx.clone()

    # Loop over the chunks
    for i in range(chunk_count):
        sta = i * chunk_size
        end = sta + chunk_size

        xx[:,sta:end], wkv_state_out = rwkv7_attn_pytorch_v2_chunk_w_compile_break(
        # xpart, wkv_state_out = rwkv7_attn_pytorch_chunk_with_w_compile_break(
            r[:,sta:end],w[:,sta:end],k[:,sta:end],v[:,sta:end], 
            kk[:,sta:end],a[:,sta:end],
            BATCH_SIZE, chunk_size, N_HEAD, HEAD_SIZE,
            # xx[:,sta:end], wkv_state_out
            torch.zeros(B,chunk_size,HC, dtype=xx.dtype, device=xx.device), wkv_state_out
        )
        # xlist.append(xpart)

    # Handle the remainder
    if chunk_remainder > 0:
        sta = chunk_count * chunk_size
        end = sta + chunk_remainder

        xx[:,sta:end], wkv_state_out = rwkv7_attn_pytorch_v2_chunk_w_compile_break(
        # xpart, wkv_state_out = rwkv7_attn_pytorch_chunk_with_w_compile_break(
            r[:,sta:end],w[:,sta:end],k[:,sta:end],v[:,sta:end], 
            kk[:,sta:end],a[:,sta:end],
            BATCH_SIZE, chunk_remainder, N_HEAD, HEAD_SIZE,
            # xx[:,sta:end], wkv_state_out,
            torch.zeros(B,chunk_remainder,HC, dtype=xx.dtype, device=xx.device), wkv_state_out,
            # offset=0, chunk_size=chunk_remainder
        )
        # xlist.append(xpart)

    # # Concatenate the list
    # xx = torch_cat_no_compiler(xlist, dim=1)

    # Return the output
    return xx, wkv_state_out.to(dtype=wkv_state_in.dtype)

####################################################################################################
# Working reference copy, that has been validated to be "identical" to the reference implementation
# However this has known pytorch compilation issues, hence the modified chunk wise version is used 
# instead for an approximate 5x speed up
####################################################################################################
@torch.compiler.disable()
def rwkv7_attn_pytorch_ref(
    r,w,k,v, kk,a, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in
):
    ######## pure pytorch method
    # See: https://github.com/BlinkDL/RWKV-LM/blob/d4c42b2cac10f8f3896ce153e2310dc763662b7a/RWKV-v7/rwkv_v7_demo_fast.py#L238
    ########
    vk_state = wkv_state_in.float()
    for t in range(SEQ_LEN):
        r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
        vk = v_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ k_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        ab = (-kk_).view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ (kk_*a_).view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        vk_state = (vk_state * w_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE).float() + vk_state @ ab.float() + vk.float())
        xx[:,t] = ((vk_state.to(dtype=xx.dtype) @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,N_HEAD*HEAD_SIZE))
    wkv_state_out = vk_state.to(dtype=wkv_state_in.dtype)
    return xx, wkv_state_out
####################################################################################################

####################################################################################################
# Modified reference computation done in fp32, 
# with changes made to bring the result closer to the cuda kernel
####################################################################################################
@torch.compiler.disable()
def rwkv7_attn_pytorch_ref_fp32(
    r,w,k,v, kk, iclr, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in
):
    ######## pure pytorch method (modified for fp32)
    # See: https://github.com/BlinkDL/RWKV-LM/blob/d4c42b2cac10f8f3896ce153e2310dc763662b7a/RWKV-v7/rwkv_v7_demo_fast.py#L238
    ########
    w = (-w.float().exp()).exp()

    # wkv_state_in = torch.zeros(BATCH_SIZE,N_HEAD,HEAD_SIZE,HEAD_SIZE, dtype=torch.float,device=w.device)
    vk_state = wkv_state_in.float()

    a = -kk
    b = kk * iclr

    for t in range(SEQ_LEN):
        r_, w_, k_, v_, a_, b_= r[:,t].float(), w[:,t].float(), k[:,t].float(), v[:,t].float(), a[:,t].float(), b[:,t].float()
        # ab = (-kk_).view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ (kk_*a_).view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        vk = v_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ k_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        vk_state = (vk_state * w_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE).float() + vk_state @ a_.float().view(BATCH_SIZE, N_HEAD,HEAD_SIZE,1) @ b_.view(BATCH_SIZE, N_HEAD,1,HEAD_SIZE) + vk.float())
        xx[:,t] = ((vk_state @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,N_HEAD*HEAD_SIZE)).to(dtype=xx.dtype)

    wkv_state_out = vk_state.to(dtype=wkv_state_in.dtype)
    return xx, wkv_state_out
####################################################################################################

def rwkv7_attn_pytorch_chunk(
    r,w,k,v, kk,a, 
    BATCH_SIZE, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in,
    offset=0, chunk_size=16
):
    '''
    Chunked version of the RWKV7 attention, for better performance. 
    If the chunk size is less then 128, this is generally compilable

    This is used by the triton/cuda implement, for the remaining % 16 chunks
    '''
    ######## pure pytorch method
    # See: https://github.com/BlinkDL/RWKV-LM/blob/d4c42b2cac10f8f3896ce153e2310dc763662b7a/RWKV-v7/rwkv_v7_demo_fast.py#L238
    ########
    vk_state = wkv_state_in.float()
    for i in range(chunk_size):
        t = offset + i
        r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
        vk = v_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ k_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        ab = (-kk_).view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ (kk_*a_).view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        vk_state = (vk_state * w_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE).float() + vk_state @ ab.float() + vk.float())
        xx[:,t] = (vk_state @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1).float()).view(BATCH_SIZE,N_HEAD*HEAD_SIZE).to(dtype=xx.dtype)
    wkv_state_out = vk_state.to(dtype=wkv_state_in.dtype)
    return xx, wkv_state_out


def rwkv7_attn_pytorch_v2_chunk_w_compile_break(
    r,w,k,v, kk,a, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in
):
    '''
    Chunked version of the RWKV7 attention, for better performance
    '''
    full_vk_ = v.view(BATCH_SIZE,SEQ_LEN,N_HEAD, HEAD_SIZE,1) @ k.view(BATCH_SIZE,SEQ_LEN,N_HEAD, 1,HEAD_SIZE)
    full_iclr_ = (kk * a).view(BATCH_SIZE,SEQ_LEN,N_HEAD,1,HEAD_SIZE)
    full_ab = (-kk).view(BATCH_SIZE,SEQ_LEN,N_HEAD, HEAD_SIZE,1) @ full_iclr_

    wkv_xx = torch.empty(BATCH_SIZE,SEQ_LEN,N_HEAD,HEAD_SIZE,HEAD_SIZE, dtype=xx.dtype, device=xx.device)
    wkv_xx, wkv_state_out = rwkv7_attn_pytorch_v2_inner_w_compile_break(
        r,w,
        full_vk_, full_ab, 
        BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
        wkv_xx, wkv_state_in
        # xx, wkv_state_in
    )

    # if BATCH_SIZE != 1:
    #     print("BATCH_SIZE != 1 : ", BATCH_SIZE)
    # if SEQ_LEN != 256:
    #     print("SEQ_LEN != 256 : ", SEQ_LEN)

    # xx[:,t] = ((wkv_state.to(dtype=xx.dtype) @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,N_HEAD*HEAD_SIZE))
    xx[:] = (wkv_xx.to(dtype=xx.dtype) @ r.view(BATCH_SIZE,SEQ_LEN,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,SEQ_LEN,N_HEAD*HEAD_SIZE)

    return xx, wkv_state_out

@torch.compiler.disable()
def rwkv7_attn_pytorch_v2_inner_w_compile_break(
    r, w,
    full_vk_, full_ab, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in
):
    '''
    Isolated sub-function with no compilation
    '''
    return rwkv7_attn_pytorch_v2_inner_jit(
        r, w,
        full_vk_, full_ab,
        BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
        xx, wkv_state_in
    )

# @torch.compile(fullgraph=True)
@torch.jit.script
def rwkv7_attn_pytorch_v2_inner_jit(
    r, w,
    full_vk_, full_ab, 
    BATCH_SIZE:int, SEQ_LEN:int, N_HEAD:int, HEAD_SIZE:int,
    wkv_xx, wkv_state_in
):
    '''
    Isolated sub-function with JIT
    '''
    # wkv_xx = torch.zeros(BATCH_SIZE,SEQ_LEN,N_HEAD,HEAD_SIZE,HEAD_SIZE, dtype=xx.dtype,device=xx.device)
    # wkv_state_in = torch.zeros(BATCH_SIZE,N_HEAD,HEAD_SIZE,HEAD_SIZE, dtype=torch.float,device=w.device)
    wkv_state = wkv_state_in
    for t in range(SEQ_LEN):
        # r_ = r[:,t]
        # w_ = w[:,t]
        # vk = full_vk_[:,t].view(BATCH_SIZE,N_HEAD,HEAD_SIZE,HEAD_SIZE)
        # ab = full_ab[:,t].view(BATCH_SIZE,N_HEAD,HEAD_SIZE,HEAD_SIZE)

        wkv_state = (wkv_state * w[:,t].view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE).float() + wkv_state @ full_ab[:,t].view(BATCH_SIZE,N_HEAD,HEAD_SIZE,HEAD_SIZE).float() + full_vk_[:,t].view(BATCH_SIZE,N_HEAD,HEAD_SIZE,HEAD_SIZE).float()).clone()
        wkv_xx[:,t] = wkv_state.to(dtype=w.dtype)
    return wkv_xx, wkv_state
    #     xx[:,t] = ((wkv_state.to(dtype=xx.dtype) @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,N_HEAD*HEAD_SIZE))
    # return xx, wkv_state
    