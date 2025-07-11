def rwkv7_attn_fla(
    r,w,k,v, kk,iclr, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in,
    output_final_state=True
):
    from fla.ops.rwkv7.chunk import chunk_rwkv7

    # Preprocessing the FLA
    r,w,k,v,a,b = [i.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1) for i in [r,w,k,v,-kk,(kk*iclr)]]
    log_w = -w.float().exp()

    # Run the FLA
    output, vk_state = chunk_rwkv7(r, log_w, k, v, a, b, initial_state=wkv_state_in.float(), output_final_state=output_final_state)
    if output_final_state:
        return output, vk_state.to(dtype=wkv_state_in.dtype)
    else:
        return output, None

def rwkv7_attn_fused_reccurent_fla(
    r,w,k,v, kk,iclr, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in,
    output_final_state=True
):
    from fla.ops.rwkv7.fused_recurrent import fused_recurrent_rwkv7

    # Preprocessing the FLA
    r,w,k,v,a,b = [i.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1) for i in [r,w,k,v,-kk,(kk*iclr)]]
    log_w = -w.float().exp()

    # Run the FLA
    output, vk_state = fused_recurrent_rwkv7(r, log_w, k, v, a, b, initial_state=wkv_state_in.float(), output_final_state=output_final_state)
    if output_final_state:
        return output, vk_state.to(dtype=wkv_state_in.dtype)
    else:
        return output, None