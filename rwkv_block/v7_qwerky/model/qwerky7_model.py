import torch, math
from torch import nn
from torch import Tensor
from typing import Union

from .qwerky7_config_map import Qwerky7ConfigMap
from ..block.qwerky7_layer_block import Qwerky7LayerBlock

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer, Qwen2RotaryEmbedding

class Qwerky7Model(nn.Module):
    '''
    Qwerky7 Model architecture
    Simplified implementation

    Note: This EXCLUDES the head layer, keeping in line with the HF format convention
    '''

    def __init__(self, config: Union[Qwerky7ConfigMap, any, None] = None):
        # Initialize the base class
        super().__init__()

        # Normalize the config
        configMap:Qwerky7ConfigMap = Qwerky7ConfigMap.normalize(config)
        self.configMap = configMap
        config = configMap
        
        # Get the required prop
        num_hidden_layers = configMap.num_hidden_layers
        vocab_size = configMap.vocab_size
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')
        hidden_size = configMap.hidden_size
        padding_idx = configMap.padding_idx
        head_size = configMap.head_size

        # Checkpoint function hook
        self.checkpoint_function = None
        # Linear module function
        # This is used to replace the linear module, with a custom implementation
        # Might be requried to work around some known deepspeed 3 issues
        self.linear_module_function = None

        # The following default device overwrite, is to speed up qwen related module initialization
        default_device = torch.get_default_device()
        default_dtype = torch.get_default_dtype()
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)

        with torch.device(device):
            # Embedding layer
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx, dtype=dtype)

            # Initialize rotary embeddings, which is used for all layers (both rwkv and qwerky)
            if configMap.use_rotary_pos_emb:
                self.rotary_emb = Qwen2RotaryEmbedding(config=config.hybrid_layer_config())

            # main layers
            self.layers = nn.ModuleList([
                # Prefix hybrid layers
                *[Qwen2DecoderLayer(config.hybrid_layer_config(), offset).bfloat16() for offset in range(config.num_prefix_hybrid_layers)],
                # Qwerky layers
                *[Qwerky7LayerBlock(config.new_block_config_map(layer_id=layer_idx)) for layer_idx in range(config.num_prefix_hybrid_layers, config.num_prefix_hybrid_layers + config.num_qwerky_layers())],
                # Suffix hybrid layers
                *[Qwen2DecoderLayer(config.hybrid_layer_config(), config.num_prefix_hybrid_layers + config.num_qwerky_layers() + offset).bfloat16() for offset in range(config.num_suffix_hybrid_layers)]
            ])

            # ln_out
            self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)
            # self.norm.weight = nn.Parameter(torch.ones(hidden_size).bfloat16().to(device)) # Annoying HF meta device bug workaround

            # init state tuning support (only for Qwerky layers, excluding hybrid layers)
            if configMap.init_state_wkv:
                n_qwerky_layers = configMap.num_qwerky_layers()
                stateTuneList = [None]*n_qwerky_layers
                for i in range(n_qwerky_layers):
                    stateTuneList[i] = nn.ParameterDict({
                        "wkv": nn.Parameter(torch.zeros(hidden_size // head_size, head_size, head_size, dtype=torch.float)),
                    })
                self.init_state = nn.ParameterList(stateTuneList)

        # Reset the default device and dtype
        torch.set_default_device(default_device)
        torch.set_default_dtype(default_dtype)

    def reset_parameters(self):
        '''
        Reset the parameters of the model, to an initial state used for training a model from scratch
        '''
        # Get the required prop
        configMap = self.configMap
        num_hidden_layers = configMap.num_hidden_layers
        vocab_size = configMap.vocab_size
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')
        hidden_size = configMap.hidden_size
        head_size = configMap.head_size
        
        # With device
        with torch.device(device):
            # Iterate and reset the layers
            for i in range(num_hidden_layers):
                if hasattr(self.layers[i], 'reset_parameters'):
                    self.layers[i].reset_parameters()

            # Reinit the Embedding layer
            self.embed_tokens.reset_parameters()

            # Reinit the RMSNorm
            self.norm.weight.data.fill_(1.0)

            # Reinit the init state tuning support (only for Qwerky layers)
            if configMap.init_state_wkv:
                n_qwerky_layers = configMap.num_qwerky_layers()
                if self.init_state is None:
                    stateTuneList = [None]*n_qwerky_layers
                    for i in range(n_qwerky_layers):
                        stateTuneList[i] = nn.ParameterDict({
                            "wkv": nn.Parameter(torch.zeros(hidden_size // head_size, head_size, head_size, dtype=torch.float)),
                        })
                    self.init_state = nn.ParameterList(stateTuneList)
                else:
                    for i in range(n_qwerky_layers):
                        self.init_state[i]["wkv"].data.copy_(torch.zeros(hidden_size // head_size, head_size, head_size, dtype=torch.float))


    def load_from_model_state_dict(self, state_dict: dict, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the RWKV_TimeMix model weights, using the layer_id
        '''
        for i, layers in enumerate(self.layers):
            layers.load_from_model_state_dict(state_dict, i, non_blocking=non_blocking)

        self.embed_tokens.weight.data.copy_(state_dict['model.embed_tokens.weight'], non_blocking=non_blocking)
        self.norm.weight.data.copy_(state_dict['model.norm.weight'], non_blocking=non_blocking)
        
        if self.configMap.init_state_wkv:
            n_qwerky_layers = self.configMap.num_qwerky_layers()
            for i in range(n_qwerky_layers):
                if 'model.init_state.'+str(i)+'.wkv' in state_dict:
                    self.init_state[i]["wkv"].data.copy_(state_dict['model.init_state.'+str(i)+'.wkv'], non_blocking=True)

    ### ---
    ###
    ### Init state handling
    ###
    ### ---

    def get_init_state(self, batch_size:int=1, skip_init_state:bool=False) -> list[torch.Tensor]:
        '''
        Get an initialized copy of the model state, for the given batch size
        '''
        # Get required configs
        hidden_size = self.configMap.hidden_size
        init_state_wkv = self.configMap.init_state_wkv
        n_qwerky_layers = self.configMap.num_qwerky_layers()
        head_size = self.configMap.head_size

        # Prepare the initial state (only for Qwerky layers)
        init_state = [ None for i in range(n_qwerky_layers) ]
        qwerky_start = self.configMap.num_prefix_hybrid_layers
        for i in range(n_qwerky_layers):
            device = self.layers[qwerky_start + i].self_attn.q_proj.weight.data.device

            # Use the saved init_state if enabled
            # TODO: Consider letting the wkv_state dtype be a parameter
            wkv_state = torch.zeros(batch_size, hidden_size // head_size, head_size, head_size, device=device, dtype=torch.float)
            if init_state_wkv and skip_init_state == False:
                init_wkv = self.init_state[i]["wkv"]
                for b in range(batch_size):
                    wkv_state[b][:] = init_wkv

            # Prepare the state
            init_state[i] = wkv_state
        return init_state

    ### ---
    ###
    ### Custom hook overwrites
    ###
    ### ---

    def setup_linear_operation(self, linear_module_function):
        '''
        Configure the linear operation function, to be used by the model
        '''
        self.linear_module_function = linear_module_function
        for layer in self.layers:
            if hasattr(layer, 'linear_module_function'):
                layer.linear_module_function = linear_module_function
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.linear_module_function = linear_module_function

    def setup_checkpoint_function(self, checkpoint_function):
        '''
        Configure the checkpoint function, to be used by the model
        '''
        self.checkpoint_function = checkpoint_function

    ### ---
    ###
    ### Model Forward
    ###
    ### ---

    def forward(
        self, idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor] = None,  
        ret_stateList:list[torch.Tensor] = None,
        position_ids:torch.Tensor = None,
        overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Forward the layer set, given the input tokens and the last state
        Last state is a list of time mix wkv state

        Returns a pair of the output embedding and the next state
        '''
        # If no return state is set, let _forward_internal, set it up
        if ret_stateList is None:
            ret_stateList = [ None for i in range(self.configMap.num_qwerky_layers()) ]
            return self._forward_internal(idx, prv_stateList, ret_stateList, position_ids=position_ids, overwrite_ret_tensor=False)

        # Forward internally
        return self._forward_internal(idx, prv_stateList, ret_stateList, position_ids=position_ids, overwrite_ret_tensor=overwrite_ret_tensor)
    
    def _forward_internal_embeddings(
            self, x_hidden_state:torch.Tensor, 
            prv_stateList:list[torch.Tensor],  
            ret_stateList:list[torch.Tensor],
            position_ids:torch.Tensor = None,
            overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Internal forward operations, which assumes the state is already initialized.
        And uses the x_hidden_state as the input (bypassing the embedding layer)
        And returns the output embedding before the head layer

        Due to the lack of safety checks, this should not be used directly
        '''
        # Get the batch and input length
        batch_size = x_hidden_state.shape[0]
        x_input_length = x_hidden_state.shape[1]

        # Normalize x_hidden_state to configured dtype
        config_dtype = self.configMap.get_dtype(None)
        if config_dtype != None and config_dtype != "auto":
            x_hidden_state = x_hidden_state.to(config_dtype)

        # Throw an error if prv_stateList is provided, with hybrid layers (prefix or suffix)
        # as KV cache reuse is not implemented, and we will need the full index
        if prv_stateList is not None and self.configMap.num_hybrid_layers() > 0:
            raise NotImplementedError('prv_stateList KV cache reuse, for hybrid models, is not implemented for hybrid layers')
        
        # Prepare the state, with the batch size
        if prv_stateList is None:
            prv_stateList = self.get_init_state(batch_size)

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(x_input_length, device=x_hidden_state.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Initialize the v_first
        v_first = None

        # Uses the input hidden state, as the v_first if v_first_with_embedding is enabled
        if self.configMap.v_first_with_embedding:
            v_first = x_hidden_state.clone()

        # Force overwrite_ret_tensor to False, if ret_stateList is None
        if ret_stateList is None:
            ret_stateList = [ None for i in range(self.configMap.num_qwerky_layers()) ]
            overwrite_ret_tensor = False

        # Get the forward chunk size, and the chunk count
        forward_chunk_size = self.configMap.forward_chunk_size
        forward_chunk_count = math.ceil( x_input_length / forward_chunk_size )

        # Apply rotary embeddings to all layers
        position_embeddings = None
        if self.configMap.use_rotary_pos_emb:
            position_embeddings = self.rotary_emb(x_hidden_state, position_ids)

        # Process prefix hybrid layers if any
        if self.configMap.num_prefix_hybrid_layers > 0:
            for i in range(self.configMap.num_prefix_hybrid_layers):
                x_hidden_state = self._forward_hybrid_layer_hook(
                    self.layers[i],
                    x_hidden_states=x_hidden_state,
                    position_embeddings=position_embeddings, # Used by layer's internal rotary_emb
                    position_ids=position_ids,  # Used by layer's internal rotary_emb
                    past_key_value=None,        # No KV cache support yet
                    output_attentions=False,    # Match Qwerky behavior
                    use_cache=False,            # No KV cache support yet
                )

        # Process Qwerky layers
        qwerky_start = self.configMap.num_prefix_hybrid_layers
        qwerky_end = qwerky_start + self.configMap.num_qwerky_layers()
        for i in range(qwerky_start, qwerky_end):
            layer = self.layers[i]
            qwerky_idx = i - qwerky_start
            
            # Single pass, optimized
            if forward_chunk_count <= 1:
                x_hidden_state, last_layer_state, v_first = self._forward_qwerky_layer_hook(
                    layer, x_hidden_state, prv_stateList[qwerky_idx], v_first, 
                    position_embeddings=position_embeddings
                )
            else:
                # Sadly, we need to chunk
                new_x_hidden_state_arr = [None]*forward_chunk_count
                v_first_arr = [None]*forward_chunk_count if v_first is None else None
                last_layer_state = prv_stateList[qwerky_idx]

                # Iterate the chunks
                for j in range(forward_chunk_count):
                    start = j * forward_chunk_size
                    endin = min( start + forward_chunk_size, x_input_length )

                    new_x_hidden_state, last_layer_state, v_first_part = self._forward_qwerky_layer_hook(
                        layer, 
                        x_hidden_state[:,start:endin], 
                        last_layer_state, 
                        v_first[:, start:endin] if v_first is not None else None,
                        # Position embedding is a tuple pair of tensors, chunk it (tensor[B,T,C], tensor[B,T,C])
                        position_embeddings=(position_embeddings[0][:, start:endin], position_embeddings[1][:, start:endin])
                    )

                    # Save the chunk
                    new_x_hidden_state_arr[j] = new_x_hidden_state
                    if v_first_arr is not None:
                        v_first_arr[j] = v_first_part

                # Merge the chunks
                x_hidden_state = torch.cat(new_x_hidden_state_arr, dim=1)
                if v_first_arr is not None:
                    v_first = torch.cat(v_first_arr, dim=1)

            # Overwrite tensor if needed
            if overwrite_ret_tensor:
                ret_stateList[qwerky_idx][:] = last_layer_state
            else:
                ret_stateList[qwerky_idx] = last_layer_state
                
        # Process suffix hybrid layers if any
        if self.configMap.num_suffix_hybrid_layers > 0:
            for i in range(qwerky_end, len(self.layers)):
                x_hidden_state = self._forward_hybrid_layer_hook(
                    self.layers[i],
                    x_hidden_states=x_hidden_state,
                    position_embeddings=position_embeddings, # Used by layer's internal rotary_emb
                    position_ids=position_ids,  # Used by layer's internal rotary_emb
                    past_key_value=None,        # No KV cache support yet
                    output_attentions=False,    # Match Qwerky behavior
                    use_cache=False,            # No KV cache support yet
                )

        # Final layer norm, without the head
        x_hidden_state = x_hidden_state.to(self.norm.weight.device, non_blocking=True)
        x_hidden_state = self.norm(x_hidden_state)

        # Return the output and the state list
        return x_hidden_state, ret_stateList
        
    def _forward_qwerky_layer_hook(
        self, 
        layer:Qwerky7LayerBlock, 
        x_hidden_state:torch.Tensor, 
        prv_stateList:list[torch.Tensor], 
        v_first:torch.Tensor,
        position_embeddings:torch.Tensor = None
    ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        '''
        Forward layer hook operation, that is easily overridable.
        To implement gradient checkpointing for use in various trainers
        '''
        x_hidden_state = x_hidden_state.to(layer.input_layernorm.weight.device, non_blocking=True)
        if self.checkpoint_function is not None:
            # Use the checkpoint function if set
            return self.checkpoint_function(
                layer, x_hidden_state, prv_stateList, v_first, position_embeddings
            )
        else:
            return layer(x_hidden_state, prv_stateList, v_first, position_embeddings)
        
    def _forward_hybrid_layer_hook(
        self,
        layer:Qwen2DecoderLayer,
        x_hidden_states:torch.Tensor,
        position_embeddings:torch.Tensor,
        position_ids:torch.Tensor,
        past_key_value:Union[None, tuple[torch.Tensor,torch.Tensor]],
        output_attentions:bool=False,
        use_cache:bool=False
    ):
        '''
        Forward layer hook operation, that is easily overridable.
        To implement gradient checkpointing for use in various trainers
        '''
        x_hidden_states = x_hidden_states.to(layer.input_layernorm.weight.device, non_blocking=True)
        if self.checkpoint_function is not None:
            # Use the checkpoint function if set
            return self.checkpoint_function(
                layer, x_hidden_states, position_embeddings, position_ids, past_key_value, output_attentions, use_cache
            )[0]
        else:
            return layer(
                hidden_states=x_hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )[0]
    
    def _forward_internal(
        self, idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor],  
        ret_stateList:list[torch.Tensor],
        position_ids:torch.Tensor = None,
        overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Internal forward operations, which assumes the state is already initialized
        Due to the lack of safety checks, this should not be used directly
        '''
        # Lets get the embedding
        if self.embed_tokens is not None:
            idx = idx.to(self.embed_tokens.weight.device, non_blocking=True)
            x_hidden_state = self.embed_tokens(idx).to( self.configMap.get_dtype("bfloat16") )
        else:
            x_hidden_state = idx # No embedding layer, used for frozen embedding training

        # Forward the layer layers
        x_output_embedding, retStateList = self._forward_internal_embeddings(x_hidden_state, prv_stateList, ret_stateList, position_ids, overwrite_ret_tensor)

        # Return the output and the state list
        return x_output_embedding, retStateList

    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor],
        ret_stateList:list[torch.Tensor],
        position_ids:torch.Tensor = None,
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        # Forward internally
        return self._forward_internal(idx, prv_stateList, ret_stateList, position_ids, overwrite_ret_tensor=True)
  
    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(
        self, in_idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor],
        position_ids:torch.Tensor = None,
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Compiled varient of the forward function, requires previous state to be passed
        '''
        return self._forward_internal(in_idx, prv_stateList, None, position_ids, overwrite_ret_tensor=False)
