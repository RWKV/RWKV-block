import torch, math
from torch import nn
from torch import Tensor
from typing import Union

from .rwkv7_goose_config_map import RWKV7GooseConfigMap
from ..block.rwkv7_layer_block import RWKV7LayerBlock

class RWKV7GooseModel(nn.Module):
    '''
    RWKV7 Goose Model architecture
    Simplified implementation
    '''

    def __init__(self, config: Union[RWKV7GooseConfigMap, any, None] = None):
        # Initialize the base class
        super().__init__()

        # Normalize the config
        configMap:RWKV7GooseConfigMap = RWKV7GooseConfigMap.normalize(config)
        self.configMap = configMap

        # Get the required prop
        num_hidden_layers = configMap.num_hidden_layers
        vocab_size = configMap.vocab_size
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')
        hidden_size = configMap.hidden_size
        head_size = configMap.head_size

        # Checkpoint function hook
        self.checkpoint_function = None

        # Embedding layer
        self.emb = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)

        # main block layers
        blockList = [None]*num_hidden_layers
        for i in range(num_hidden_layers):
            blockList[i] = RWKV7LayerBlock(configMap.new_block_config_map(layer_id=i))
        self.blocks = nn.ModuleList(blockList)

        # ln_out and head
        self.ln_out = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)

        # init state tuning support
        if configMap.init_wkv_state:
            stateTuneList = [None]*num_hidden_layers
            for i in range(num_hidden_layers):
                stateTuneList[i] = nn.ParameterDict({
                    "wkv": nn.Parameter(torch.zeros(hidden_size // head_size, head_size, head_size, device=device, dtype=dtype)),
                })
            self.init_state = nn.ParameterList(stateTuneList)

            if configMap.freeze_wkv_state:
                for i in range(num_hidden_layers):
                    self.init_state[i]["wkv"].requires_grad = False

        # Freeze full weights if needed
        if configMap.freeze_full_weights:
            # Iterate named parameters
            for name, param in self.named_parameters():
                # Check if the parameter is not in the init_state
                if 'init_state' in name:
                    if configMap.freeze_wkv_state:
                        # Freeze the parameter
                        param.requires_grad = False
                    else:
                        # Unfreeze the parameter
                        param.requires_grad = True
                else:
                    # Freeze the parameter
                    param.requires_grad = False


    def reset_parameters(self):
        '''
        Reset the parameters of the block, to an initial state used for training a model from scratch
        '''

        # Get the required prop
        configMap = self.configMap
        num_hidden_layers = configMap.num_hidden_layers
        vocab_size = configMap.vocab_size
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')
        hidden_size = configMap.hidden_size
        
        # Iterate and reset the blocks
        for i in range(num_hidden_layers):
            self.blocks[i].reset_parameters()

        # Reinit the Embedding layer
        self.emb.reset_parameters()

        # Reinit the  ln_out and head
        self.ln_out.reset_parameters()
        if self.head is not None:
            self.head.reset_parameters()

        # Reinit the init state tuning support
        if configMap.init_wkv_state:
            if self.init_state is None:
                stateTuneList = [None]*num_hidden_layers
                for i in range(num_hidden_layers):
                    stateTuneList[i] = nn.ParameterDict({
                        "wkv": nn.Parameter(torch.zeros(hidden_size // 64, 64, 64, device=device, dtype=torch.float)),
                    })
                self.init_state = nn.ParameterList(stateTuneList)
            else:
                for i in range(num_hidden_layers):
                    self.init_state[i]["wkv"].data.copy_(torch.zeros(hidden_size // 64, 64, 64, device=device, dtype=torch.float))

    def load_from_model_state_dict(self, state_dict: dict, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the RWKV_TimeMix model weights, using the layer_id
        '''
        for i, block in enumerate(self.blocks):
            block.load_from_model_state_dict(state_dict, i, non_blocking=non_blocking)
        
        self.ln_out.weight.data.copy_(state_dict['ln_out.weight'], non_blocking=True)
        self.ln_out.bias.data.copy_(state_dict['ln_out.bias'], non_blocking=True)
        self.head.weight.data.copy_(state_dict['head.weight'], non_blocking=True)
        self.emb.weight.data.copy_(state_dict['emb.weight'], non_blocking=True)

        if self.configMap.init_wkv_state:
            for i in range(self.configMap.num_hidden_layers):
                if 'init_state.'+str(i)+'.wkv' in state_dict:
                    self.init_state[i]["wkv"].data.copy_(state_dict['init_state.'+str(i)+'.wkv'], non_blocking=True)

    ### ---
    ###
    ### Custom hook overwrites
    ###
    ### ---

    def setup_checkpoint_function(self, checkpoint_function):
        '''
        Configure the checkpoint function, to be used by the model
        '''
        self.checkpoint_function = checkpoint_function

    ### ---
    ###
    ### Init state handling
    ###
    ### ---

    def get_init_state(self, batch_size:int=1, skip_init_state:bool=False) -> list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]:
        '''
        Get an initialized copy of the model state, for the given batch size
        '''
        # Get required configs
        hidden_size = self.configMap.hidden_size
        init_wkv_state = self.configMap.init_wkv_state
        num_hidden_layers = self.configMap.num_hidden_layers
        head_size = self.configMap.head_size

        # Prepare the initial state
        init_state = [ None for i in range(num_hidden_layers) ]
        for i in range(num_hidden_layers):
            device = self.blocks[i].ln1.weight.data.device
            dtype = self.blocks[i].ln1.weight.data.dtype

            # Use the saved init_state if enabled
            # TODO: Consider letting the wkv_state dtype be a parameter
            if init_wkv_state and skip_init_state == False:
                # Get a repeated clone view of `self.init_state[i]["wkv"]`
                wkv_state = self.init_state[i]["wkv"].expand(batch_size, -1, -1, -1).clone()
                if self.init_state[i]["wkv"].requires_grad:
                    # If the parameter requires grad, we need to detach it
                    if wkv_state.requires_grad != True:
                        raise RuntimeError(f"wkv_state should require grad, but got {wkv_state.requires_grad}")
            else:
                wkv_state = torch.zeros(batch_size, hidden_size // head_size, head_size, head_size, device=device, dtype=torch.float)

            # Prepare the state
            init_state[i] = (
                torch.zeros(batch_size, hidden_size, device=device, dtype=dtype),
                wkv_state,
                torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
            )
        return init_state

    ### ---
    ###
    ### Model Forward
    ###
    ### ---

    def forward(
        self, idx:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]] = None,  
        ret_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]] = None,
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Forward the block set, given the input tokens and the last state
        Last state is a list of tuple of the following
        - time mix shift state
        - time mix wkv state
        - channel mix shift state

        Returns a pair of the output embedding and the next state
        '''
        # Prepare the state, with the batch size
        if prv_stateList is None:
            prv_stateList = self.get_init_state(idx.shape[0])

        # If no return state is set, let _forward_internal, set it up
        if ret_stateList is None:
            ret_stateList = [ None for i in range(self.configMap.num_hidden_layers) ]
            return self._forward_internal(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=False)

        # Forward internally
        return self._forward_internal(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=True)
    
    def _forward_block_hook(self, 
            block:RWKV7LayerBlock, 
            x_hidden_state:torch.Tensor, 
            prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]], 
            v_first:torch.Tensor
        ) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Forward block hook operation, that is easily overridable.
        To implement gradient checkpointing for use in various trainers
        '''
        x_hidden_state = x_hidden_state.to(block.ln1.weight.device, non_blocking=True)
        if self.checkpoint_function is not None:
            # Use checkpointing if available
            return self.checkpoint_function(block, x_hidden_state, prv_stateList, v_first)
        return block(x_hidden_state, prv_stateList, v_first)

    def _forward_internal(
        self, idx:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],  
        ret_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],
        overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Internal forward operations, which assumes the state is already initialized
        Due to the lack of safety checks, this should not be used directly
        '''
        # Lets get the embedding
        idx = idx.to(self.emb.weight.device, non_blocking=True)
        x_hidden_state = self.emb(idx)

        # Forward the block layers
        x_output_embedding, retStateList = self._forward_internal_embeddings(x_hidden_state, prv_stateList, ret_stateList, overwrite_ret_tensor)

        # Perform the head operation
        x_output = self.head(x_output_embedding.to(self.head.weight.device, non_blocking=True))
        
        # Return the output and the state list
        return x_output, retStateList

    def _forward_internal_embeddings(
        self, x_hidden_state:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],  
        ret_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],
        overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Internal forward operations, which assumes the state is already initialized.
        And uses the x_hidden_state as the input (generating by the embedding layer)
        And returns the output embedding before the head layer

        Due to the lack of safety checks, this should not be used directly
        '''
        # Get the batch and input length
        batch_size = x_hidden_state.shape[0]
        x_input_length = x_hidden_state.shape[1]

        # Initialize the v_first
        v_first = None

        # Force overwrite_ret_tensor to False, if ret_stateList is None
        if ret_stateList is None:
            overwrite_ret_tensor = False

        # Get the forward chunk size, and the chunk count
        forward_chunk_size = self.configMap.forward_chunk_size
        forward_chunk_count = math.ceil( x_input_length / forward_chunk_size )

        # Iterate the block layers, compute the x_hidden_state
        for i, block in enumerate(self.blocks):
            
            # Single pass, optimized
            if forward_chunk_count <= 1:
                x_hidden_state, last_block_state, v_first = self._forward_block_hook(block, x_hidden_state, prv_stateList[i], v_first)
            else:
                # Damn it, we need to chunk
                new_x_hidden_state_arr = [None]*forward_chunk_count
                v_first_arr = [None]*forward_chunk_count if v_first is None else None
                last_block_state = prv_stateList[i]

                # Iterate the chunks
                for j in range(forward_chunk_count):
                    start = j * forward_chunk_size
                    endin = min( start + forward_chunk_size, x_input_length )

                    new_x_hidden_state, last_block_state, v_first_part = self._forward_block_hook(
                        block, 
                        x_hidden_state[:,start:endin], 
                        last_block_state, 
                        v_first[:, start:endin] if v_first is not None else None
                    )

                    # Save the chunk
                    new_x_hidden_state_arr[j] = new_x_hidden_state
                    if v_first_arr is not None:
                        v_first_arr[j] = v_first_part

                # Merge the chunks
                x_hidden_state = torch.cat(new_x_hidden_state_arr, dim=1)
                if v_first_arr is not None:
                    v_first = torch.cat(v_first_arr, dim=1)

            # last_block_state = prv_stateList[i]
            # Overwrite tensor if needed
            if overwrite_ret_tensor:
                ret_stateList[i][0][:] = last_block_state[0]
                ret_stateList[i][1][:] = last_block_state[1]
                ret_stateList[i][2][:] = last_block_state[2]
            else:
                ret_stateList[i] = last_block_state
                
        # Final layer norm, and head
        x_hidden_state = x_hidden_state.to(self.ln_out.weight.device, non_blocking=True)
        x_hidden_state = self.ln_out(x_hidden_state)

        # Return the output and the state list
        return x_hidden_state, ret_stateList
    
    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, idx:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],
        ret_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]],
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        # Forward internally
        return self._forward_internal(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=True)
  
    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(
        self, in_idx:torch.Tensor, 
        prv_stateList:list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]
    ) -> tuple[torch.Tensor,list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        '''
        Compiled varient of the forward function, requires previous state to be passed
        '''
        return self._forward_internal(in_idx, prv_stateList, None, overwrite_ret_tensor=False)
