import torch, math
from torch import nn
from torch import Tensor
from typing import Union

from .qwerky7_config_map import Qwerky7ConfigMap
from ..block.qwerky7_layer_block import Qwerky7LayerBlock
from .qwerky7_model import Qwerky7Model

class Qwerky7CausalLM(nn.Module):
    def __init__(self, config: Union[Qwerky7ConfigMap, any]):
        super().__init__()
        self.config = Qwerky7ConfigMap.normalize(config)
        self.model = Qwerky7Model(self.config)

        device = self.config.get_device(None)
        dtype = self.config.get_dtype('bfloat16')
        with torch.device(device):
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False).to(dtype=dtype)

    def reset_parameters(self):
        '''
        Reset the parameters of the model, to an initial state used for training a model from scratch
        '''
        self.model.reset_parameters()
        self.lm_head.reset_parameters()
        
    def load_from_model_state_dict(self, state_dict: dict, non_blocking:bool=True):
        '''
        Given the Full/partial qwerky model weights, loaded via `torch.load`
        Setup the RWKV_TimeMix model weights, using the layer_id
        '''
        self.model.load_from_model_state_dict(state_dict, non_blocking=non_blocking)
        self.lm_head.weight.copy_(state_dict['lm_head.weight'])

    def get_init_state(self, batch_size:int=1, skip_init_state:bool=False) -> list[torch.Tensor]:
        '''
        Get an initialized copy of the model state, for the given batch size
        '''
        return self.model.get_init_state(batch_size, skip_init_state=skip_init_state)

    ### ---
    ###
    ### Custom hook overwrites
    ###
    ### ---

    def setup_linear_operation(self, linear_module_function):
        '''
        Configure the linear operation function, to be used by the model
        '''
        self.model.setup_linear_operation(linear_module_function)

    def setup_checkpoint_function(self, checkpoint_function):
        '''
        Configure the checkpoint function, to be used by the model
        '''
        self.model.setup_checkpoint_function(checkpoint_function)

    ### ---
    ###
    ### Forward operation
    ###
    ### ---

    def forward(
            self, 
            input_ids: torch.Tensor,
            prv_stateList:list[torch.Tensor] = None,  
            ret_stateList:list[torch.Tensor] = None,
            overwrite_ret_tensor:bool=False
        ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Forward the layer set, given the input tokens and the last state
        Last state is a list of time mix wkv state

        Returns the output logits and the next state
        '''
        hidden_state, ret_stateList = self.model(input_ids, prv_stateList, ret_stateList, overwrite_ret_tensor=overwrite_ret_tensor)
        logits = self.lm_head(hidden_state)
        return logits, ret_stateList
    
    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, 
        idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor],
        ret_stateList:list[torch.Tensor],
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        # Forward internally
        return self.forward(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=True)
  
    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(
        self, 
        in_idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor]
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Compiled varient of the forward function, requires previous state to be passed
        '''
        return self.forward(in_idx, prv_stateList, None, overwrite_ret_tensor=False)

