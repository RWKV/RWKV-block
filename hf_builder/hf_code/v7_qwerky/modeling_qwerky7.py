""" Qwerky Modeling"""

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

import torch, math
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union, Any

# Load the config and model
from .configuration_qwerky7 import Qwerky7Config
from .modeling_blocks_qwerky7 import Qwerky7Model as RwkvBlockQwerky7Model

class Qwerky7State(Cache):
    '''
    HF Cache API implementation for qwerky7
    To speed up inference time by "reusing" KV cache's as a means of "state"
    '''
    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.qwerky_state: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int):
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return self.qwerky_state[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield self.qwerky_state[layer_idx]

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.qwerky_state)

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Linear Attention variants do not have a maximum length
        return new_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError('Cannot reorder Linear Attention state')

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length.
        """
        return None

    def crop(self, max_length: int):
        # can't implement this for linear attention variants, skips
        raise NotImplementedError('Cannot crop Linear Attention state')

    @torch.no_grad
    def update(
        self,
        layer_qwerky_state: torch.Tensor,
        token_count: int,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:        
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += token_count

        # Update the cache
        # There may be skipped layers, fill them with empty lists
        for _ in range(len(self.qwerky_state), layer_idx + 1):
            self.qwerky_state.append([
                layer_qwerky_state
            ])

        # Check the batch sizing matches, reset the cache on mismatch
        batch_size = layer_qwerky_state.shape[0]
        if batch_size != self.qwerky_state[layer_idx].shape[0]:
            self.qwerky_state[layer_idx] = torch.zeros_like(layer_qwerky_state)

        # Update the cache
        self.qwerky_state[layer_idx].copy_(layer_qwerky_state)

        # Return the updated cache
        return self.qwerky_state[layer_idx]

class Qwerky7PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """
    config_class = Qwerky7Config
    
    base_model_prefix = "model"
    is_parallelizable = True
    _no_split_modules = ["Qwerky7LayerBlock"]
    _keep_in_fp32_modules = []

    # Enable cacching of k/v's
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = False
    _skip_keys_device_placement = "past_key_values"

    # Enable gradient checkpointing by default
    supports_gradient_checkpointing = True
    gradient_checkpointing = True

    def __init__(self, config: Qwerky7Config=None):
        # Work around for multiple inheritance
        if config is None and self.config is not None:
            config = self.config
        else:
            self.config = config
        if config is None:
            raise ValueError("Missing `config` or `config` attribute in the class")
            
        super().__init__(config)
        
    def _init_weights(
        self,
        module
    ):
        # Fallback to the default init weights
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            return

        # Default FP initializer_range for Linear / LN layers
        initializer_range = 0.02

        if isinstance(module, (nn.ParameterList, nn.ModuleList)):
            # Iterate and initialize each parameter
            for param in module:
                self._init_weights(param)
        elif isinstance(module, nn.ParameterDict):
            # Iterate and initialize each parameter
            for key, param in module.items():
                self._init_weights(param)

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=initializer_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)

            # # Qwerky does not use a blank pad idx. The pad_idx is a training token
            # if module.padding_idx is not None:
            #     module.weight.data[module.padding_idx].zero_()

@dataclass
class Qwerky7Output(ModelOutput):
    """
    Class for the Qwerky model outputs.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    qwerky_state: Optional[list[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Qwerky7State] = None


@dataclass
class Qwerky7CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    qwerky_state: Optional[list[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Qwerky7State] = None

QWERKY7_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config ([`Rwkv7Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

Qwerky7_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary. If `past_key_values` is used, only `input_ids` that do not have their
            past calculated should be passed as `input_ids`. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        
        state (List block states, representing the Qwerky various internal states per layer `(batch_size, hidden_state)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).

        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare Qwerky7 Model transformer outputting raw hidden-states without activating the head (variable is still declared)",
    QWERKY7_START_DOCSTRING,
)
class Qwerky7BaseModel(RwkvBlockQwerky7Model, Qwerky7PreTrainedModel):
    def __init__(self, config: Qwerky7Config):
        # Work around for multiple inheritance
        self.config = config
        super().__init__(config)

        # Add rotary embeddings for all layers
        self.rotary_emb = Qwen2RotaryEmbedding(config=config.hybrid_layer_config())

        # Setup pipeline parallel devices if specified
        self.pipeline_parallel_devices = getattr(config, 'pipeline_parallel_devices', None)
        if self.pipeline_parallel_devices:
            # Validate devices
            for device in self.pipeline_parallel_devices:
                if not isinstance(device, str):
                    raise ValueError(f"Pipeline parallel device must be string, got {type(device)}")
            
            num_devices = len(self.pipeline_parallel_devices)
            # Count total layers including embeddings and final norm
            total_layers = len(self.layers) + 2  # +2 for embeddings and final norm
            layers_per_device = math.ceil(total_layers / num_devices)
            
            # Move embeddings to first device
            self.embed_tokens = self.embed_tokens.to(self.pipeline_parallel_devices[0])
            
            # Move main layers
            for i in range(len(self.layers)):
                device_idx = (i + 1) // layers_per_device  # +1 because embeddings is first layer
                device_idx = min(device_idx, num_devices - 1)  # Ensure we don't exceed device count
                self.layers[i] = self.layers[i].to(self.pipeline_parallel_devices[device_idx])
            
            # Move final norm to last device
            self.norm = self.norm.to(self.pipeline_parallel_devices[-1])
    
    def get_input_embeddings(self):
        return self.emb
    def set_input_embeddings(self, value):
        self.emb = value

    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(Qwerky7_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        output_type=Qwerky7Output,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        qwerky_state: Optional[list[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, Qwerky7Output]:
        '''
        Forward with input_ids, and the qwerky_state
        To get the output hidden state (not logits), and the next qwerky_state
        '''
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            warnings.warning_once("`Qwerky7Model` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        
        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False
 
        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if output_hidden_states:
            warnings.warning_once("`Qwerky7Model` does not `output_hidden_states` now, setting it to `False`.")
            output_hidden_states = False

        # ---

        # Compute the input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
        x_hidden_state = inputs_embeds

        # Input length to perform chunking by
        batch_size = x_hidden_state.shape[0]
        x_input_length = x_hidden_state.shape[1]

        # ---

        # Initialize the cache if needed
        if use_cache and not isinstance(past_key_values, Qwerky7State):
            past_key_values = Qwerky7State()
            past_key_values.qwerky_state = self.get_init_state(batch_size=batch_size)

        # Get the qwerky_state / prv_stateList
        # Uses the given qwerky_state if given
        # (high priority over the past_key_values cache)
        if qwerky_state is None:
            # From the cache if provided, and if caching is enabled
            if past_key_values is not None and use_cache:
                qwerky_state = past_key_values.qwerky_state
            
            # No cache, no qwerky_state, initialize a new one
            qwerky_state = self.get_init_state(batch_size=batch_size)
        prv_stateList = qwerky_state

        # Initialize the ret_stateList
        ret_stateList = self.get_init_state(batch_size=batch_size, skip_init_state=True)

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(x_input_length, device=x_hidden_state.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Initialize the v_first
        v_first = None

        # Uses the input hidden state, as the v_first if v_first_with_embedding is enabled
        if self.config.v_first_with_embedding:
            v_first = x_hidden_state.clone()

        # Get the forward chunk size, and the chunk count
        forward_chunk_size = self.config.forward_chunk_size
        forward_chunk_count = math.ceil(x_input_length / forward_chunk_size)

        # Initialize internal states
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        # Block forward, with gradient if needed
        def qwerky_layer_forward(layer, in_x_state, in_qwerky_state, in_v_first, in_position_embeddings):
            if self.gradient_checkpointing and self.training:
                return self._gradient_checkpointing_func(
                    layer.__call__, in_x_state, in_qwerky_state, in_v_first, position_embeddings=in_position_embeddings
                )
            else:
                return layer(in_x_state, in_qwerky_state, in_v_first, position_embeddings=in_position_embeddings)
        
        # Generate rotary embeddings for all layers
        position_embeddings = self.rotary_emb(x_hidden_state, position_ids)

        # Process prefix hybrid layers if any
        if self.config.num_prefix_hybrid_layers > 0:
            for i in range(self.config.num_prefix_hybrid_layers):
                layer = self.layers[i]
                x_hidden_state = x_hidden_state.to(layer.input_layernorm.weight.device, non_blocking=True)
                x_hidden_state = layer(
                    hidden_states=x_hidden_state,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]

                if output_hidden_states:
                    all_hidden_states += (x_hidden_state,)

        # Process Qwerky layers
        qwerky_start = self.config.num_prefix_hybrid_layers
        qwerky_end = qwerky_start + (self.config.num_hidden_layers - self.config.num_suffix_hybrid_layers - self.config.num_prefix_hybrid_layers)
        for i in range(qwerky_start, qwerky_end):
            layer = self.layers[i]
            
            # Move tensors to layer's device
            if self.pipeline_parallel_devices:
                layer_device = layer.input_layernorm.weight.device
                x_hidden_state = x_hidden_state.to(layer_device, non_blocking=True)
                if v_first is not None:
                    v_first = v_first.to(layer_device, non_blocking=True)
                if isinstance(position_embeddings, tuple):
                    position_embeddings = (position_embeddings[0].to(layer_device, non_blocking=True),
                                        position_embeddings[1].to(layer_device, non_blocking=True))
            
            qwerky_idx = i - qwerky_start
            if self.pipeline_parallel_devices and prv_stateList[qwerky_idx] is not None:
                prv_stateList[qwerky_idx] = prv_stateList[qwerky_idx].to(layer_device, non_blocking=True)

            # Single pass, optimized  
            if forward_chunk_count <= 1:
                x_hidden_state, ret_subList, v_first = qwerky_layer_forward(layer, x_hidden_state, prv_stateList[qwerky_idx], v_first, position_embeddings)
                ret_stateList[qwerky_idx] = ret_subList
            else:
                # Chunk processing
                new_x_hidden_state_arr = [None]*forward_chunk_count
                v_first_arr = [None]*forward_chunk_count if v_first is None else None
                qwerky_idx = i - qwerky_start
                ret_subList = prv_stateList[qwerky_idx]
                
                # Forward in chunks
                for chunk_idx in range(forward_chunk_count):
                    start = chunk_idx * forward_chunk_size
                    endin = min(start + forward_chunk_size, x_input_length)

                    new_x_hidden_state, ret_subList, v_first_part = qwerky_layer_forward(
                        layer, 
                        x_hidden_state[:, start:endin], 
                        ret_subList, 
                        v_first[:, start:endin] if v_first is not None else None,
                        # Position embedding is a tuple pair of tensors, chunk it (tensor[B,T,C], tensor[B,T,C])
                        position_embeddings=(position_embeddings[0][:, start:endin], position_embeddings[1][:, start:endin])
                    )

                    new_x_hidden_state_arr[chunk_idx] = new_x_hidden_state
                    if v_first_arr is not None:
                        v_first_arr[chunk_idx] = v_first_part

                # Merge the chunks
                x_hidden_state = torch.cat(new_x_hidden_state_arr, dim=1)
                if v_first_arr is not None:
                    v_first = torch.cat(v_first_arr, dim=1)
                ret_stateList[qwerky_idx] = ret_subList

            # Save the state to cache if needed
            if past_key_values is not None and use_cache:
                past_key_values.update(ret_stateList[qwerky_idx], x_input_length, qwerky_idx)

            if output_hidden_states:
                all_hidden_states += (x_hidden_state,)

        # Process suffix hybrid layers if any
        if self.config.num_suffix_hybrid_layers > 0:
            for i in range(qwerky_end, len(self.layers)):
                layer = self.layers[i]
                x_hidden_state = x_hidden_state.to(layer.input_layernorm.weight.device, non_blocking=True)
                x_hidden_state = layer(
                    hidden_states=x_hidden_state,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_value=None,  # No KV cache support yet
                    output_attentions=False,
                    use_cache=False,  # No KV cache support yet
                )[0]

                if output_hidden_states:
                    all_hidden_states += (x_hidden_state,)

        # Final layer norm
        x_hidden_state = x_hidden_state.to(self.norm.weight.device, non_blocking=True)
        x_hidden_state = self.norm(x_hidden_state)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (x_hidden_state,)

        if not return_dict:
            return tuple(i for i in [x_hidden_state, qwerky_state, all_hidden_states, all_attns] if i is not None)
        return Qwerky7Output(
            last_hidden_state=x_hidden_state,
            qwerky_state=qwerky_state,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            past_key_values=past_key_values
        )

# @add_start_docstrings(
#     """
#     The Qwerky Model transformer with a language modeling head on top (linear layer with weights tied to the input
#     embeddings).
#     """,
#     QWERKY7_START_DOCSTRING,
# )
class Qwerky7ForCausalLM(Qwerky7PreTrainedModel, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwerky7BaseModel(config)

        dtype=self.model.embed_tokens.weight.dtype
        device=self.model.norm.weight.device
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype, device=device)

        self.post_init()

    def prepare_inputs_for_generation(
        self, 
        input_ids=None, 
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        qwerky_state: Optional[list[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # num_new_tokens_if_qwerky_state: int = 1, # Only triggers if given input_ids + qwerky_state
        num_logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        '''
        Personal Notes: On huggingface barely documented "Transformer" hooks.

        I assume this is triggered once, for the start of AI inference.
        With subsequent calls for forward on each token step, being updated with
        `_update_model_kwargs_for_generation` function instead?
        '''
        # # only last token for `inputs_ids` if the `past_key_values` is passed along.
        # if qwerky_state is not None and input_ids is not None:
        #     input_ids = input_ids[:, -num_new_tokens_if_qwerky_state:]
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None:
            if input_ids is not None:
                raise ValueError("You cannot specify both `inputs_ids` and `inputs_embeds` at the same time")
            model_inputs = {'inputs_embeds': inputs_embeds}

            # Get input sizing
            batch_size = inputs_embeds.shape[0]
            # x_input_length = inputs_embeds.shape[1]

        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

            # Get input sizing
            batch_size = input_ids.shape[0]
            # x_input_length = input_ids.shape[1]

        # ---

        # Initialize the cache if needed
        if use_cache and not isinstance(past_key_values, Qwerky7State):
            past_key_values = Qwerky7State()
            past_key_values.qwerky_state = self.model.get_init_state(batch_size=batch_size)

        if num_logits_to_keep is not None:
            model_inputs['num_logits_to_keep'] = num_logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'qwerky_state': qwerky_state,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'num_logits_to_keep': num_logits_to_keep,
        })
        return model_inputs

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        num_new_tokens: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        # Get the past_key_values from the outputs
        past_key_values = outputs.get("past_key_values", None)
        if past_key_values is not None:
            qwerky_state = past_key_values.qwerky_state
        else:
            qwerky_state = outputs.get("qwerky_state", model_kwargs.get("qwerky_state", None))
        
        # Prefer the output qwerky_state, if provided
        input_ids = model_kwargs.get("input_ids", None)
        attention_mask = model_kwargs.get("attention_mask", None)

        # only last token for inputs_ids if the state is passed along.
        if qwerky_state is not None and input_ids is not None and num_new_tokens > 0:
            input_ids = input_ids[:, -num_new_tokens:]
            model_kwargs["input_ids"] = input_ids

            if attention_mask is not None:
                attention_mask = attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))
                model_kwargs["attention_mask"] = attention_mask

        # Return the formated output
        return model_kwargs
    
    # @add_start_docstrings_to_model_forward(Qwerky7_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     output_type=Qwerky7CausalLMOutput,
    # )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        qwerky_state: Optional[list[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, Qwerky7CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rwkv_outputs = self.model.forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            inputs_embeds=inputs_embeds, 
            past_key_values=past_key_values, 
            qwerky_state=qwerky_state, 
            use_cache=use_cache, 
            cache_position=cache_position,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=False
        )

        # Get the hidden state, and the updated Qwerky state
        hidden_states = rwkv_outputs[0]
        qwerky_state = rwkv_outputs[1]

        # Get the ALL hidden states and attentions dumps
        all_hidden_states = rwkv_outputs[2] if output_hidden_states else None
        if output_hidden_states:
            all_attns = rwkv_outputs[3] if output_attentions else None
        else:
            all_attns = rwkv_outputs[2] if output_attentions else None
        
        # Forward the head state
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))

        # Compute the loss from the labels
        loss = None
        if labels is not None:

            # Setup loss function
            if self._loss_function_cache is None:
                self._loss_function_cache = CrossEntropyLoss()

            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Compute the token loss

            if attention_mask is not None:
                token_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
                submask = attention_mask[..., 1:].contiguous().view(-1)
                loss = (token_loss * submask).sum() / submask.sum()
            else:
                loss = F.cross_entropy(shift_logits.view(-1, shift_labels.size(-1)), shift_labels.view(-1), reduction="mean")

        if not return_dict:
            return tuple(i for i in [loss, logits, qwerky_state, all_hidden_states, all_attns] if i is not None)
        
        return Qwerky7CausalLMOutput(
            loss=loss,
            logits=logits,
            qwerky_state=qwerky_state,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            past_key_values=past_key_values
        )

__all__ = ["Qwerky7ForCausalLM", "Qwerky7BaseModel", "Qwerky7PreTrainedModel"]
