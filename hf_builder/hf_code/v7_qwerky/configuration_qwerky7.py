""" RWKV configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
# from transformers.utils import logging
# logger = logging.get_logger(__name__)

# Import the dependencies
from .modeling_blocks_qwerky7 import Qwerky7ConfigMap as RwkvBlockQwerky7ConfigMap, Qwerky7BlockConfigMap

class Qwerky7Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Qwerky7Model`]. It is used to instantiate a RWKV7
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RWVK-7

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the RWKV7 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwerky7Model`].
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        pipeline_parallel_devices (`List[str]`, *optional*):
            List of devices for pipeline parallel execution. Each device string should be in the format "cuda:N".
            When provided, the model layers will be distributed across these devices.
        
        hidden_size_att (`int`, *optional*):
            Dimensionality of the attention hidden states. Will be computed from `hidden_size` if unset.
        hidden_size_ffn (`int`, *optional*):
            Dimensionality of the FFN hidden states. Will be computed from `hidden_size` if unset.
        head_size (`int`, *optional*, defaults to 64): 
            head_size of rwkv7 self_attention module.
        tmix_backend (`str`, *optional*, defaults to "auto"):
            Backend to use for the time mix module. "auto" defaults to "pytorch" if the device is "cpu" and "cuda" otherwise.
            (Valid values: "auto", "pytorch", "cuda", "triton", "triton_bighead", "fla", "fla_fused", "pytorch_ref", "pytorch_ref_fp32")
        init_state_wkv (`bool`, *optional*, defaults to `False`):
            Whether to initialize the wkv state in the model. Used for WKV state tuning.
        forward_chunk_size (`int`, *optional*, defaults to 4096):
            Chunk size for the forward pass. Used to break large inputs into smaller chunks to avoid OOM errors.
            
        num_prefix_hybrid_layers (`int`, *optional*, defaults to 0):
            Number of Qwen2 transformer layers to use at the start of the model.
        num_suffix_hybrid_layers (`int`, *optional*, defaults to 0):
            Number of Qwen2 transformer layers to use at the end of the model.
        hybrid_num_attention_heads (`int`, *optional*, defaults to 0):
            Number of attention heads for Qwen2 layers.
        hybrid_num_key_value_heads (`int`, *optional*, defaults to 0):
            Number of key/value heads for Qwen2 layers.
        hybrid_attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights in Qwen2 layers.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Base period for rotary position embeddings in Qwen2 layers.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length supported by position embeddings.

        device (`str`, *optional*):
            Device to use for the model. Use the respective torch.device types
        dtype (`str`, *optional*):
            Model weights data type. Use the respective torch.dtype types

        use_cache (bool):
            Reuse of the past rwkv state to reduce token computations. Defaults to `True`.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary. Defaults to 0.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary. Defaults to 0.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the input token embeddings.
            (this value is currently ignored in our implementation)

    Example:

    ```python
    >>> from transformers import Qwerky7Config, Qwerky7Model

    >>> # Initializing a Rwkv7 configuration
    >>> configuration = Qwerky7Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Qwerky7Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwerky7"

    def __init__(
        self,
        ########################################
        # RWKV-block configuration
        ########################################
        # Vocab, layer count, and hidden size
        vocab_size=152064,
        num_hidden_layers=24,
        hidden_size=768,
        # Optional hidden sizes
        hidden_size_att=None,
        hidden_size_ffn=None,
        # Headsize, timemix backend
        head_size=64,
        tmix_backend="auto",
        init_state_wkv=False,
        # Trainer model configs
        dropout_rate=0.0,
        # Internal forward chunk size
        forward_chunk_size=4096,
        # V First embedding support
        v_first_with_embedding=False,
        ########################################
        # Hybrid model configuration
        ########################################
        num_prefix_hybrid_layers=0,
        num_suffix_hybrid_layers=0,
        hybrid_num_attention_heads=0,
        hybrid_num_key_value_heads=0,
        hybrid_attention_dropout=0.0,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        ########################################
        # HF specific configuration
        ########################################
        use_cache=False,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
        use_bfloat16=True,
        pipeline_parallel_devices=None,
        ########################################
        **kwargs,
    ):
        # Normalize dtype if torch_dtype is set within kwargs
        if "torch_dtype" in kwargs:
            kwargs["dtype"] = kwargs["torch_dtype"]

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.hidden_size_att = hidden_size_att
        self.hidden_size_ffn = hidden_size_ffn
        self.v_first_with_embedding = v_first_with_embedding

        self.head_size = head_size
        self.tmix_backend = tmix_backend
        self.init_state_wkv = init_state_wkv
        self.v_first_with_embedding = v_first_with_embedding
        self.forward_chunk_size = forward_chunk_size

        self.dropout_rate = dropout_rate
        self.use_cache = use_cache
        self.pipeline_parallel_devices = pipeline_parallel_devices

        # Hybrid model configuration
        self.num_prefix_hybrid_layers = num_prefix_hybrid_layers
        self.num_suffix_hybrid_layers = num_suffix_hybrid_layers
        self.hybrid_num_attention_heads = hybrid_num_attention_heads
        self.hybrid_num_key_value_heads = hybrid_num_key_value_heads
        self.hybrid_attention_dropout = hybrid_attention_dropout
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        # Forward to the HF PretrainedConfig
        super().__init__(
            tie_word_embeddings=tie_word_embeddings, 
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id, 
            use_bfloat16=use_bfloat16,
            **kwargs
        )

    @staticmethod
    def from_model_state_dict(state_dict: dict, **kwargs):
        base_config = RwkvBlockQwerky7ConfigMap.from_model_state_dict(state_dict, **kwargs)
        # Join dictionary with base config and kwargs
        return Qwerky7Config(**{**base_config.__dict__, **kwargs})
    
    def new_block_config_map(self, **kwargs) -> 'Qwerky7BlockConfigMap':
        '''
        Returns a new config map with updated values
        '''
        new_dict = {}
        for key in Qwerky7BlockConfigMap.__dataclass_fields__:
            if key in self.__dict__:
                new_dict[key] = self.__dict__[key]
        new_dict.update(kwargs)

        return Qwerky7BlockConfigMap(**new_dict)

    def hybrid_layer_config(self) -> Qwen2Config:
        '''
        Returns the Qwen2 configuration for hybrid layers
        '''
        return Qwen2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.hidden_size_ffn or 4 * self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.hybrid_num_attention_heads,
            num_key_value_heads=self.hybrid_num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            attention_dropout=self.hybrid_attention_dropout,
            rope_theta=self.rope_theta,
            use_cache=self.use_cache,
        )

    def num_qwerky_layers(self) -> int:
        """
        Returns the number of qwerky layers in the model
        """
        return self.num_hidden_layers - self.num_suffix_hybrid_layers - self.num_prefix_hybrid_layers

    def num_hybrid_layers(self) -> int:
        """
        Returns the total number of hybrid layers in the model
        """
        return self.num_suffix_hybrid_layers + self.num_prefix_hybrid_layers
