""" RWKV configuration"""

from transformers.configuration_utils import PretrainedConfig
# from transformers.utils import logging
# logger = logging.get_logger(__name__)

# Import the dependencies
from .modeling_blocks_rwkv6 import RWKV6FinchConfigMap

class RWKV6Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Rwkv6Model`]. It is used to instantiate a RWKV6
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RWVK-6
    [RWKV/v6-Finch-1.5B-HF](https://huggingface.co/RWKV/v6-Finch-1.5B-HF) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the RWKV6 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Rwkv6Model`].
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        
        hidden_size_att (`int`, *optional*):
            Dimensionality of the attention hidden states. Will be computed from `hidden_size` if unset.
        hidden_size_ffn (`int`, *optional*):
            Dimensionality of the FFN hidden states. Will be computed from `hidden_size` if unset.
        head_size (`int`, *optional*, defaults to 64): 
            head_size of rwkv6 self_attention module.
        tmix_backend (`str`, *optional*, defaults to "auto"):
            Backend to use for the time mix module. "auto" defaults to "pytorch" if the device is "cpu" and "cuda" otherwise.
            (Valid values: "auto", "pytorch", "cuda", "triton", "triton_bighead", "fla", "fla_fused", "pytorch_ref", "pytorch_ref_fp32")
        init_wkv_state (`bool`, *optional*, defaults to `False`):
            Whether to initialize the wkv state in the model. Used for WKV state tuning.
        forward_chunk_size (`int`, *optional*, defaults to 4096):
            Chunk size for the forward pass. Used to break large inputs into smaller chunks to avoid OOM errors.
            
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
    >>> from transformers import Rwkv6Config, Rwkv6Model

    >>> # Initializing a Rwkv6 configuration
    >>> configuration = Rwkv6Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Rwkv6Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rwkv6"

    def __init__(
        self,
        ########################################
        # RWKV-block configuration
        ########################################
        # Vocab, layer count, and hidden size
        vocab_size=65536,
        num_hidden_layers=24,
        hidden_size=768,
        # Optional hidden sizes
        hidden_size_att=None,
        hidden_size_ffn=None,
        # Headsize, timemix backend
        head_size=64,
        tmix_backend="auto",
        init_wkv_state=False,
        # Trainer model configs
        dropout_rate=0.0,
        # Internal forward chunk size
        forward_chunk_size=4096,
        # # Torch device and dtype
        # device=None,
        # dtype=None,
        ########################################
        # HF specific configuration
        ########################################
        use_cache=True,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
        use_bfloat16=True,
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

        self.head_size = head_size
        self.tmix_backend = tmix_backend
        self.init_wkv_state = init_wkv_state
        self.forward_chunk_size = forward_chunk_size

        # self.device = device
        # self.dtype = dtype

        self.dropout_rate = dropout_rate
        self.use_cache = use_cache
        
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
        finch_config = RWKV6FinchConfigMap.from_model_state_dict(state_dict)
        # Join dictionary with **finch_config.__dict__ and **kwargs
        return RWKV6Config(**{**finch_config.__dict__, **kwargs})
