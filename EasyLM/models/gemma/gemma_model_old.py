import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import tempfile
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning
import einops

import sentencepiece as spm
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from mlxu import function_args_to_config, load_pickle, open_file

from EasyLM.bpt import blockwise_ffn, blockwise_attn
from EasyLM.jax_utils import (
    with_sharding_constraint,
    get_jax_mesh,
    get_gradient_checkpoint_policy,
)


GEMMA_STANDARD_CONFIGS = {
    "2b": {
        "vocab_size": 256000,
        "hidden_size": 2048,
        "head_dim": 256,  # for Gemma
        "intermediate_size": 16384,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "max_sequence_length": 8192,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
    },
    "7b": {
        "vocab_size": 256000,
        "hidden_size": 3072,
        "head_dim": 256,  # for Gemma
        "intermediate_size": 24576,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_sequence_length": 8192,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
    },
    "debug": {  # A small model for debugging
        "vocab_size": 256000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_sequence_length": 2048,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
    },
}


class GemmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GemmaModel`]. It is used to instantiate an Gemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Gemma model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~GemmaModel`] or [`~TFGemmaModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_sequence_length (`int`, *optional*, defaults to 2048):
            Max sequence length for model (for RoPE computation)
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from transformers import GemmaModel, GemmaConfig
    >>> # Initializing a Gemma gemma-7b style configuration
    >>> configuration = GemmaConfig()
    >>> # Initializing a model from the gemma-7b style configuration
    >>> model = GemmaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        head_dim=256,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # for GQA?
        max_sequence_length=2048,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        # pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        remat_block="",
        remat_attention="",
        remat_mlp="",
        scan_attention=False,
        scan_mlp=False,
        scan_query_chunk_size=1024,
        scan_key_chunk_size=1024,
        scan_mlp_chunk_size=1024,
        fcm_min_ratio=0.0,
        fcm_max_ratio=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.remat_block = remat_block
        self.remat_attention = remat_attention
        self.remat_mlp = remat_mlp
        self.scan_attention = scan_attention
        self.scan_mlp = scan_mlp
        self.scan_query_chunk_size = scan_query_chunk_size
        self.scan_key_chunk_size = scan_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        super().__init__(
            # pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @staticmethod
    def get_jax_mesh(axis_dims):
        return get_jax_mesh(axis_dims, ("dp", "fsdp", "mp"))

    @staticmethod
    def get_partition_rules():
        """Parition rules for GPTJ. Note that these rules are orderd, so that
        the beginning rules match first. It is important to use
        PartitionSpec() instead of None here because JAX does not treat
        None as a pytree leaf.
        """
        return (
            # embeddings
            # ("transformer/embed_tokens/embedding", PS("mp", "fsdp")),
            ("embed_tokens/embedding", PS("mp", "fsdp")),
            # atention
            ("attention/(q_proj|k_proj|v_proj)/kernel", PS("fsdp", "mp")),
            ("attention/o_proj/kernel", PS("mp", "fsdp")),
            # mlp
            ("mlp/w1/kernel", PS("fsdp", "mp")),
            ("mlp/w2/kernel", PS("mp", "fsdp")),
            ("mlp/w3/kernel", PS("fsdp", "mp")),
            # layer norms
            ("input_layernorm/kernel", PS(None)),
            ("post_attention_layernorm/kernel", PS(None)),
            # output head
            ("transformer/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "mp")),
            (".*", PS(None)),
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ("params", "dropout", "fcm")

    @staticmethod
    def get_tokenizer_config(updates=None):
        config = ConfigDict()
        config.vocab_file = ""
        config.add_bos_token = True
        config.add_eos_token = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_tokenizer(cls, config, padding_side="left", truncation_side="right"):
        config = cls.get_tokenizer_config(config)
        assert config.vocab_file != "", "vocab_file must be specified"
        tokenizer = GemmaTokenizer(
            vocab_file=config.vocab_file,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )
        return tokenizer

    @classmethod
    def load_config(cls, path):
        if path in GEMMA_STANDARD_CONFIGS:
            return cls.from_dict(GEMMA_STANDARD_CONFIGS[path])
        load_type, load_path = path.split("::", 1)
        if load_type == "pickle":
            return cls.from_dict(load_pickle(load_path)["gemma_config"])
        elif load_type == "json":
            with open_file(load_path, "r") as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f"Unsupported load config type: {load_type}")


remat = nn_partitioning.remat

logger = logging.get_logger(__name__)


# class RMSNorm(nn.Module):
#     dim: int
#     eps: float = 1e-6
#     dtype: jnp.dtype = jnp.float32
#     param_dtype: jnp.dtype = jnp.float32

#     def setup(self) -> None:
#         self.weight = self.param(
#             # "kernel",
#             "weight",  # Gemma
#             nn.initializers.ones,
#             (self.dim,),
#             self.param_dtype,
#         )

#     def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
#         return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
#         output = self._norm(x).astype(self.dtype)
#         weight = jnp.asarray(self.weight, self.dtype)
#         # return output * weight
#         return output * (1 + weight)  # Update at Gemma


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(
        *xq_out.shape[:-1], -1
    )

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(
        *xk_out.shape[:-1], -1
    )

    return xq_out.astype(dtype), xk_out.astype(dtype)


# class FlaxGemmaAttention(nn.Module):
#     config: GemmaConfig
#     dtype: jnp.dtype = jnp.float32
#     param_dtype: jnp.dtype = jnp.float32
#     precision: Optional[Union[jax.lax.Precision, str]] = None

#     def setup(self):
#         config = self.config
#         self.embed_dim = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         # self.head_dim = self.embed_dim // self.num_heads
#         self.head_dim = config.head_dim  # for Gemma

#         # for Gemma
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads

#         self.q_proj = nn.Dense(
#             # config.num_attention_heads * self.head_dim,
#             self.num_heads * self.head_dim,  # for Gemma
#             dtype=self.dtype,
#             param_dtype=self.param_dtype,
#             use_bias=False,
#             kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
#             precision=self.precision,
#         )
#         self.k_proj = nn.Dense(
#             # config.num_attention_heads * self.head_dim,
#             self.num_key_value_heads * self.head_dim,  # for Gemma
#             dtype=self.dtype,
#             param_dtype=self.param_dtype,
#             use_bias=False,
#             kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
#             precision=self.precision,
#         )
#         self.v_proj = nn.Dense(
#             # config.num_attention_heads * self.head_dim,
#             self.num_key_value_heads * self.head_dim,  # for Gemma
#             dtype=self.dtype,
#             param_dtype=self.param_dtype,
#             use_bias=False,
#             kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
#             precision=self.precision,
#         )
#         self.o_proj = nn.Dense(
#             # config.hidden_size,
#             self.embed_dim,
#             dtype=self.dtype,
#             param_dtype=self.param_dtype,
#             use_bias=False,
#             kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
#             precision=self.precision,
#         )

#         self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

#         self.causal_mask = make_causal_mask(
#             jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool"
#         )

#         self.freqs_cis = precompute_freqs_cis(
#             self.head_dim,
#             config.max_sequence_length * 2,
#             dtype=self.dtype,
#         )

#     def _split_heads(self, hidden_states, num_heads):
#         # return hidden_states.reshape(
#         #     hidden_states.shape[:2] + (self.num_heads, self.head_dim)
#         # )
#         # for Gemma
#         return hidden_states.reshape(
#             hidden_states.shape[:2] + (num_heads, self.head_dim)
#         )

#     def _merge_heads(self, hidden_states):
#         # return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))
#         # for Gemma
#         return hidden_states.reshape(
#             hidden_states.shape[:2] + (self.num_heads * self.head_dim,)
#         )

#     @nn.compact
#     def _concatenate_to_cache(self, key, value, query, attention_mask):
#         """
#         This function takes projected key, value states from a single input token and concatenates the states to cached
#         states from previous steps. This function is slighly adapted from the official Flax repository:
#         https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
#         """
#         # detect if we're initializing by absence of existing cache data.
#         is_initialized = self.has_variable("cache", "cached_key")
#         cached_key = self.variable(
#             "cache", "cached_key", jnp.zeros, key.shape, key.dtype
#         )
#         cached_value = self.variable(
#             "cache", "cached_value", jnp.zeros, value.shape, value.dtype
#         )
#         cache_index = self.variable(
#             "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
#         )

#         if is_initialized:
#             *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
#             # update key, value caches with our new 1d spatial slices
#             cur_index = cache_index.value
#             indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
#             key = lax.dynamic_update_slice(cached_key.value, key, indices)
#             value = lax.dynamic_update_slice(cached_value.value, value, indices)
#             cached_key.value = key
#             cached_value.value = value
#             num_updated_cache_vectors = query.shape[1]
#             cache_index.value = cache_index.value + num_updated_cache_vectors
#             # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
#             pad_mask = jnp.broadcast_to(
#                 jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
#                 tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
#             )
#             attention_mask = combine_masks(pad_mask, attention_mask)
#         return key, value, attention_mask

#     def __call__(
#         self,
#         hidden_states,
#         attention_mask,
#         position_ids,
#         deterministic: bool = True,
#         init_cache: bool = False,
#         output_attentions: bool = False,
#         fcm_mask=None,
#     ):
#         xq, xk, xv = (
#             self.q_proj(hidden_states),
#             self.k_proj(hidden_states),
#             self.v_proj(hidden_states),
#         )

#         xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
#         xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
#         xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))

#         xq = self._split_heads(xq, self.num_heads)
#         xk = self._split_heads(xk, self.num_key_value_heads)
#         xv = self._split_heads(xv, self.num_key_value_heads)

#         freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)

#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)

#         dropout_rng = None
#         if not deterministic and self.config.attn_pdrop > 0.0:
#             dropout_rng = self.make_rng("dropout")

#         if self.config.scan_attention and not (
#             self.has_variable("cache", "cached_key") or init_cache
#         ):
#             # doesn't need blockwise attention if we are doing autoregressive decoding since no quadratic memory

#             # attention mask without nxn materlization, blockwise_attn will handle the rest
#             attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
#             # transform boolean mask into float mask
#             attention_bias = lax.select(
#                 attention_mask > 0,
#                 jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
#                 jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
#                     self.dtype
#                 ),
#             )

#             # for Gemma GQA
#             xk = jnp.repeat(xk, repeats=self.num_key_value_groups, axis=2)
#             xv = jnp.repeat(xv, repeats=self.num_key_value_groups, axis=2)

#             attn_weights = None
#             attn_output = blockwise_attn(
#                 xq,
#                 xk,
#                 xv,
#                 bias=attention_bias,
#                 deterministic=deterministic,
#                 dropout_rng=dropout_rng,
#                 attn_pdrop=self.config.attn_pdrop,
#                 causal=True,
#                 query_chunk_size=self.config.scan_query_chunk_size,
#                 key_chunk_size=self.config.scan_key_chunk_size,
#                 dtype=self.dtype,
#                 policy=get_gradient_checkpoint_policy("nothing_saveable"),
#                 precision=self.precision,
#                 float32_logits=True,
#                 prevent_cse=True,
#             )
#             attn_output = with_sharding_constraint(
#                 attn_output, PS(("dp", "fsdp"), None, "mp", None)
#             )
#         else:
#             query_length, key_length = xq.shape[1], xk.shape[1]

#             if self.has_variable("cache", "cached_key"):
#                 mask_shift = self.variables["cache"]["cache_index"]
#                 max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
#                 causal_mask = lax.dynamic_slice(
#                     self.causal_mask,
#                     (0, 0, mask_shift, 0),
#                     (1, 1, query_length, max_decoder_length),
#                 )
#             else:
#                 causal_mask = self.causal_mask[:, :, :query_length, :key_length]

#             batch_size = hidden_states.shape[0]
#             causal_mask = jnp.broadcast_to(
#                 causal_mask, (batch_size,) + causal_mask.shape[1:]
#             )

#             attention_mask = jnp.broadcast_to(
#                 jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
#             )
#             attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

#             # During fast autoregressive decoding, we feed one position at a time,
#             # and cache the keys and values step by step.
#             if self.has_variable("cache", "cached_key") or init_cache:
#                 xk, xv, attention_mask = self._concatenate_to_cache(
#                     xk, xv, xq, attention_mask
#                 )

#             # transform boolean mask into float mask
#             attention_bias = lax.select(
#                 attention_mask > 0,
#                 jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
#                 jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
#                     self.dtype
#                 ),
#             )

#             # for Gemma GQA
#             xk = jnp.repeat(xk, repeats=self.num_key_value_groups, axis=2)
#             xv = jnp.repeat(xv, repeats=self.num_key_value_groups, axis=2)

#             attn_weights = dot_product_attention_weights(
#                 xq,
#                 xk,
#                 bias=attention_bias,
#                 dropout_rng=dropout_rng,
#                 dropout_rate=self.config.attn_pdrop,
#                 deterministic=deterministic,
#                 dtype=jnp.promote_types(self.dtype, jnp.float32),
#                 precision=self.precision,
#             )
#             attn_weights = with_sharding_constraint(
#                 attn_weights, PS(("dp", "fsdp"), "mp", None, None)
#             )
#             attn_output = jnp.einsum(
#                 "...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision
#             )

#         attn_output = self._merge_heads(attn_output)
#         attn_output = self.o_proj(attn_output)
#         attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
#         outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
#         return outputs

### 공식 HF 복사 ###
# Copied from transformers.models.llama.modeling_flax_llama.rotate_half
def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]),
        axis=-1,
    )
    return rotate_half_tensor
def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])
# Copied from transformers.models.llama.modeling_flax_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)

# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaRotaryEmbedding with Llama->Gemma
class FlaxGemmaRotaryEmbedding(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    # Ignore copy
    def setup(self):
        head_dim = self.config.head_dim
        self.sincos = create_sinusoidal_positions(
            self.config.max_sequence_length, head_dim
        )

    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)

        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query

class FlaxGemmaAttention(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False
    param_dtype: jnp.dtype = jnp.float32 # For EasyLM
    precision: Optional[Union[jax.lax.Precision, str]] = None # For EasyLM

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        kernel = jax.nn.initializers.normal(self.config.initializer_range)
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=kernel,
            param_dtype=self.param_dtype, # For EasyLM
        )
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=kernel,
            param_dtype=self.param_dtype, # For EasyLM
        )
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=kernel,
            param_dtype=self.param_dtype, # For EasyLM
        )
        self.o_proj = nn.Dense(
            self.embed_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=kernel,
            param_dtype=self.param_dtype, # For EasyLM
        )

        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool"
        )
        self.rotary_emb = FlaxGemmaRotaryEmbedding(config, dtype=self.dtype)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads * self.head_dim,)
        )

    @nn.compact
    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, key.shape, key.dtype
        )
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, value.shape, value.dtype
        )
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        )

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)

        key, query = self.rotary_emb(key, query, position_ids)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )

        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(
                key, value, query, attention_mask
            )

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        key = jnp.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, repeats=self.num_key_value_groups, axis=2)

        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=attention_dtype,
        )

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
### 공식 HF 복사 ###
    
    

class FlaxGemmaMLP(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config

        self.gate_proj = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.down_proj = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.up_proj = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    # def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    # # x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
    # x = self.w2(nn.gelu(self.w1(x)) * self.w3(x)) # GELU for Gemmma
    # x = self.dropout(x, deterministic=deterministic)
    # return x

    def __call__(self, hidden_states, _=None):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = nn.silu(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


# class FlaxGemmaBlock(nn.Module):
#     config: GemmaConfig
#     dtype: jnp.dtype = jnp.float32
#     param_dtype: jnp.dtype = jnp.float32
#     precision: Optional[Union[jax.lax.Precision, str]] = None

#     def setup(self) -> None:
#         attention_module = FlaxGemmaAttention
#         mlp_module = FlaxGemmaMLP
#         if self.config.remat_attention != "":
#             attention_module = remat(
#                 FlaxGemmaAttention,
#                 static_argnums=(3, 4, 5),
#                 policy=get_gradient_checkpoint_policy(self.config.remat_attention),
#                 prevent_cse=True,
#             )
#         if self.config.remat_mlp != "":
#             mlp_module = remat(
#                 FlaxGemmaMLP,
#                 static_argnums=(1,),
#                 policy=get_gradient_checkpoint_policy(self.config.remat_mlp),
#                 prevent_cse=True,
#             )

#         self.self_attn = attention_module(
#             self.config,
#             dtype=self.dtype,
#             # param_dtype=self.param_dtype,
#             precision=self.precision,
#         )
#         self.mlp = mlp_module(
#             self.config,
#             dtype=self.dtype,
#             param_dtype=self.param_dtype,
#             precision=self.precision,
#         )
#         # self.attention_norm = RMSNorm(
#         self.input_layernorm = RMSNorm(
#             self.config.hidden_size,
#             eps=self.config.rms_norm_eps,
#             dtype=self.dtype,
#             param_dtype=self.param_dtype,
#         )
#         # self.ffn_norm = RMSNorm(
#         self.post_attention_layernorm = RMSNorm(
#             self.config.hidden_size,
#             eps=self.config.rms_norm_eps,
#             dtype=self.dtype,
#             param_dtype=self.param_dtype,
#         )

#     def __call__(
#         self,
#         hidden_states,
#         attention_mask=None,
#         position_ids=None,
#         deterministic: bool = True,
#         init_cache: bool = False,
#         output_attentions: bool = False,
#         fcm_mask: Optional[jnp.ndarray] = None,
#     ):
#         attn_outputs = self.self_attn(
#             self.input_layernorm(hidden_states),
#             attention_mask,
#             position_ids,
#             deterministic,
#             init_cache,
#             output_attentions,
#             # fcm_mask,
#         )
#         attn_output = attn_outputs[0]
#         hidden_states = hidden_states + attn_output

#         mlp_input = self.post_attention_layernorm(hidden_states)

#         if self.config.scan_mlp:
#             mlp_hidden_states = blockwise_ffn(
#                 self.mlp,
#                 mlp_input,
#                 self.config.scan_mlp_chunk_size,
#                 deterministic,
#             )
#         else:
#             mlp_hidden_states = self.mlp(
#                 mlp_input,
#                 deterministic,
#             )
#         mlp_hidden_states = with_sharding_constraint(
#             mlp_hidden_states, PS(("dp", "fsdp"), None, "mp")
#         )

#         hidden_states = hidden_states + mlp_hidden_states

#         return (hidden_states,) + attn_outputs[1:]




### FLAX MODEL HF꺼 카피 ###
class FlaxGemmaRMSNorm(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param(
            "weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size
        )

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return (1 + self.weight) * jnp.asarray(hidden_states, dtype=self.dtype)
# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaDecoderLayer with Llama->Gemma
class FlaxGemmaBlock(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.input_layernorm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)
        self.self_attn = FlaxGemmaAttention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)
        self.mlp = FlaxGemmaMLP(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states

        return (hidden_states,) + outputs[1:]
### FLAX MODEL HF꺼 카피 ###




class FlaxGemmaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GemmaConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: GemmaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, position_ids, return_dict=False
            )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
        )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`."
                )

            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTJAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxGemmaBlockCollection(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        block = FlaxGemmaBlock
        # if self.config.remat_block != "":
        #     block = remat(
        #         FlaxGemmaBlock,
        #         static_argnums=(3, 4, 5),
        #         policy=get_gradient_checkpoint_policy(self.config.remat_block),
        #     )
        self.blocks = [
            block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # if not deterministic and self.config.fcm_max_ratio > 0:
        #     # Apply forgetful causal mask
        #     batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
        #     fcm_ratio = jax.random.uniform(
        #         self.make_rng("fcm"),
        #         shape=(batch_size, 1, 1, 1),
        #         minval=self.config.fcm_min_ratio,
        #         maxval=self.config.fcm_max_ratio,
        #     )
        #     fcm_mask = (
        #         jax.random.uniform(
        #             self.make_rng("fcm"), shape=(batch_size, 1, 1, seq_length)
        #         )
        #         > fcm_ratio
        #     )
        #     fcm_mask = fcm_mask.at[:, :, :, 0].set(True)
        #     fcm_mask = fcm_mask.astype("bool")
        # else:
        #     fcm_mask = None
        fcm_mask = None # FIXME: Gemma FCM 안써

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
                # fcm_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxGPTJModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxGemmaModule(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.layers = FlaxGemmaBlockCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.norm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)

        # self.norm = RMSNorm(
        #     self.config.hidden_size,
        #     eps=self.config.rms_norm_eps,
        #     dtype=self.dtype,
        #     param_dtype=self.param_dtype,
        # )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # print(self.embed_tokens)
        input_embeds = self.embed_tokens(input_ids.astype("i4"))

        # FIXME: 이거 왜있지??????
        input_embeds = input_embeds * (self.config.hidden_size**0.5)

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


@add_start_docstrings("", "")
class FlaxGemmaModel(FlaxGemmaPreTrainedModel):
    module_class = FlaxGemmaModule


# append_call_sample_docstring(
#     FlaxGemmaModel,
#     _TOKENIZER_FOR_DOC,
#     _CHECKPOINT_FOR_DOC,
#     FlaxCausalLMOutput,
#     _CONFIG_FOR_DOC,
# )


class FlaxGemmaForCausalLMModule(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.model = FlaxGemmaModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length),
            )
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # if self.config.tie_word_embeddings:  # Gemma uses tied embeddings
        #     print("Use tie_word_embeddings")
        #     # shared_kernel = self.layers.variables["params"]["embed_tokens"]["embedding"].T
        #     # shared_kernel = self.model.embed_tokens.variables["params"][
        #     #     "embedding"
        #     # ].T  # For Gemma
        #     shared_kernel = self.model.variables["params"]["embed_tokens"][
        #         "embedding"
        #     ].T # For Gemma
        #     # print(shared_kernel)
        #     lm_logits = self.lm_head.apply(
        #         {"params": {"kernel": shared_kernel}}, hidden_states
        #     )
        # else:
        #     lm_logits = self.lm_head(hidden_states)
    
        print("Use tie_word_embeddings")
        shared_kernel = self.model.variables["params"]["embed_tokens"][
            "embedding"
        ].T # For Gemma
        lm_logits = self.lm_head.apply(
            {"params": {"kernel": shared_kernel}}, hidden_states
        )
        # lm_logits = self.lm_head(hidden_states)
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("", "")
class FlaxGemmaForCausalLM(FlaxGemmaPreTrainedModel):
    module_class = FlaxGemmaForCausalLMModule

    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: Optional[jax.Array] = None
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPTJ uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


# append_call_sample_docstring(
#     FlaxGPTJForCausalLM,
#     _TOKENIZER_FOR_DOC,
#     _CHECKPOINT_FOR_DOC,
#     FlaxCausalLMOutput,
#     _CONFIG_FOR_DOC,
# )


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {}


class GemmaTokenizer(PreTrainedTokenizer):
    """
    Construct a Gemma tokenizer. Based on byte-level Byte-Pair-Encoding.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=False,
        add_eos_token=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        super().__init__(
            bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs
        )
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)

        with tempfile.NamedTemporaryFile() as tfile:
            with open_file(self.vocab_file, "rb") as fin:
                tfile.write(fin.read())
                tfile.flush()
                tfile.seek(0)
            self.sp_model.Load(tfile.name)
        """ Initialisation"""
        self.add_special_tokens(
            dict(
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
            )
        )
        self.pad_token_id = self.unk_token_id

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp_model.eos_id()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def save_vocabulary(
        self, save_directory, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is not None:
            output = output + token_ids_1

        if self.add_eos_token:
            output = output + [self.eos_token_id]

        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]
