"""
Compatibility shim for transformers >=5.x where __all__ restricts wildcard imports.
All files that previously used `from transformers.models.llama.modeling_llama import *`
should now also `from modeling._llama_compat import *` to get the missing names.
"""
from typing import Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

# These may have been removed or relocated in newer transformers versions
try:
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
except ImportError:
    import torch

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if position_ids is not None and cos.dim() == 2:
            cos = cos[position_ids].unsqueeze(unsqueeze_dim)
            sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        else:
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

try:
    from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES
except ImportError:
    LLAMA_ATTENTION_CLASSES = {"eager": LlamaAttention}

try:
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_causal_attention_mask,
        _prepare_4d_causal_attention_mask_for_sdpa,
    )
except ImportError:
    pass

try:
    from transformers.models.llama.modeling_llama import _get_unpad_data
except ImportError:
    try:
        from transformers.modeling_flash_attention_utils import _get_unpad_data
    except ImportError:
        _get_unpad_data = None
