"""
OmniKV wrappers for Qwen3 model.
Mirrors modeling/omnikv.py but uses Qwen3 base classes.
"""
import math
import time

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from modeling.qwen3_modeling import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
    Qwen3Attention,
    Qwen3FlashAttention2,
    QWEN3_ATTENTION_CLASSES,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP
from modeling.spec_cache import OmniKVMultiStageCache, WOPackCache, get_cache_cls
from tiny_tools.log import logger, warning_once

last_call_t = time.time()


def time_analyze():
    global last_call_t
    temp = round(time.time() - last_call_t, 4)
    last_call_t = time.time()
    return temp


# ===================== Config =====================

class Qwen3CompressorConfig(Qwen3Config):
    """Qwen3 config with OmniKV compressor settings."""

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_config_of_compressor(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        if not hasattr(self, key):
            setattr(self, key, default)
            logger.warning(f"{key}不存在，被设置为{default}")
        return getattr(self, key, default)

    def _rope_scaling_validation(self):
        return


# ===================== Token Selection (adapted for Qwen3) =====================

def select_tokens_by_attn_universal_qwen3(
    raw_attn,
    hidden_states,
    position_ids,
    past_key_value,
    num_selected_tokens,
    consider_len,
    layer_idx=None,
    selector_cls="last",
):
    """Token selection using attention scores, adapted for Qwen3's rotary_emb interface and Q/K norm."""
    bsz, q_len, _ = hidden_states.size()
    assert past_key_value

    query_states = raw_attn.q_proj(hidden_states)
    key_states = raw_attn.k_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, raw_attn.num_heads, raw_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, raw_attn.num_key_value_heads, raw_attn.head_dim).transpose(1, 2)

    # Qwen3: apply Q/K normalization
    query_states = raw_attn.q_norm(query_states)
    key_states = raw_attn.k_norm(key_states)

    # Qwen3/Qwen2 rotary_emb interface: (x, seq_len=N) instead of Llama's (x, position_ids)
    kv_seq_len = position_ids.max().item() + 1
    cos, sin = raw_attn.rotary_emb(key_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # Use cached keys for scoring
    key_states = past_key_value.key_cache[raw_attn.layer_idx][:, :, :consider_len]
    key_states = repeat_kv(key_states, raw_attn.num_key_value_groups)

    if selector_cls == "last":
        attn_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)
        attn_score = torch.max(attn_score[..., -1, :], dim=1).values
        num_selected_tokens = min(num_selected_tokens, attn_score.shape[-1])
        v, idx = torch.topk(attn_score, k=num_selected_tokens, dim=-1, sorted=True)
    elif selector_cls == "softmax_before_last":
        attn_score = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)
        attn_score = torch.nn.functional.softmax(attn_score, dim=-1)
        attn_score = torch.max(attn_score[..., -1, :], dim=1).values
        num_selected_tokens = min(num_selected_tokens, attn_score.shape[-1])
        v, idx = torch.topk(attn_score, k=num_selected_tokens, dim=-1, sorted=True)
    elif selector_cls == "uniform":
        qs = torch.split(query_states, 1, dim=2)
        attn_sum = None
        for q in qs:
            attn_score = torch.matmul(q, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)
            attn_score = torch.nn.functional.softmax(attn_score, dim=-1)
            attn_score = torch.max(attn_score, dim=1).values
            attn_score = torch.sum(attn_score, dim=-2)
            if attn_sum is None:
                attn_sum = attn_score
            else:
                attn_sum += attn_score
        num_selected_tokens = min(num_selected_tokens, attn_sum.shape[-1])
        v, idx = torch.topk(attn_sum, k=num_selected_tokens, dim=-1, sorted=True)
    elif selector_cls == "exp":
        qs = torch.split(query_states, 1, dim=2)
        attn_sum = None
        for q in qs:
            attn_score = torch.matmul(q, key_states.transpose(2, 3)) / math.sqrt(raw_attn.head_dim)
            attn_score = torch.nn.functional.softmax(attn_score, dim=-1)
            attn_score = torch.max(attn_score, dim=1).values
            q_len_local = attn_score.shape[-2]
            alpha = 2 ** torch.arange(-q_len_local + 1, 1, device=attn_score.device)[None, :, None]
            attn_score = torch.sum(attn_score * alpha, dim=-2)
            if attn_sum is None:
                attn_sum = attn_score
            else:
                attn_sum = attn_sum * (2**-q_len_local) + attn_score
        num_selected_tokens = min(num_selected_tokens, attn_sum.shape[-1])
        v, idx = torch.topk(attn_sum, k=num_selected_tokens, dim=-1, sorted=True)
    else:
        raise NotImplementedError

    idx = torch.sort(idx, descending=False).values
    return idx


# ===================== OmniKV Layer =====================

class Qwen3OmniKVMulLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)

        self.config = config
        self.layer_idx = layer_idx
        self.prefill_len = None
        self.cache_cls = get_cache_cls(config)
        self.sparse_in_prefill = config.get("sparse_in_prefill", False)
        self.max_len_can_hold = config.get("max_len_can_hold", 32_000)
        self.attn_seg_sz = config.get("attn_seg_sz", 8000)
        self.do_select_layers = [
            int(i) for i in config.get("do_select_layers").split(",")
        ]
        self.hidden_state_window = None
        self.selector_cls = config.get("selector_cls", "softmax_before_last")
        self.window_size = config.get("window_size", 16)
        self.decode_step = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[OmniKVMultiStageCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if hidden_states.shape[1] > 1:
            # Accumulate prefill_len for chunked prefill support
            if self.prefill_len is None:
                self.prefill_len = hidden_states.shape[1]
            else:
                self.prefill_len += hidden_states.shape[1]
            if (
                "last" not in self.selector_cls
                and self.layer_idx in self.do_select_layers
            ):
                self.hidden_state_window = hidden_states[:, -self.window_size:]
                self.decode_step = 1
        if past_key_value:
            assert isinstance(past_key_value, self.cache_cls)

        consider_len = self.prefill_len
        num_selected_tokens = self.config.get("num_of_selected_tokens", 4096)
        if isinstance(num_selected_tokens, float):
            num_selected_tokens = int(num_selected_tokens * consider_len)
        if (
            hidden_states.shape[1] == 1
            and past_key_value
            and self.layer_idx in self.do_select_layers
        ):
            window_hs = hidden_states
            num_prefill_token_in_window = max(0, self.window_size - self.decode_step)
            if "last" not in self.selector_cls:
                self.hidden_state_window = torch.cat(
                    [self.hidden_state_window, hidden_states], dim=1
                )[:, -self.window_size:]
                window_hs = self.hidden_state_window
                consider_len -= num_prefill_token_in_window
                num_selected_tokens -= num_prefill_token_in_window
                num_selected_tokens = max(1, num_selected_tokens)
            idx = select_tokens_by_attn_universal_qwen3(
                self.self_attn,
                window_hs,
                position_ids,
                past_key_value,
                num_selected_tokens,
                consider_len,
                self.layer_idx,
                self.selector_cls,
            )
            if "last" not in self.selector_cls:
                idx = torch.cat(
                    [
                        idx,
                        torch.arange(
                            self.prefill_len - num_prefill_token_in_window,
                            self.prefill_len,
                            device=idx.device,
                        )[None, :].repeat(idx.shape[0], 1),
                    ],
                    dim=1,
                )
            if self.config.get("dense_more", False):
                past_key_value.set_idx_on_gpu(idx, self.layer_idx)
            else:
                raise ValueError("不支持dense_more=False")
            # Record selected KV count on first select layer
            if self.layer_idx == self.do_select_layers[0]:
                import infer as _infer_module
                _infer_module.last_inference_meta["num_selected_kv"] = idx.shape[-1]
            past_key_value.stage = "decoding"
            self.decode_step += 1

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hsl = torch.split(hidden_states, 4000, dim=1)
        hidden_states = [self.mlp(hs) for hs in hsl]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ===================== OmniKV Model =====================

class Qwen3OmniKVMulModel(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3OmniKVMulLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


# ===================== OmniKV LM =====================

class Qwen3OmniKVMulLM(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3OmniKVMulModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_context_len = config.get("max_context_len", 50_000)
        self.cache_cls = get_cache_cls(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        n = input_ids.shape[1]
        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if not isinstance(past_key_values, self.cache_cls):
            # Reset per-layer state for new generation
            for layer in self.model.layers:
                if hasattr(layer, 'prefill_len'):
                    layer.prefill_len = None
                    layer.decode_step = 0
                    layer.hidden_state_window = None
            kwargs = {}
            if (cache_cls_name := self.config.get("cache_cls", "default")) == "multi" or cache_cls_name == "without_pack":
                do_sel_layers = [int(i) for i in self.config.get("do_select_layers").split(",")]
                full_attn_layers = list(range(0, do_sel_layers[0])) + do_sel_layers + [self.config.num_hidden_layers]
                kwargs["full_attn_layers"] = full_attn_layers
                kwargs["num_hidden_layers"] = self.config.num_hidden_layers
                kwargs["num_wait_load_layers"] = self.config.get("num_wait_load_layers", 2)
                kwargs["real_offload"] = self.config.get("real_offload", True)
            else:
                raise NotImplementedError
            past_key_values = self.cache_cls.from_dynamic_cache(past_key_values, **kwargs)

        if n == 1:
            past_key_values.stage = "decoding"
        else:
            past_key_values.stage = "prefill"

        # Chunked prefill to avoid OOM on long sequences
        prefill_chunk_size = self.config.get("prefill_chunk_size", 8192)
        if n > 1 and n > prefill_chunk_size:
            # Process input in chunks, each chunk attends to all previous via KV cache
            for chunk_start in range(0, n, prefill_chunk_size):
                chunk_end = min(chunk_start + prefill_chunk_size, n)
                chunk_input_ids = input_ids[:, chunk_start:chunk_end]
                chunk_position_ids = position_ids[:, chunk_start:chunk_end] if position_ids is not None else None
                chunk_cache_position = cache_position[chunk_start:chunk_end] if cache_position is not None else None

                # For chunked prefill, attention_mask covers all tokens seen so far
                chunk_attention_mask = attention_mask[:, :chunk_end] if attention_mask is not None else None

                outputs = self.model(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    position_ids=chunk_position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=chunk_cache_position,
                )
                # Update past_key_values from outputs for next chunk
                if hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
                elif isinstance(outputs, tuple) and len(outputs) > 1:
                    past_key_values = outputs[1]
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        hidden_states = outputs[0][:, -1:]
        if getattr(self.config, "pretraining_tp", 1) > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            raise NotImplementedError

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if n > 1:
            logger.info(f"---prefill time {round(time_analyze(), 3)}s")
        else:
            logger.info(f"---decoding time {round(time_analyze(), 3)}s")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
