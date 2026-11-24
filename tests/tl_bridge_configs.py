"""Build the HuggingFace configs that the TransformerLens backend's tests now run on.

TransformerLens 4 removed ``HookedTransformer``, and the TransformerLens replacement model is now a
``TransformerBridge`` over a HuggingFace model, so it is built from a HuggingFace config. These
tests described their small random models as ``HookedTransformerConfig`` dicts; rather than restate
every one, :func:`hf_config_from_tl_dict` reads the fields that shape the model from such a dict.
"""

from typing import Any

import torch
from transformers import AutoTokenizer, Gemma2Config, LlamaConfig, PretrainedConfig

from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig

# The tests used GPT-2's tokenizer to avoid gated repositories; its ids above the small models'
# vocabularies are never produced because the tests pass token ids directly.
TOKENIZER_NAME = "gpt2"


def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def hf_config_from_tl_dict(cfg: dict) -> PretrainedConfig:
    """The HuggingFace config of the model a ``HookedTransformerConfig`` dict describes.

    Supports the two architectures these tests use. ``torch_dtype`` carries the dict's dtype.
    """
    architecture = cfg["original_architecture"]
    common: dict[str, Any] = dict(
        vocab_size=cfg["d_vocab"],
        hidden_size=cfg["d_model"],
        intermediate_size=cfg["d_mlp"],
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg.get("n_key_value_heads") or cfg["n_heads"],
        head_dim=cfg["d_head"],
        rms_norm_eps=cfg.get("eps", 1e-6),
        rope_theta=cfg.get("rotary_base", 10000.0),
        max_position_embeddings=min(cfg.get("n_ctx", 2048), 2048),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )
    if architecture == "Gemma2ForCausalLM":
        attn_scale = float(cfg.get("attn_scale", cfg["d_head"] ** 0.5))
        config: PretrainedConfig = Gemma2Config(
            **common,
            hidden_activation=cfg.get("act_fn", "gelu_pytorch_tanh"),
            query_pre_attn_scalar=round(attn_scale**2),
            attn_logit_softcapping=cfg.get("attn_scores_soft_cap") or None,
            final_logit_softcapping=cfg.get("output_logits_soft_cap") or None,
            sliding_window=cfg.get("window_size", 4096),
        )
        config._name_or_path = "google/gemma-2-2b"
    elif architecture == "LlamaForCausalLM":
        config = LlamaConfig(**common, hidden_act=cfg.get("act_fn", "silu"))
        config._name_or_path = "meta-llama/Llama-3.2-1B"
    else:
        raise ValueError(f"No HuggingFace config mapping for {architecture!r} in these tests")
    config.architectures = [architecture]
    config.torch_dtype = cfg.get("dtype", torch.float32)
    return config


def graph_config(cfg: dict) -> UnifiedConfig:
    """The config a ``Graph`` keeps, from a ``HookedTransformerConfig`` dict.

    A graph records the model's shape and names, nothing a model needs to run, so tests that only
    build graphs take this rather than a model config.
    """
    return UnifiedConfig(
        n_layers=cfg["n_layers"],
        d_model=cfg["d_model"],
        d_head=cfg["d_head"],
        n_heads=cfg["n_heads"],
        d_mlp=cfg["d_mlp"],
        d_vocab=cfg["d_vocab"],
        tokenizer_name=cfg.get("tokenizer_name", TOKENIZER_NAME),
        model_name=cfg["model_name"],
        original_architecture=cfg.get("original_architecture", "unknown"),
        n_key_value_heads=cfg.get("n_key_value_heads"),
        dtype=cfg.get("dtype"),
    )
