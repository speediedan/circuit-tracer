"""All three backends expose the embedding matrices under one name, in one orientation.

Code holding a `ReplacementModel` should be able to ask for the embedding or unembedding matrix
without first asking which backend it holds. Before this, only two of the three answered to
`unembed_weight`: TransformerLens had the matrix under `unembed.W_U` instead, so every generic
caller carried a `model.unembed.W_U if backend == "transformerlens" else model.unembed_weight`
branch, and Neuronpedia's graph server -- which duck-typed the same three spellings -- raised
"Model has neither 'unembed' nor 'lm_head' attribute" at request time against the interp_engine
backend.

The orientation matters as much as the name, and is the reason a plain `W_U` alias would not do:
TL stores the unembedding as `(d_model, d_vocab)` while the Hugging Face-shaped backends store its
transpose. `unembed_weight` is `(d_vocab, d_model)` on all three, one row per token, matching what
`embed_weight` already meant everywhere.

The orientation check builds a randomly initialized 8-dimensional model (it needs GPT-2's
tokenizer, from the cache or the Hub), and the rest inspects classes without instantiating them.
No weights or GPU.
"""

import torch

from circuit_tracer.replacement_model.replacement_model_interp_engine import (
    InterpEngineReplacementModel,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU
from tests.tl_bridge_configs import gpt2_tokenizer, hf_config_from_tl_dict

BACKENDS = [
    TransformerLensReplacementModel,
    NNSightReplacementModel,
    InterpEngineReplacementModel,
]

D_VOCAB = 13
D_MODEL = 8


def _declares(cls: type, name: str) -> bool:
    """Whether `cls` provides `name`, as a property or as an annotated instance attribute.

    nnsight assigns its two in `__init__` rather than computing them, so a `hasattr` on the class
    would miss it; the class-level annotation is what stands in for the property there.
    """
    if isinstance(getattr(cls, name, None), property):
        return True
    return any(name in vars(base).get("__annotations__", {}) for base in cls.__mro__)


def test_every_backend_declares_both_accessors():
    for cls in BACKENDS:
        for name in ("embed_weight", "unembed_weight"):
            assert _declares(cls, name), (
                f"{cls.__name__} does not declare `{name}`, so generic code holding a "
                "ReplacementModel has to branch on the backend again"
            )


def test_transformerlens_accessors_are_vocab_major():
    """The one backend that has to transpose. The other two store `(d_vocab, d_model)` already."""
    config = hf_config_from_tl_dict(
        dict(
            n_layers=1,
            d_model=D_MODEL,
            d_head=4,
            n_heads=2,
            d_mlp=16,
            d_vocab=D_VOCAB,
            original_architecture="LlamaForCausalLM",
        )
    )
    transcoders = TranscoderSet(
        {0: SingleLayerTranscoder(D_MODEL, 16, JumpReLU(torch.tensor(0.0), 0.1), 0)},
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
    )
    model = TransformerLensReplacementModel.from_config(
        config, transcoders, tokenizer=gpt2_tokenizer(), device="cpu"
    )

    assert tuple(model.W_U.shape) == (D_MODEL, D_VOCAB), (
        "TL's own W_U is expected to be d_model-major"
    )
    assert tuple(model.embed_weight.shape) == (D_VOCAB, D_MODEL)
    assert tuple(model.unembed_weight.shape) == (D_VOCAB, D_MODEL)

    token = 5
    assert torch.equal(model.unembed_weight[token], model.W_U[:, token])
    assert torch.equal(model.embed_weight[token], model.W_E[token])
