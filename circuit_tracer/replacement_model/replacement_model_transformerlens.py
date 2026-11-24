import copy
import warnings
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from functools import partial
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from circuit_tracer.attribution.context_transformerlens import AttributionContext
from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

# Type definition for an intervention tuple (layer, position, feature_idx, value)
Intervention = tuple[
    int | torch.Tensor,
    int | slice | torch.Tensor,
    int | torch.Tensor,
    int | float | torch.Tensor,
]

# The bridge's own names for the unembedding's input and output. HookedTransformer had neither,
# so this model used to wrap the unembedding to add `hook_pre`/`hook_post`; the bridge's
# `unembed.hook_in` is the final normalized residual and `unembed.hook_out` the logits before
# any soft cap, which are the two tensors those hooks carried.
UNEMBED_IN = "unembed.hook_in"
UNEMBED_OUT = "unembed.hook_out"


class _StopAtUnembed(Exception):
    """Raised at the unembedding's input to end a forward pass that needs no logits."""


class TransformerLensReplacementModel(TransformerBridge):
    """A replacement model on a TransformerLens ``TransformerBridge``.

    TransformerLens 4 removed ``HookedTransformer``, which this model used to subclass. The bridge
    wraps the HuggingFace model without converting its weights and exposes the same
    TransformerLens hook names over it, so the replacement model keeps its hooks, freezes and
    skip connections and only changes how the model is built. The bridge is built by
    TransformerLens and then given this class, so every bridge method stays available.

    Two names differ from the ``HookedTransformer`` version: the unembedding's hooks are the
    bridge's own ``unembed.hook_in``/``unembed.hook_out`` (:data:`UNEMBED_IN`,
    :data:`UNEMBED_OUT`), and a transcoder set's hooks are the bridge's, which carry the
    HuggingFace tensors (``mlp.hook_in``/``mlp.hook_out`` are the MLP's input and output, and
    ``ln2.hook_normalized`` the unweighted normalized residual).
    """

    transcoders: TranscoderSet | CrossLayerTranscoder  # Support both types
    feature_input_hook: str
    feature_output_hook: str
    skip_transcoder: bool
    scan_name: str | list[str] | None
    backend: Literal["transformerlens"]

    @classmethod
    def from_bridge(
        cls,
        bridge: TransformerBridge,
        transcoders: TranscoderSet | CrossLayerTranscoder,
    ) -> "TransformerLensReplacementModel":
        """Turn a TransformerBridge into a replacement model with the given transcoders, in place.

        Args:
            bridge (TransformerBridge): a bridge over a causal LM, as TransformerLens builds it
            transcoders (TranscoderSet | CrossLayerTranscoder): the transcoders to splice in

        Returns:
            TransformerLensReplacementModel: the same object, now a replacement model
        """
        if type(bridge) is not TransformerBridge:
            raise TypeError(
                f"from_bridge takes a TransformerBridge as TransformerLens builds it, got "
                f"{type(bridge).__name__}"
            )
        bridge.__class__ = cls
        model: TransformerLensReplacementModel = bridge  # type: ignore[assignment]
        model._configure_replacement_model(transcoders)
        return model

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        transcoders: TranscoderSet | CrossLayerTranscoder,  # Accept both
        tokenizer=None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "TransformerLensReplacementModel":
        """Create a TransformerLensReplacementModel from a HuggingFace config, with fresh weights.

        Args:
            config (PretrainedConfig): the HuggingFace config of the causal LM. It is copied, so
                the caller's config is left as it was.
            transcoders (TranscoderSet): The transcoder set with configuration
            tokenizer: the tokenizer; loaded from ``config._name_or_path`` when not given
            device: where to build the model; the default device when None
            dtype: the model's dtype
            **kwargs: passed to ``TransformerBridge.boot_transformers``

        Returns:
            TransformerLensReplacementModel: The loaded TransformerLensReplacementModel
        """
        config = copy.deepcopy(config)
        config._attn_implementation = "eager"  # type: ignore[attr-defined]
        hf_model = AutoModelForCausalLM.from_config(config).to(dtype=dtype)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)  # type: ignore[attr-defined]
        bridge = TransformerBridge.boot_transformers(
            config._name_or_path,  # type: ignore[attr-defined]
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device if device is not None else get_default_device(),
            dtype=dtype,
            **kwargs,
        )
        return cls.from_bridge(bridge, transcoders)

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,  # Accept both
        **kwargs,
    ) -> "TransformerLensReplacementModel":
        """Create a TransformerLensReplacementModel from a HuggingFace model name and transcoders

        The bridge runs the HuggingFace weights as they are (no LayerNorm folding or centering), as
        the transcoders were trained on them.

        Args:
            model_name (str): the HuggingFace name of the pretrained model
            transcoders (TranscoderSet): The transcoder set with configuration
            **kwargs: passed to ``TransformerBridge.boot_transformers``

        Returns:
            TransformerLensReplacementModel: The loaded TransformerLensReplacementModel
        """
        bridge = TransformerBridge.boot_transformers(model_name, **kwargs)
        return cls.from_bridge(bridge, transcoders)

    @classmethod
    def from_pretrained(  # type:ignore
        cls,
        model_name: str,
        transcoder_set: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
        **kwargs,
    ) -> "TransformerLensReplacementModel":
        """Create a TransformerLensReplacementModel from model name and transcoder config

        Args:
            model_name (str): the HuggingFace name of the pretrained model
            transcoder_set (str): Either a predefined transcoder set name, or a config file
            device (torch.device | None): The device to load the model and transcoders on.
                If None, uses the default device. Defaults to None.
            dtype (torch.dtype): The dtype to use for the model and transcoders.
                Defaults to torch.float32.
            lazy_encoder (bool): Whether to lazily load encoder weights. If True, encoder
                weights are not loaded into memory until needed. Defaults to False.
            lazy_decoder (bool): Whether to lazily load decoder weights. If True, decoder
                weights are not loaded into memory until needed. Defaults to True.
            **kwargs: Additional keyword arguments passed to ``TransformerBridge.boot_transformers``

        Returns:
            TransformerLensReplacementModel: The loaded TransformerLensReplacementModel
        """
        if device is None:
            device = get_default_device()

        (
            transcoders,
            _,
        ) = load_transcoder_from_hub(transcoder_set, device=device, dtype=dtype)  # type: ignore

        return cls.from_pretrained_and_transcoders(
            model_name,
            transcoders,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def _configure_replacement_model(self, transcoder_set: TranscoderSet | CrossLayerTranscoder):
        self.backend = "transformerlens"
        transcoder_set.to(self.cfg.device, self.cfg.dtype)

        # special case to zero out <bos><start_of_turn>user\n for gemmascope 2 (-it) transcoders
        gemma_3_it = "gemma-3" in self.cfg.model_name and self.cfg.model_name.endswith("-it")
        self.zero_positions = slice(0, 4) if gemma_3_it else slice(0, 1)

        self.transcoders = transcoder_set
        self.feature_input_hook = transcoder_set.feature_input_hook
        self.original_feature_output_hook = transcoder_set.feature_output_hook
        self.feature_output_hook = transcoder_set.feature_output_hook + ".hook_out_grad"
        self.skip_transcoder = transcoder_set.skip_connection
        self.scan_name = transcoder_set.scan_name

        for hook in (self.feature_input_hook, self.original_feature_output_hook):
            missing = [
                f"blocks.{layer}.{hook}"
                for layer in range(self.cfg.n_layers)
                if f"blocks.{layer}.{hook}" not in self.hook_dict
            ]
            if missing:
                raise ValueError(
                    f"The transcoders read or write at {hook!r}, which this model's TransformerBridge "
                    f"does not have ({missing[0]} and {len(missing) - 1} more are missing)."
                )

        self._configure_gradient_flow()

    def _hook_point(self, name: str) -> HookPoint:
        """The HookPoint the bridge resolves ``name`` to (canonical or alias)."""
        return self.hook_dict[name]

    def _register_hook_point(self, name: str, hook_point: HookPoint) -> None:
        """Make a HookPoint the model created itself addressable by name, like the bridge's own."""
        hook_point.name = name
        self._hook_registry[name] = hook_point

    def _configure_gradient_flow(self):
        for layer in range(self.cfg.n_layers):
            self._configure_skip_connection(self.transcoders, layer)

        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
        self.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore

        for param in self.parameters():
            param.requires_grad = False

        def enable_gradient(acts, hook):
            acts.requires_grad = True
            return acts

        self._hook_point("hook_embed").add_hook(enable_gradient, is_permanent=True)  # type: ignore

    def _configure_skip_connection(
        self, transcoders: TranscoderSet | CrossLayerTranscoder, layer: int
    ):
        cached = {}

        def cache_activations(acts, hook):
            cached["acts"] = acts

        def add_skip_connection(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint):
            # We add grad_hook because we need a way to hook into the gradients of the output
            # of this function. If we put the backwards hook here at hook, the grads will be 0
            # because we detached acts.
            skip_input_activation = cached.pop("acts")
            if transcoders.skip_connection:
                skip = transcoders.compute_skip(layer, skip_input_activation)
            else:
                skip = skip_input_activation * 0
            return grad_hook(skip + (acts - skip).detach())

        # add feature input hook
        self._hook_point(f"blocks.{layer}.{self.feature_input_hook}").add_hook(
            cache_activations,  # type: ignore
            is_permanent=True,
        )

        # add feature output hook and special grad hook
        output_hook = self._hook_point(f"blocks.{layer}.{self.original_feature_output_hook}")
        grad_hook = HookPoint()
        output_hook.hook_out_grad = grad_hook
        self._register_hook_point(f"blocks.{layer}.{self.feature_output_hook}", grad_hook)
        output_hook.add_hook(
            partial(add_skip_connection, grad_hook=grad_hook),  # type: ignore
            is_permanent=True,
        )

    def _hook_names(self) -> list[str]:
        """Every HookPoint once, by its canonical name (``hook_dict`` also lists the aliases)."""
        names: dict[int, str] = {}
        for name, hook_point in self.hook_dict.items():
            names.setdefault(id(hook_point), hook_point.name or name)
        return list(names.values())

    def forward_trunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run everything except the unembedding, returning the normalized final residual.

        Attribution runs a wide batch (one row per source node) and never needs logits from it,
        so materializing ``[batch, pos, vocab]`` would dominate memory for no purpose. The
        bridge's ``stop_at_layer`` stops before a block, so it cannot stop after the final norm;
        the pass is ended at the unembedding's input instead, after every hook registered there
        before this call has run.
        """
        captured: dict[str, torch.Tensor] = {}

        def stop(acts, hook):
            captured["resid"] = acts
            raise _StopAtUnembed

        try:
            # a nested hooks() context removes only its own hook on exit
            with self.hooks(fwd_hooks=[(UNEMBED_IN, stop)]):
                self.forward(input_ids)
        except _StopAtUnembed:
            pass
        if "resid" not in captured:
            raise RuntimeError(
                f"The forward pass never reached {UNEMBED_IN}, so the trunk has no output"
            )
        return captured["resid"]

    def _output_softcap(self) -> float:
        """The logit soft cap the HuggingFace model applies after the unembedding (0 for none)."""
        return float(getattr(self.original_model.config, "final_logit_softcapping", None) or 0.0)

    def _get_activation_caching_hooks(
        self,
        sparse: bool = False,
        apply_activation_function: bool = True,
        append: bool = False,
    ) -> tuple[list[torch.Tensor], list[tuple[str, Callable]]]:
        activation_matrix = (
            [[] for _ in range(self.cfg.n_layers)] if append else [None] * self.cfg.n_layers
        )

        def cache_activations(acts, hook, layer):
            transcoder_acts = (
                self.transcoders.encode_layer(
                    acts, layer, apply_activation_function=apply_activation_function
                )
                .detach()
                .squeeze(0)
            )

            if not append:
                transcoder_acts[self.zero_positions] = 0

            if sparse:
                transcoder_acts = transcoder_acts.to_sparse()

            if append:
                activation_matrix[layer].append(transcoder_acts)
            else:
                activation_matrix[layer] = transcoder_acts  # type: ignore

        activation_hooks = [
            (
                f"blocks.{layer}.{self.feature_input_hook}",
                partial(cache_activations, layer=layer),
            )
            for layer in range(self.cfg.n_layers)
        ]
        return activation_matrix, activation_hooks  # type: ignore

    def get_activations(
        self,
        inputs: str | torch.Tensor,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the transcoder activations for a given prompt

        Args:
            inputs (str | torch.Tensor): The inputs you want to get activations over
            sparse (bool, optional): Whether to return a sparse tensor of activations.
                Useful if d_transcoder is large. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the model logits on the inputs and the
                associated activation cache
        """

        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            apply_activation_function=apply_activation_function,
        )
        with torch.inference_mode(), self.hooks(activation_hooks):  # type: ignore
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache

    @contextmanager
    def zero_softcap(self):
        # the HuggingFace model applies the cap after the unembedding, reading its config each call
        config: PretrainedConfig = self.original_model.config  # type: ignore
        current_softcap = getattr(config, "final_logit_softcapping", None)
        try:
            config.final_logit_softcapping = None  # type: ignore
            yield
        finally:
            config.final_logit_softcapping = current_softcap  # type: ignore

    def ensure_tokenized(self, prompt: str | torch.Tensor | list[int]) -> torch.Tensor:
        """Convert prompt to 1-D tensor of token ids with proper special token handling.

        This method ensures that a special token (BOS/PAD) is prepended to the input sequence.
        The first token position in transformer models typically exhibits unusually high norm
        and an excessive number of active features due to how models process the beginning of
        sequences. By prepending a special token, we ensure that actual content tokens have
        more consistent and interpretable feature activations, avoiding the artifacts present
        at position 0. This prepended token is later ignored during attribution analysis.

        Args:
            prompt: String, tensor, or list of token ids representing a single sequence

        Returns:
            1-D tensor of token ids with BOS/PAD token at the beginning

        Raises:
            TypeError: If prompt is not str, tensor, or list
            ValueError: If tensor has wrong shape (must be 1-D or 2-D with batch size 1)
        """

        if isinstance(prompt, str):
            tokens = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.squeeze(0)  # type: ignore
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if tokens.ndim > 1:
            raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

        tokens = tokens.to(self.cfg.device)

        gemma_3_it = "gemma-3" in self.cfg.model_name and self.cfg.model_name.endswith("-it")
        if gemma_3_it:
            ignore_prefix = torch.tensor(
                [2, 105, 2364, 107], dtype=tokens.dtype, device=tokens.device
            )
            tokenization_error = (
                "Input tokens should start with <bos><start_of_turn>user\n, but got {tokens}"
            )
            assert tokens.size(0) >= 4 and torch.all(tokens[:4] == ignore_prefix), (
                tokenization_error.format(tokens=self.tokenizer.decode(tokens.cpu().tolist()))  # type: ignore
            )
            return tokens

        # Check if a special token is already present at the beginning
        if tokens[0] in self.tokenizer.all_special_ids:  # type: ignore
            return tokens

        # Prepend a special token to avoid artifacts at position 0
        candidate_bos_token_ids = [
            self.tokenizer.bos_token_id,  # type: ignore
            self.tokenizer.pad_token_id,  # type: ignore
            self.tokenizer.eos_token_id,  # type: ignore
        ]
        candidate_bos_token_ids += self.tokenizer.all_special_ids  # type: ignore

        dummy_bos_token_id = next(filter(None, candidate_bos_token_ids))
        if dummy_bos_token_id is None:
            warnings.warn(
                "No suitable special token found for BOS token replacement. The first token will be ignored."
            )
        else:
            tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])

        return tokens.to(self.cfg.device)

    @torch.no_grad()
    def setup_attribution(self, inputs: str | torch.Tensor):
        """Precomputes the transcoder activations and error vectors, saving them and the
        token embeddings.

        Args:
            inputs (str): the inputs to attribute - hard coded to be a single string (no
                batching) for now
        """

        if isinstance(inputs, str):
            tokens = self.ensure_tokenized(inputs)
        else:
            tokens = inputs.squeeze()

        assert isinstance(tokens, torch.Tensor), "Tokens must be a tensor"
        assert tokens.ndim == 1, "Tokens must be a 1D tensor"

        # Named per layer rather than matched by substring: the bridge's hook_dict also lists
        # aliases, and a substring match would cache an aliased point twice.
        layers = range(self.cfg.n_layers)
        in_names = [f"blocks.{layer}.{self.feature_input_hook}" for layer in layers]
        out_names = [f"blocks.{layer}.{self.feature_output_hook}" for layer in layers]
        cache: dict[str, torch.Tensor] = {}

        def keep(acts, hook, name):
            cache[name] = acts.detach()

        fwd_hooks = [
            (name, partial(keep, name=name)) for name in [*in_names, *out_names, "hook_embed"]
        ]
        logits = self.run_with_hooks(tokens, fwd_hooks=fwd_hooks)  # type: ignore

        mlp_in_cache = torch.cat([cache[name] for name in in_names], dim=0)
        mlp_out_cache = torch.cat([cache[name] for name in out_names], dim=0)

        attribution_data = self.transcoders.compute_attribution_components(
            mlp_in_cache, self.zero_positions
        )

        # Compute error vectors
        error_vectors = mlp_out_cache - attribution_data["reconstruction"]

        error_vectors[:, self.zero_positions] = 0
        # The tensor at hook_embed, whose gradient the attribution reads: on Gemma it is the
        # embedding before or after the sqrt(d_model) normalizer according to where the bridge
        # places the hook, so it is read rather than rebuilt from W_E.
        token_vectors = cache["hook_embed"][0]  # (n_pos, d_model)

        return AttributionContext(
            activation_matrix=attribution_data["activation_matrix"],
            logits=logits,
            error_vectors=error_vectors,
            token_vectors=token_vectors,
            decoder_vecs=attribution_data["decoder_vecs"],
            encoder_vecs=attribution_data["encoder_vecs"],
            encoder_to_decoder_map=attribution_data["encoder_to_decoder_map"],
            decoder_locations=attribution_data["decoder_locations"],
        )

    def setup_intervention_with_freeze(
        self,
        inputs: str | torch.Tensor,
        constrained_layers: range | None = None,
    ) -> tuple[torch.Tensor, list[tuple[str, Callable]]]:
        """Sets up an intervention with either frozen attention + LayerNorm(default) or frozen
        attention, LayerNorm, and MLPs, for constrained layers

        Args:
            inputs (str | torch.Tensor): The inputs to intervene on
            constrained_layers: (tuple[int,int] | range | None = None): Whether to freeze not just attention, but also
                LayerNorm and MLPs, at the specified layer range. Defaults to None.

        Returns:
            list[tuple[str, Callable]]: The freeze hooks needed to run the desired intervention.
        """

        hookpoints_to_freeze = ["hook_pattern"]
        if constrained_layers:
            if set(range(self.cfg.n_layers)).issubset(set(constrained_layers)):
                hookpoints_to_freeze.append("hook_scale")
            hookpoints_to_freeze.append(self.feature_output_hook)
            if self.skip_transcoder:
                hookpoints_to_freeze.append(self.feature_input_hook)

        # only freeze outputs in constrained range
        selected_hook_points = []
        # canonical names only: a point frozen under two names would be cached and replaced twice
        for hook_point in self._hook_names():
            hook_obj = self._hook_point(hook_point)
            if any(
                hookpoint_to_freeze in hook_point for hookpoint_to_freeze in hookpoints_to_freeze
            ):
                # don't freeze feature outputs if the layer is not in the constrained range
                if (
                    self.feature_output_hook in hook_point
                    and constrained_layers
                    and hook_obj.layer() not in constrained_layers
                ):
                    continue
                selected_hook_points.append(hook_point)

        freeze_cache, cache_hooks, _ = self.get_caching_hooks(names_filter=selected_hook_points)

        original_activations, activation_caching_hooks = self._get_activation_caching_hooks()
        self.run_with_hooks(inputs, fwd_hooks=cache_hooks + activation_caching_hooks)

        def freeze_hook(activations, hook):
            cached_values = freeze_cache[hook.name]

            assert activations.shape == cached_values.shape, (
                f"Activations shape {activations.shape} does not match cached values"
                f" shape {cached_values.shape} at hook {hook.name}"
            )
            return cached_values

        fwd_hooks = [
            (hookpoint, freeze_hook)
            for hookpoint in freeze_cache.keys()
            if self.feature_input_hook not in hookpoint
        ]

        if not (constrained_layers and self.skip_transcoder):
            return torch.stack(original_activations), fwd_hooks

        skip_diffs = {}

        def diff_hook(activations, hook, layer: int):
            # The MLP hook out freeze hook sets the value of the MLP to the value it
            # had when run on the inputs normally. We subtract out the skip that
            # corresponds to such a run, and add in the skip with direct effects.
            frozen_skip = self.transcoders.compute_skip(layer, freeze_cache[hook.name])
            normal_skip = self.transcoders.compute_skip(layer, activations)

            skip_diffs[layer] = normal_skip - frozen_skip

        def add_diff_hook(activations, hook, layer: int):
            return activations + skip_diffs[layer]

        fwd_hooks += [
            (
                f"blocks.{layer}.{self.feature_input_hook}",
                partial(diff_hook, layer=layer),
            )
            for layer in constrained_layers
        ]
        fwd_hooks += [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(add_diff_hook, layer=layer),
            )
            for layer in constrained_layers
        ]
        return torch.stack(original_activations), fwd_hooks

    def _get_feature_intervention_hooks(
        self,
        inputs: str | torch.Tensor,
        interventions: Sequence[Intervention],
        constrained_layers: range | None = None,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
        sparse: bool = False,
        using_past_kv_cache: bool = False,
        return_activations: bool = True,
    ):
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, allowing all effects to propagate (optionally allowing its effects to
        propagate through transcoders)

        Args:
            input (_type_): the input prompt to intervene on
            intervention_dict (list[Intervention]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            constrained_layers (range | tuple | None): whether to apply interventions only to a certain range.
                Mostly applicable to CLTs. If the given range includes all model layers,
                we also freeze layernorm denominators, computing direct effects. None means no
                constraints (iterative patching)
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
            using_past_kv_cache (bool): whether we are generating with past_kv_cache, meaning that
                n_pos is 1, and we must append onto the existing logit / activation cache if the
                hooks are run multiple times. Defaults to False
            return_activations (bool): Whether to compute and return feature activations. If False,
                activation computation is skipped for layers not being intervened on (when
                constrained_layers is not set), saving time. Activations are not returned.
                Defaults to True.
        """

        interventions_by_layer = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            interventions_by_layer[layer].append((pos, feature_idx, value))

        if using_past_kv_cache:
            # We're generating one token at a time
            original_activations, freeze_hooks = [], []
            n_pos = 1
        elif (freeze_attention or constrained_layers) and interventions:
            original_activations, freeze_hooks = self.setup_intervention_with_freeze(
                inputs, constrained_layers=constrained_layers
            )
            n_pos = original_activations.size(1)
        else:
            original_activations, freeze_hooks = [], []
            if isinstance(inputs, torch.Tensor):
                n_pos = inputs.size(0)
            else:
                n_pos = len(self.tokenizer(inputs).input_ids)  # type: ignore

        layer_deltas = torch.zeros(
            [self.cfg.n_layers, n_pos, self.cfg.d_model],
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        # This activation cache will fill up during our forward intervention pass
        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            apply_activation_function=apply_activation_function,
            sparse=sparse,
            append=using_past_kv_cache,
        )

        if not return_activations:
            new_activation_hooks = []
            if not constrained_layers:
                for loc, hook in activation_hooks:
                    layer = int(loc.split(".")[1])
                    if layer in interventions_by_layer:
                        new_activation_hooks.append((loc, hook))
            activation_hooks = new_activation_hooks

        def calculate_delta_hook(activations, hook, layer: int, layer_interventions):
            if constrained_layers:
                # base deltas on original activations; don't let effects propagate
                transcoder_activations = original_activations[layer]
            else:
                # recompute deltas based on current activations
                transcoder_activations = (
                    activation_cache[layer][-1] if using_past_kv_cache else activation_cache[layer]
                )
                if transcoder_activations.is_sparse:
                    transcoder_activations = transcoder_activations.to_dense()

                if not apply_activation_function:
                    transcoder_activations = self.transcoders.apply_activation_function(
                        layer, transcoder_activations.unsqueeze(0)
                    ).squeeze(0)

            activation_deltas = torch.zeros_like(transcoder_activations)
            for pos, feature_idx, value in layer_interventions:
                activation_deltas[pos, feature_idx] = (
                    value - transcoder_activations[pos, feature_idx]
                )

            poss, feature_idxs = activation_deltas.nonzero(as_tuple=True)
            new_values = activation_deltas[poss, feature_idxs]

            decoder_vectors = self.transcoders._get_decoder_vectors(layer, feature_idxs)

            if decoder_vectors.ndim == 2:
                # Single-layer transcoder case: [n_feature_idxs, d_model]
                decoder_vectors = decoder_vectors * new_values.unsqueeze(1)
                layer_deltas[layer].index_add_(0, poss, decoder_vectors)
            else:
                # Cross-layer transcoder case: [n_feature_idxs, n_remaining_layers, d_model]
                decoder_vectors = decoder_vectors * new_values.unsqueeze(-1).unsqueeze(-1)

                # Transpose to [n_remaining_layers, n_feature_idxs, d_model]
                decoder_vectors = decoder_vectors.transpose(0, 1)

                # Distribute decoder vectors across layers
                n_remaining_layers = decoder_vectors.shape[0]
                layer_deltas[-n_remaining_layers:].index_add_(1, poss, decoder_vectors)

        def intervention_hook(activations, hook, layer: int):
            new_acts = activations
            if layer in intervention_range:
                new_acts = new_acts + layer_deltas[layer]
            layer_deltas[layer] *= 0  # clearing this is important for multi-token generation
            return new_acts

        delta_hooks = [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(
                    calculate_delta_hook,
                    layer=layer,
                    layer_interventions=layer_interventions,
                ),
            )
            for layer, layer_interventions in interventions_by_layer.items()
        ]

        intervention_range = constrained_layers if constrained_layers else range(self.cfg.n_layers)
        intervention_hooks = [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(intervention_hook, layer=layer),
            )
            for layer in intervention_range
        ]

        all_hooks = freeze_hooks + activation_hooks + delta_hooks + intervention_hooks
        cached_logits = []

        def logit_cache_hook(activations, hook):
            # we need to manually apply the softcap (if used by the model), as it comes post-hook
            softcap = self._output_softcap()
            if softcap > 0.0:
                logits = softcap * F.tanh(activations / softcap)
            else:
                logits = activations.clone()
            cached_logits.append(logits)

        all_hooks.append((UNEMBED_OUT, logit_cache_hook))

        return all_hooks, cached_logits, activation_cache

    @torch.no_grad
    def feature_intervention(
        self,
        inputs: str | torch.Tensor,
        interventions: Sequence[Intervention],
        constrained_layers: range | None = None,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
        sparse: bool = False,
        return_activations: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, and returns the logits and feature activations. If freeze_attention or
        constrained_layers is True, attention patterns will be frozen, along with MLPs and
        LayerNorms. If constrained_layers is set, the effects of intervention will not propagate
        through the constrained layers, and CLTs will write only to those layers. Otherwise, the
        effects of the intervention will propagate through transcoders / LayerNorms

        Args:
            input (_type_): the input prompt to intervene on
            interventions (list[tuple[int, int, slice | torch.Tensor], int,
                int | torch.Tensor]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            constrained_layers (range | tuple | None): whether to apply interventions only to a certain range.
                Mostly applicable to CLTs.
            freeze_attention (bool): whether to freeze all attention patterns an layernorms
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
            return_activations (bool): Whether to compute and return feature activations. If False,
                activation computation is skipped for layers not being intervened on (when
                constrained_layers is not set), saving time. Activations are not returned.
                Defaults to True.
        """

        hooks, _, activation_cache = self._get_feature_intervention_hooks(
            inputs,
            interventions,
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            sparse=sparse,
            return_activations=return_activations,
        )

        with self.hooks(hooks):  # type: ignore
            logits = self(inputs)

        activation_cache = torch.stack(activation_cache) if return_activations else None

        return logits, activation_cache

    def _convert_open_ended_interventions(
        self,
        interventions: Sequence[Intervention],
    ) -> list[Intervention]:
        """Convert open-ended interventions into position-0 equivalents.

        An intervention is *open-ended* if its position component is a ``slice`` whose
        ``stop`` attribute is ``None`` (e.g. ``slice(1, None)``). Such interventions will
        also apply to tokens generated in an open-ended generation loop. In such cases,
        when use_past_kv_cache=True, the model only runs the most recent token
        (and there is thus only 1 position).
        """
        converted = []
        for layer, pos, feature_idx, value in interventions:
            if isinstance(pos, slice) and pos.stop is None:
                converted.append((layer, 0, feature_idx, value))
        return converted

    @torch.no_grad
    def feature_intervention_generate(
        self,
        inputs: str | torch.Tensor,
        interventions: Sequence[Intervention],
        constrained_layers: range | None = None,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
        sparse: bool = False,
        return_activations: bool = True,
        **kwargs,
    ) -> tuple[str, torch.Tensor, torch.Tensor | None]:  # logits: (seq_len, vocab_size)
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, and generates a continuation, along with the logits and activations at
        each generation position.
        This function accepts all kwargs valid for TransformerBridge.generate(). Note that
        freeze_attention applies only to the first token generated.

        Note that if kv_cache is True (default), generation will be faster, as the model will cache the KVs, and only
        process the one new token per step; if it is False, the model will generate by doing a full forward pass across
        all tokens. Note that due to numerical precision issues, you are only guaranteed that the logits / activations of
        model.feature_intervention_generate(s, ...) are equivalent to model.feature_intervention(s, ...) if kv_cache is False.

        .. note::
            Unlike ``feature_intervention`` (which returns 3-D logits of shape
            ``(batch, seq_len, vocab_size)``), this method returns **2-D** logits of shape
            ``(seq_len, vocab_size)`` with no batch dimension, since generation is always
            batch=1. This was changed in commit a1ca600.

        Args:
            input (_type_): the input prompt to intervene on
            interventions (list[tuple[int, int, slice | torch.Tensor], int,
                int | torch.Tensor]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            constrained_layers: (tuple[int,int] | range | None = None): whether to freeze all MLPs/transcoders /
                attn patterns / layernorm denominators. This will only apply to the very first token generated.
            freeze_attention (bool): whether to freeze all attention patterns.
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
            return_activations (bool): Whether to compute and return feature activations. If False,
                activation computation is skipped for layers not being intervened on (when
                constrained_layers is not set), saving time. Returns None for activations.
                Defaults to True.

        Returns:
            tuple[str, torch.Tensor, torch.Tensor | None]: A tuple of (generated_text,
                logits, activations) where logits has shape ``(seq_len, vocab_size)`` (2-D).
        """

        feature_intervention_hook_output = self._get_feature_intervention_hooks(
            inputs,
            interventions,
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            sparse=sparse,
            return_activations=return_activations,
        )

        hooks, logit_cache, activation_cache = feature_intervention_hook_output

        assert kwargs.get("use_past_kv_cache", True), (
            "Generation is only possible with use_past_kv_cache=True"
        )
        # Next, convert any open-ended interventions so they target position `0` (the
        # only token present during the incremental forward passes performed by
        # `generate`) and build the corresponding hooks.
        open_ended_interventions = self._convert_open_ended_interventions(interventions)

        # get new hooks that will target pos 0 / append logits / acts to the cache (not overwrite)
        open_ended_hooks, open_ended_logits, open_ended_activations = (
            self._get_feature_intervention_hooks(
                inputs,
                open_ended_interventions,
                constrained_layers=None,
                freeze_attention=False,
                apply_activation_function=apply_activation_function,
                sparse=sparse,
                using_past_kv_cache=True,
                return_activations=return_activations,
            )
        )

        # at the end of the model, clear original hooks and add open-ended hooks
        def clear_and_add_hooks(tensor, hook):
            self.reset_hooks()
            for open_ended_name, open_ended_hook in open_ended_hooks:
                self.add_hook(open_ended_name, open_ended_hook)

        for name, hook in hooks:
            self.add_hook(name, hook)

        self.add_hook(UNEMBED_OUT, clear_and_add_hooks)

        # the bridge's generate reads a 1-D token tensor as a batch of one-token prompts
        prompt = (
            inputs.unsqueeze(0) if isinstance(inputs, torch.Tensor) and inputs.ndim == 1 else inputs
        )
        generation: str = self.generate(prompt, **kwargs)  # type:ignore
        self.reset_hooks()

        logits = torch.cat((logit_cache[0][:, -1:], *open_ended_logits), dim=1)  # type:ignore

        if return_activations:
            activation_cache = torch.stack(activation_cache)
            if open_ended_activations and any(acts for acts in open_ended_activations):
                open_ended_activations = torch.stack(
                    [torch.cat(acts, dim=0) for acts in open_ended_activations],  # type:ignore
                    dim=0,
                )

                activations = torch.cat((activation_cache, open_ended_activations), dim=1)
            else:
                activations = activation_cache
            if sparse:
                activations = activations.coalesce()
        else:
            activations = None

        return generation, logits.squeeze(0), activations

    def __del__(self):
        # Prevent memory leaks; a model whose construction failed has no hook registry to reset
        if "_hook_registry" not in self.__dict__:
            return
        try:
            self.reset_hooks(including_permanent=True)
        except TypeError:
            # at interpreter shutdown torch's own module globals may already be gone
            pass
