from __future__ import annotations
from typing import Dict, List, Sequence
import torch
import torch.nn as nn
from transformers import WhisperConfig, WhisperForConditionalGeneration

CONFIG_PREFIX = "bottleneck_adapter_"


class BottleneckAdapter(nn.Module):
    # Sequential bottleneck adapter: hidden_size -> adapter_dim -> hidden_size.

    def __init__(self, hidden_size: int, adapter_dim: int, dropout: float = 0.0):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # Zero-init the up-projection so the adapter starts as an identity function.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.down_proj(hidden_states)
        output = self.activation(output)
        output = self.up_proj(output)
        return self.dropout(output)


def _adapter_residual_hook(layer: nn.Module, inputs, output: torch.Tensor) -> torch.Tensor:
    # Whisper encoder/decoder layers return the hidden states tensor directly.
    return output + layer.adapter(output)


def config_has_adapters(config: WhisperConfig) -> bool:
    return hasattr(config, CONFIG_PREFIX + "dim")


class WhisperBottleneckAdapterForConditionalGeneration(WhisperForConditionalGeneration):
    # Whisper with a trainable adapter added after each selected Transformer block.
    # The adapters are registered as submodules of the blocks they follow and applied
    # through a forward hook.

    CONFIG_PREFIX = CONFIG_PREFIX

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self._initialize_adapters_from_config()

    def _initialize_adapters_from_config(self) -> None:
        config = self.config
        self.adapter_dim = int(getattr(config, CONFIG_PREFIX + "dim"))
        self.adapter_dropout = float(getattr(config, CONFIG_PREFIX + "dropout", 0.0))
        self.adapter_encoder_layers = list(getattr(config, CONFIG_PREFIX + "encoder_layers", []))
        self.adapter_decoder_layers = list(getattr(config, CONFIG_PREFIX + "decoder_layers", []))
        for layer_index in self.adapter_encoder_layers:
            self._attach_adapter(self.model.encoder.layers[layer_index])
        for layer_index in self.adapter_decoder_layers:
            self._attach_adapter(self.model.decoder.layers[layer_index])

    def _attach_adapter(self, layer: nn.Module) -> None:
        layer.adapter = BottleneckAdapter(
            self.config.d_model, self.adapter_dim, self.adapter_dropout
        )
        layer.register_forward_hook(_adapter_residual_hook)

    def freeze_backbone(self) -> None:
        # Train the adapters only; every pretrained Whisper weight stays frozen.
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.modules():
            if isinstance(module, BottleneckAdapter):
                for parameter in module.parameters():
                    parameter.requires_grad = True


def attach_adapters(
    model: WhisperForConditionalGeneration,
    adapter_dim: int,
    dropout: float,
    encoder_layers: Sequence[int],
    decoder_layers: Sequence[int],
) -> WhisperBottleneckAdapterForConditionalGeneration:
    # Add adapters to a loaded Whisper model and record them on its config.
    # Loading the pretrained weights before injecting keeps Transformers from
    # discarding them, and the config fields let ``from_pretrained`` rebuild the
    # same structure from the saved checkpoint.

    config = model.config
    setattr(config, CONFIG_PREFIX + "dim", int(adapter_dim))
    setattr(config, CONFIG_PREFIX + "dropout", float(dropout))
    setattr(config, CONFIG_PREFIX + "encoder_layers", list(encoder_layers))
    setattr(config, CONFIG_PREFIX + "decoder_layers", list(decoder_layers))
    config.architectures = [WhisperBottleneckAdapterForConditionalGeneration.__name__]
    model.__class__ = WhisperBottleneckAdapterForConditionalGeneration
    model._initialize_adapters_from_config()
    return model


def adapter_summary(model: WhisperBottleneckAdapterForConditionalGeneration) -> Dict[str, object]:
    return {
        "adapter_dim": model.adapter_dim,
        "dropout": model.adapter_dropout,
        "encoder_layers": list(model.adapter_encoder_layers),
        "decoder_layers": list(model.adapter_decoder_layers),
    }
