"""Model registry for baseline and alternate NanoQEC decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from nanoqec.contracts import DatasetManifest


@dataclass(slots=True)
class LayoutSpec:
    """Model-facing view of the stored detector representation."""

    detector_count: int
    time_bucket_indices: list[list[int]]
    max_time_bucket_size: int

    @classmethod
    def from_manifest(cls, manifest: DatasetManifest) -> LayoutSpec:
        representation = manifest.representation
        return cls(
            detector_count=int(manifest.detector_count),
            time_bucket_indices=[
                [int(index) for index in bucket]
                for bucket in representation["time_bucket_indices"]
            ],
            max_time_bucket_size=int(representation["max_time_bucket_size"]),
        )

    @property
    def time_steps(self) -> int:
        return len(self.time_bucket_indices)


def default_model_spec(model_name: str, layout: LayoutSpec) -> dict[str, Any]:
    """Return default model hyperparameters for a named model."""

    if model_name == "minimal_aq2":
        return {
            "d_model": 32,
            "nhead": 4,
            "dropout": 0.0,
            "feedforward_mult": 2,
            "time_steps": layout.time_steps,
            "max_time_bucket_size": layout.max_time_bucket_size,
        }
    if model_name == "trivial_linear":
        return {"detector_count": layout.detector_count}
    raise ValueError(f"unknown model: {model_name}")


def build_model(
    model_name: str,
    layout: LayoutSpec,
    model_spec: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a model from the registry."""

    effective_model_spec = model_spec or default_model_spec(model_name, layout)
    if model_name == "minimal_aq2":
        return MinimalAQ2Decoder(layout=layout, **effective_model_spec)
    if model_name == "trivial_linear":
        return TrivialLinearDecoder(detector_count=layout.detector_count)
    raise ValueError(f"unknown model: {model_name}")


class MinimalAQ2Decoder(nn.Module):
    """A small AQ2-style baseline using spatial attention and temporal recurrence."""

    def __init__(
        self,
        layout: LayoutSpec,
        d_model: int,
        nhead: int,
        dropout: float,
        feedforward_mult: int,
        time_steps: int,
        max_time_bucket_size: int,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.scalar_projection = nn.Linear(1, d_model)
        self.slot_embedding = nn.Embedding(max_time_bucket_size, d_model)
        self.time_embedding = nn.Embedding(time_steps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * feedforward_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.spatial_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.temporal_recurrence = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, detector_events: Tensor) -> Tensor:
        padded_events, valid_mask = self._to_padded_time_buckets(detector_events)
        token_states = self.scalar_projection(padded_events.unsqueeze(-1))
        slot_ids = torch.arange(self.layout.max_time_bucket_size, device=detector_events.device)
        time_ids = torch.arange(self.layout.time_steps, device=detector_events.device)
        token_states = token_states + self.slot_embedding(slot_ids)[None, None, :, :]
        token_states = token_states + self.time_embedding(time_ids)[None, :, None, :]

        summaries: list[Tensor] = []
        for time_index in range(self.layout.time_steps):
            time_slice = token_states[:, time_index, :, :]
            key_padding_mask = ~valid_mask[time_index].unsqueeze(0).expand(
                detector_events.shape[0], -1
            )
            encoded = self.spatial_encoder(time_slice, src_key_padding_mask=key_padding_mask)
            valid = valid_mask[time_index].to(encoded.dtype).view(1, -1, 1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            summaries.append(pooled)

        temporal_input = torch.stack(summaries, dim=1)
        _, hidden = self.temporal_recurrence(temporal_input)
        logits = self.head(hidden[-1]).squeeze(-1)
        return logits

    def _to_padded_time_buckets(self, detector_events: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = detector_events.shape[0]
        padded = detector_events.new_zeros(
            (batch_size, self.layout.time_steps, self.layout.max_time_bucket_size)
        )
        valid_mask = torch.zeros(
            (self.layout.time_steps, self.layout.max_time_bucket_size),
            dtype=torch.bool,
            device=detector_events.device,
        )
        for time_index, bucket in enumerate(self.layout.time_bucket_indices):
            bucket_tensor = torch.tensor(bucket, dtype=torch.long, device=detector_events.device)
            bucket_values = detector_events.index_select(dim=1, index=bucket_tensor)
            bucket_size = len(bucket)
            padded[:, time_index, :bucket_size] = bucket_values
            valid_mask[time_index, :bucket_size] = True
        return padded, valid_mask


class TrivialLinearDecoder(nn.Module):
    """A small alternate model used to prove architecture-pluggable loading."""

    def __init__(self, detector_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(detector_count, 1)

    def forward(self, detector_events: Tensor) -> Tensor:
        return self.linear(detector_events).squeeze(-1)
