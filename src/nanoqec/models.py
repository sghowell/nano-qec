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
    time_steps: int
    max_time_bucket_size: int
    gather_index_grid: list[list[int]]
    detector_index_grid: list[list[int]]
    coord_grid: list[list[list[float]]]
    valid_mask_grid: list[list[bool]]
    adjacency_grid: list[list[list[float]]]

    @classmethod
    def from_manifest(cls, manifest: DatasetManifest) -> LayoutSpec:
        representation = manifest.representation
        max_time_bucket_size = int(representation["max_time_bucket_size"])
        time_bucket_indices = [
            [int(index) for index in bucket]
            for bucket in representation["time_bucket_indices"]
        ]
        normalized_xy = [
            [float(value) for value in coordinates]
            for coordinates in representation["normalized_xy"]
        ]
        gather_index_grid: list[list[int]] = []
        detector_index_grid: list[list[int]] = []
        coord_grid: list[list[list[float]]] = []
        valid_mask_grid: list[list[bool]] = []
        adjacency_grid: list[list[list[float]]] = []
        graph_k = 3
        for bucket in time_bucket_indices:
            gather_row = [0] * max_time_bucket_size
            detector_row = [0] * max_time_bucket_size
            coord_row = [[0.0, 0.0] for _ in range(max_time_bucket_size)]
            valid_row = [False] * max_time_bucket_size
            for slot, detector_index in enumerate(bucket):
                gather_row[slot] = detector_index + 1
                detector_row[slot] = detector_index + 1
                coord_row[slot] = list(normalized_xy[detector_index])
                valid_row[slot] = True
            adjacency_row = [[0.0] * max_time_bucket_size for _ in range(max_time_bucket_size)]
            valid_slots = [slot for slot, is_valid in enumerate(valid_row) if is_valid]
            for slot in valid_slots:
                neighbor_pairs: list[tuple[float, int]] = []
                for other_slot in valid_slots:
                    if other_slot == slot:
                        continue
                    dx = coord_row[slot][0] - coord_row[other_slot][0]
                    dy = coord_row[slot][1] - coord_row[other_slot][1]
                    neighbor_pairs.append((dx * dx + dy * dy, other_slot))
                adjacency_row[slot][slot] = 1.0
                for _, other_slot in sorted(neighbor_pairs)[:graph_k]:
                    adjacency_row[slot][other_slot] = 1.0
                degree = sum(adjacency_row[slot][other] for other in valid_slots)
                if degree > 0.0:
                    adjacency_row[slot] = [weight / degree for weight in adjacency_row[slot]]
            gather_index_grid.append(gather_row)
            detector_index_grid.append(detector_row)
            coord_grid.append(coord_row)
            valid_mask_grid.append(valid_row)
            adjacency_grid.append(adjacency_row)
        return cls(
            detector_count=int(manifest.detector_count),
            time_steps=len(time_bucket_indices),
            max_time_bucket_size=max_time_bucket_size,
            gather_index_grid=gather_index_grid,
            detector_index_grid=detector_index_grid,
            coord_grid=coord_grid,
            valid_mask_grid=valid_mask_grid,
            adjacency_grid=adjacency_grid,
        )


def default_model_spec(model_name: str, layout: LayoutSpec) -> dict[str, Any]:
    """Return default model hyperparameters for a named model."""

    if model_name == "minimal_aq2":
        return {
            "d_model": 64,
            "n_blocks": 2,
            "n_transformer_per_block": 2,
            "nhead": 4,
            "dropout": 0.0,
            "feedforward_mult": 4,
            "group_size": 2,
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


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class DetectorGatedRecurrence(nn.Module):
    """A lightweight gated recurrence cell shared across detector positions."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        self.candidate = nn.Linear(2 * d_model, d_model)

    def forward(self, hidden: Tensor, inputs: Tensor) -> Tensor:
        joined = torch.cat([hidden, inputs], dim=-1)
        gate = torch.sigmoid(self.gate(joined))
        candidate = torch.tanh(self.candidate(joined))
        return gate * hidden + (1.0 - gate) * candidate


class SpatialTransformer(nn.Module):
    """A small stack of spatial self-attention layers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        feedforward_mult: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * feedforward_mult,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs: Tensor, key_padding_mask: Tensor) -> Tensor:
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=key_padding_mask)
        return hidden


class LocalMessagePassing(nn.Module):
    """A lightweight learned local message-passing layer over one time bucket."""

    def __init__(self, d_model: int, init_scale: float = 0.05) -> None:
        super().__init__()
        self.message_projection = nn.Linear(d_model, d_model)
        self.message_gate = nn.Linear(2 * d_model, d_model)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        self.second_pass_logit = nn.Parameter(torch.tensor(-6.0, dtype=torch.float32))

    def forward(self, hidden: Tensor, adjacency: Tensor, valid_mask: Tensor) -> Tensor:
        neighbor_hidden = torch.einsum("ij,bjd->bid", adjacency, hidden)
        gate_inputs = torch.cat([hidden, neighbor_hidden], dim=-1)
        gate = torch.sigmoid(self.message_gate(gate_inputs))
        message = self.message_projection(neighbor_hidden)
        scale = torch.exp(self.log_scale)
        updated = hidden + scale * gate * message
        return updated * valid_mask[None, :, None].to(updated.dtype)


class AQ2Block(nn.Module):
    """One AQ2-style temporal-plus-spatial processing block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        feedforward_mult: int,
        n_transformer_per_block: int,
        use_message_passing: bool = False,
    ) -> None:
        super().__init__()
        self.recurrence = DetectorGatedRecurrence(d_model)
        self.message_passing = LocalMessagePassing(d_model) if use_message_passing else None
        self.spatial = SpatialTransformer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            feedforward_mult=feedforward_mult,
            n_layers=n_transformer_per_block,
        )

    def forward(self, inputs: Tensor, valid_mask: Tensor, adjacency: Tensor) -> Tensor:
        batch_size, time_steps, token_count, hidden_dim = inputs.shape
        hidden = inputs.new_zeros((batch_size, token_count, hidden_dim))
        outputs: list[Tensor] = []
        for time_index in range(time_steps):
            key_padding_mask = ~valid_mask[time_index].unsqueeze(0).expand(batch_size, -1)
            hidden = self.recurrence(hidden, inputs[:, time_index, :, :])
            if self.message_passing is not None:
                hidden = self.message_passing(
                    hidden,
                    adjacency=adjacency[time_index],
                    valid_mask=valid_mask[time_index],
                )
                second_hidden = self.message_passing(
                    hidden,
                    adjacency=adjacency[time_index],
                    valid_mask=valid_mask[time_index],
                )
                second_pass_weight = torch.sigmoid(self.message_passing.second_pass_logit)
                hidden = hidden + second_pass_weight * (second_hidden - hidden)
            hidden = self.spatial(hidden, key_padding_mask=key_padding_mask)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class TemporalCompression(nn.Module):
    """Compress consecutive time summaries before the final recurrent readout."""

    def __init__(self, d_model: int, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.projection = nn.Linear(d_model * group_size, d_model)

    def forward(self, sequence: Tensor) -> Tensor:
        if self.group_size <= 1:
            return sequence
        batch_size, time_steps, hidden_dim = sequence.shape
        pad_steps = (-time_steps) % self.group_size
        if pad_steps:
            padding = sequence.new_zeros((batch_size, pad_steps, hidden_dim))
            sequence = torch.cat([sequence, padding], dim=1)
        grouped_steps = sequence.shape[1] // self.group_size
        reshaped = sequence.reshape(batch_size, grouped_steps, hidden_dim * self.group_size)
        return self.projection(reshaped)


class SequenceGatedRecurrence(nn.Module):
    """A gated recurrence over pooled time summaries."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        self.candidate = nn.Linear(2 * d_model, d_model)

    def forward(self, sequence: Tensor) -> Tensor:
        hidden = sequence.new_zeros((sequence.shape[0], sequence.shape[-1]))
        for time_index in range(sequence.shape[1]):
            inputs = sequence[:, time_index, :]
            joined = torch.cat([hidden, inputs], dim=-1)
            gate = torch.sigmoid(self.gate(joined))
            candidate = torch.tanh(self.candidate(joined))
            hidden = gate * hidden + (1.0 - gate) * candidate
        return hidden


class MinimalAQ2Decoder(nn.Module):
    """A stronger AQ2-style baseline using detector-aware spatial-temporal modeling."""

    def __init__(
        self,
        layout: LayoutSpec,
        d_model: int,
        n_blocks: int,
        n_transformer_per_block: int,
        nhead: int,
        dropout: float,
        feedforward_mult: int,
        group_size: int,
        time_steps: int,
        max_time_bucket_size: int,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.event_projection = nn.Linear(1, d_model)
        self.event_state_embedding = nn.Embedding(2, d_model)
        self.detector_embedding = nn.Embedding(layout.detector_count + 1, d_model)
        self.coordinate_projection = nn.Linear(2, d_model)
        self.time_embedding = nn.Embedding(time_steps, d_model)
        self.physical_error_projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.flat_syndrome_projection = nn.Sequential(
            nn.LayerNorm(layout.detector_count),
            nn.Linear(layout.detector_count, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                AQ2Block(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    feedforward_mult=feedforward_mult,
                    n_transformer_per_block=n_transformer_per_block,
                    use_message_passing=(block_index == n_blocks - 1),
                )
                for block_index in range(n_blocks)
            ]
        )
        self.temporal_compression = TemporalCompression(d_model=d_model, group_size=group_size)
        self.summary_recurrence = SequenceGatedRecurrence(d_model=d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.register_buffer(
            "gather_index_grid",
            torch.tensor(layout.gather_index_grid, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "detector_index_grid",
            torch.tensor(layout.detector_index_grid, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "coord_grid",
            torch.tensor(layout.coord_grid, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "valid_mask_grid",
            torch.tensor(layout.valid_mask_grid, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "adjacency_grid",
            torch.tensor(layout.adjacency_grid, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "time_index_grid",
            torch.arange(time_steps, dtype=torch.long),
            persistent=False,
        )

    def forward(self, detector_events: Tensor, p_error: Tensor | None = None) -> Tensor:
        bucketed_events = self._to_bucket_grid(detector_events)
        event_features = 2.0 * bucketed_events.unsqueeze(-1) - 1.0
        hidden = self.event_projection(event_features)
        hidden = hidden + self.event_state_embedding(bucketed_events.to(torch.long))
        hidden = hidden + self.detector_embedding(self.detector_index_grid)[None, :, :, :]
        hidden = hidden + self.coordinate_projection(self.coord_grid)[None, :, :, :]
        physical_error_embedding = self._physical_error_embedding(detector_events, p_error)
        hidden = hidden + self.time_embedding(self.time_index_grid)[None, :, None, :]
        hidden = hidden + physical_error_embedding[:, None, None, :]
        hidden = self.input_dropout(hidden)
        valid_mask = self.valid_mask_grid
        hidden = hidden * valid_mask[None, :, :, None].to(hidden.dtype)

        adjacency = self.adjacency_grid
        for block in self.blocks:
            hidden = block(hidden, valid_mask=valid_mask, adjacency=adjacency)

        pooled = self._masked_mean(hidden, valid_mask)
        compressed = self.temporal_compression(pooled)
        summary = self.summary_recurrence(compressed)
        summary = (
            summary
            + self.flat_syndrome_projection(detector_events)
            + physical_error_embedding
        )
        return self.head(summary).squeeze(-1)

    def _physical_error_embedding(self, detector_events: Tensor, p_error: Tensor | None) -> Tensor:
        if p_error is None:
            p_error = detector_events.new_zeros((detector_events.shape[0],))
        p_error = p_error.to(dtype=detector_events.dtype, device=detector_events.device)
        raw_and_log = torch.stack(
            [
                p_error,
                torch.log10(p_error.clamp_min(1e-6)),
            ],
            dim=-1,
        )
        return self.physical_error_projection(raw_and_log)

    def _to_bucket_grid(self, detector_events: Tensor) -> Tensor:
        batch_size = detector_events.shape[0]
        padded = torch.cat(
            [detector_events.new_zeros((batch_size, 1)), detector_events],
            dim=1,
        )
        gathered = padded.index_select(1, self.gather_index_grid.reshape(-1))
        bucketed = gathered.reshape(
            batch_size,
            self.layout.time_steps,
            self.layout.max_time_bucket_size,
        )
        return bucketed * self.valid_mask_grid[None, :, :].to(bucketed.dtype)

    @staticmethod
    def _masked_mean(hidden: Tensor, valid_mask: Tensor) -> Tensor:
        weights = valid_mask[None, :, :, None].to(hidden.dtype)
        return (hidden * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


class TrivialLinearDecoder(nn.Module):
    """A small alternate model used to prove architecture-pluggable loading."""

    def __init__(self, detector_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(detector_count, 1)

    def forward(self, detector_events: Tensor, p_error: Tensor | None = None) -> Tensor:
        del p_error
        return self.linear(detector_events).squeeze(-1)
