# SPDX-License-Identifier: Apache-2.0
"""Pretrained recurrent encoder for event histories and future targets."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class LocalEncoder(nn.Module):
    """Predict the next provenance event and source-local behavioral state."""

    def __init__(
        self,
        num_relations: int,
        state_dim: int,
        history_length: int = 8,
        hidden_dim: int = 96,
        num_gap_buckets: int = 16,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.history_length = history_length
        self.num_gap_buckets = num_gap_buckets
        self.relation_embedding = nn.Embedding(num_relations + 1, 16)
        self.source_type_embedding = nn.Embedding(4, 4)
        self.target_type_embedding = nn.Embedding(4, 4)
        self.gap_embedding = nn.Embedding(num_gap_buckets, 8)
        self.history_encoder = nn.GRU(32, hidden_dim, batch_first=True)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 32), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.relation_head = nn.Linear(hidden_dim, num_relations + 1)
        self.target_type_head = nn.Linear(hidden_dim, 4)
        self.gap_head = nn.Linear(hidden_dim, num_gap_buckets)
        self.next_state_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, history, state, source_type):
        history = history.long()
        relation = self.relation_embedding(history[..., 0])
        source = self.source_type_embedding(history[..., 1])
        target = self.target_type_embedding(history[..., 2])
        gap = self.gap_embedding(history[..., 3].clamp(0, self.num_gap_buckets - 1))
        _, hidden = self.history_encoder(torch.cat((relation, source, target, gap), dim=-1))
        fused = torch.cat(
            (
                hidden[-1],
                self.state_encoder(state.float()),
                self.source_type_embedding(source_type.long()),
            ),
            dim=-1,
        )
        decoded = self.decoder(fused)
        return {
            "relation_logits": self.relation_head(decoded),
            "target_type_logits": self.target_type_head(decoded),
            "gap_logits": self.gap_head(decoded),
            "next_state": self.next_state_head(decoded),
        }


def local_loss(output, relation, target_type, gap, next_state):
    """Joint objective for event prediction and the post-event state forecast."""
    return (
        0.45 * F.cross_entropy(output["relation_logits"], relation)
        + 0.20 * F.cross_entropy(output["target_type_logits"], target_type)
        + 0.10 * F.cross_entropy(output["gap_logits"], gap)
        + 0.25 * F.smooth_l1_loss(output["next_state"], next_state)
    )
