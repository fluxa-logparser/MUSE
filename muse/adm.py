# SPDX-License-Identifier: Apache-2.0
"""Agent Dynamics Modeling: belief, policy, and future-state prediction."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HORIZON = 4


LATENT_DIM = 96 + 10 + 9


def actor_windows(actors, times, horizon=HORIZON):
    """Same-day same-actor windows; discard incomplete/timestamp-tied futures."""
    actors, times = np.asarray(actors), np.asarray(times, np.int64)
    if len(actors) != len(times) or np.any(times[1:] < times[:-1]):
        raise ValueError("Expected an ordered audit stream")
    order = np.argsort(actors, kind="stable")
    future = np.full((len(actors), horizon), -1, np.int64)
    # A timestamp tie is not an independently later observation. Start after
    # the actor's entire equal-timestamp block, then take four later events.
    sorted_actors, sorted_times = actors[order], times[order]
    ends = np.r_[np.flatnonzero((sorted_actors[1:] != sorted_actors[:-1]) |
                               (sorted_times[1:] != sorted_times[:-1])) + 1, len(order)]
    starts = np.r_[0, ends[:-1]]
    next_position = np.repeat(ends, ends-starts)
    for offset in range(horizon):
        position = next_position + offset
        inside = position < len(order)
        origin = np.flatnonzero(inside)
        destination = position[inside]
        same = sorted_actors[origin] == sorted_actors[destination]
        future[order[origin[same]], offset] = order[destination[same]]
    valid = (future >= 0).all(1)
    valid &= times[np.maximum(future[:, 0], 0)] > times
    future[~valid] = -1
    return future, valid


def historical_neighborhood(neighborhood, actors):
    """Actor-only context, independent of the current chosen object's history.

    Retain the actor's prior interaction summary (neutralize its current-pair
    frequency); use the prior actor event's object context for the second half.
    """
    result = np.asarray(neighborhood, np.float32).copy()
    result[:, 9] = 0
    result[:, 12:] = 0
    order = np.argsort(np.asarray(actors), kind="stable")
    left, right = order[:-1], order[1:]
    same = np.asarray(actors)[left] == np.asarray(actors)[right]
    result[right[same], 12:] = np.asarray(neighborhood)[left[same], 12:]
    return result


@torch.no_grad()
def encode_tokens(encoder, tokens):
    values = torch.cat((encoder.relation_embedding(tokens[..., 0].long()),
                        encoder.source_type_embedding(tokens[..., 1].long()),
                        encoder.target_type_embedding(tokens[..., 2].long()),
                        encoder.gap_embedding(tokens[..., 3].long())), -1)
    return encoder.history_encoder(values)[1][-1]


def action_vector(encoder, relation, target_type, gap, category, features):
    """Encode the observed operation, object attributes, and timing."""
    source = torch.ones_like(relation)
    return torch.cat((encoder.relation_embedding(relation.long()),
                      encoder.source_type_embedding(source.long()),
                      encoder.target_type_embedding(target_type.long()),
                      encoder.gap_embedding(gap.long()),
                      F.one_hot(category.long(), 9).float(), features.float()), -1)


class ADM(nn.Module):
    """A five-view belief and action-conditioned Gaussian future-state model."""

    def __init__(self, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.output_sizes = (num_relations + 1, 4, 16, 9, 10)
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(dim, 48), nn.GELU(), nn.LayerNorm(48))
                                      for dim in (112, 24, 24, 24, 24)])
        self.gate = nn.Linear(240, 5)
        self.belief = nn.Sequential(nn.Linear(240, 64), nn.GELU(), nn.Linear(64, 64), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, sum(self.output_sizes)))
        self.dynamics = nn.Sequential(nn.Linear(64 + 51, 96), nn.GELU(),
                                      nn.Linear(96, 96), nn.GELU())
        self.distribution = nn.Linear(96, 2 * LATENT_DIM)
        self.observation_decoder = nn.Sequential(nn.Linear(LATENT_DIM, 96), nn.GELU(),
                                                nn.Linear(96, HORIZON * sum(self.output_sizes)))
        self.register_buffer("active_views", torch.ones(5, dtype=torch.bool))

    def forward(self, history_hidden, state, action):
        views = (torch.cat((history_hidden, state[:, :16]), -1), *state[:, 16:].split(24, -1))
        encoded = [layer(value) for layer, value in zip(self.encoders, views)]
        joined = torch.cat(encoded, -1)
        gate = self.gate(joined).masked_fill(~self.active_views[None], -1e9).softmax(-1)
        belief = self.belief(torch.cat([x * gate[:, k:k+1] for k, x in enumerate(encoded)], -1))
        mean, logvar = self.distribution(self.dynamics(torch.cat((belief, action), -1))).chunk(2, -1)
        logvar = logvar.clamp(-5, 5)
        return {"belief": belief, "mean": mean, "logvar": logvar,
                "policy": self.policy(belief),
                "future_observations": self.observation_decoder(mean).reshape(-1, HORIZON, sum(self.output_sizes))}


def observation_loss(logits, labels, sizes):
    relation, target_type, gap, category, features = labels
    parts = logits.split(sizes, -1)
    shape = logits.shape[:-1]
    losses = [F.cross_entropy(p.reshape(-1, p.shape[-1]), y.long().reshape(-1), reduction="none").reshape(shape)
              for p, y in zip(parts[:4], (relation, target_type, gap, category))]
    return (.45 * losses[0] + .15 * losses[1] + .10 * losses[2] + .15 * losses[3]
            + .15 * F.binary_cross_entropy_with_logits(parts[4], features.float(), reduction="none").mean(-1))


def future_scores(output, target, active_mask):
    mask = active_mask.to(target)
    squared = (output["mean"] - target).square()
    mse = (squared * mask).sum(-1) / mask.sum()
    nll = .5 * (np.log(2 * np.pi) + output["logvar"] + squared * (-output["logvar"]).exp())
    return mse, (nll * mask).sum(-1) / mask.sum()


def world_loss(output, current, future, target, active_mask, sizes, future_weight=0.2):
    behavior = observation_loss(output["policy"], current, sizes).mean()
    mse, nll = future_scores(output, target, active_mask)
    observation = observation_loss(output["future_observations"], future, sizes).mean()
    loss = behavior + nll.mean() + future_weight * observation
    return loss, {"behavior": behavior, "future_mse": mse.mean(), "future_nll": nll.mean(), "observation": observation}
