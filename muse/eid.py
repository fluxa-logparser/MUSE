# SPDX-License-Identifier: Apache-2.0
"""Entity Interaction Dynamics with persistent directed-pair memory."""
from __future__ import annotations
from collections import defaultdict, deque
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CausalEndpointEncoder(nn.Module):
    """Encode one node using only its previously observed incident edges."""

    def __init__(self, num_relations, history_length=20, hidden_dim=64,
                 num_gap_buckets=16, label_vocab_size=8192):
        super().__init__()
        self.history_length = history_length
        self.relation_embedding = nn.Embedding(
            num_relations + 1, 16, padding_idx=0
        )
        self.neighbor_type_embedding = nn.Embedding(4, 4, padding_idx=0)
        self.direction_embedding = nn.Embedding(3, 4, padding_idx=0)
        self.gap_embedding = nn.Embedding(
            num_gap_buckets, 8, padding_idx=0
        )
        self.neighbor_label_embedding = nn.Embedding(
            label_vocab_size, 16, padding_idx=0
        )
        self.token_projection = nn.Sequential(
            nn.Linear(48, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.position_embedding = nn.Embedding(history_length, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.node_type_embedding = nn.Embedding(4, 8)
        self.label_embedding = nn.Embedding(
            label_vocab_size, 16, padding_idx=0
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + 8 + 2 + 16, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, history, node_type, label_tokens=None):
        history = history.long()
        tokens = torch.cat((
            self.relation_embedding(history[..., 0]),
            self.neighbor_type_embedding(history[..., 1]),
            self.direction_embedding(history[..., 2]),
            self.gap_embedding(history[..., 3].clamp(0, 15)),
            self.neighbor_label_embedding(history[..., 4]),
        ), dim=-1)
        tokens = self.token_projection(tokens)
        positions = torch.arange(self.history_length, device=history.device)
        tokens = tokens + self.position_embedding(positions)
        valid = history[..., 0].ne(0)
        logits = self.attention(tokens).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=-1) * valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0)
        pooled = (tokens * weights.unsqueeze(-1)).sum(1)
        count = valid.sum(-1).float()
        activity = torch.stack((
            count / self.history_length,
            torch.log1p(count) / torch.log1p(
                count.new_tensor(float(self.history_length))
            ),
        ), dim=-1)
        if label_tokens is None:
            label = pooled.new_zeros((len(pooled), 16))
        else:
            label_tokens = label_tokens.long()
            label_mask = label_tokens.ne(0).float()
            label = self.label_embedding(label_tokens)
            label = (label * label_mask.unsqueeze(-1)).sum(1)
            label = label / label_mask.sum(1, keepdim=True).clamp_min(1.0)
        return self.output(torch.cat((
            pooled, self.node_type_embedding(node_type.long()), activity, label,
        ), dim=-1))


class RelationshipMemory(nn.Module):
    """Persistent hidden state for a directed entity pair."""

    def __init__(self, num_relations, history_length=20, hidden_dim=64,
                 num_gap_buckets=16, label_vocab_size=8192):
        super().__init__()
        self.history_length = history_length
        self.relation_embedding = nn.Embedding(
            num_relations + 1, 16, padding_idx=0
        )
        self.neighbor_type_embedding = nn.Embedding(4, 4, padding_idx=0)
        self.direction_embedding = nn.Embedding(3, 4, padding_idx=0)
        self.gap_embedding = nn.Embedding(
            num_gap_buckets, 8, padding_idx=0
        )
        self.label_embedding = nn.Embedding(
            label_vocab_size, 16, padding_idx=0
        )
        self.token_projection = nn.Sequential(
            nn.Linear(48, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def encode_tokens(self, observations):
        observations = observations.long()
        return self.token_projection(torch.cat((
            self.relation_embedding(observations[..., 0]),
            self.neighbor_type_embedding(observations[..., 1]),
            self.direction_embedding(observations[..., 2]),
            self.gap_embedding(observations[..., 3].clamp(0, 15)),
            self.label_embedding(observations[..., 4]),
        ), dim=-1))

    def observe(self, state, observation):
        return self.gru(self.encode_tokens(observation), state)

    def forward(self, state, seen):
        return self.output(state) * seen.to(state).unsqueeze(-1)


class EID(nn.Module):
    """Interaction likelihood conditioned on endpoint beliefs and pair memory.

    Relation likelihood conditions on both known endpoints. Object semantics
    are instead modeled from the actor belief and prior relationship memory:
    the current target's type, label and history cannot leak into that head.
    This is a conditional likelihood of an observed interaction, not a model
    that generates target identities or a full-system causal simulator.
    """


    def __init__(self, num_relations, history_length=20, hidden_dim=64):
        super().__init__()
        self.config = dict(num_relations=num_relations,
                           history_length=history_length, hidden_dim=hidden_dim)
        self.endpoint_encoder = CausalEndpointEncoder(
            num_relations=num_relations, history_length=history_length,
            hidden_dim=hidden_dim,
        )
        self.relationship_memory = RelationshipMemory(
            num_relations=num_relations, history_length=history_length,
            hidden_dim=hidden_dim,
        )
        self.relation_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim * 2, num_relations + 1),
        )
        # category(9), semantic properties(10), pair inter-arrival bucket(16).
        self.event_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 35),
        )

    def forward(self, source_history, target_history, source_type, target_type,
                source_label_tokens, target_label_tokens, raw_memory, seen):
        source = self.endpoint_encoder(source_history, source_type, source_label_tokens)
        target = self.endpoint_encoder(target_history, target_type, target_label_tokens)
        memory = self.relationship_memory(raw_memory, seen)
        return self.decode_interaction(source, target, memory)

    def decode_interaction(self, source, target, memory):
        """Predict relation, resource semantics, and pair timing."""
        interaction_belief = torch.cat((source, target,
                                       torch.abs(source - target),
                                       source * target, memory), dim=-1)
        event = self.event_decoder(torch.cat((source, memory), dim=-1))
        return {
            "source": source, "target": target,
            "relationship_memory": memory,
            "interaction_belief": interaction_belief,
            "relation_logits": self.relation_decoder(interaction_belief),
            "object_category_logits": event[..., :9],
            "object_feature_logits": event[..., 9:19],
            "pair_gap_logits": event[..., 19:],
        }


def interaction_dynamics_parts(output, relation, object_category,
                               object_features, pair_gap, class_weight=None):
    return {
        "relation": F.cross_entropy(output["relation_logits"], relation.long(),
                                    weight=class_weight, reduction="none"),
        "object_semantics": (
            F.cross_entropy(output["object_category_logits"],
                            object_category.long(), reduction="none")
            + F.binary_cross_entropy_with_logits(
                output["object_feature_logits"], object_features.float(),
                reduction="none",
            ).mean(-1)
        ),
        "pair_gap": F.cross_entropy(output["pair_gap_logits"], pair_gap.long(),
                                    reduction="none"),
    }


def interaction_dynamics_scores(output, relation, object_category,
                                object_features, pair_gap,
                                semantic_weight=0.25, gap_weight=0.1):
    parts = interaction_dynamics_parts(output, relation, object_category,
                                      object_features, pair_gap)
    return (parts["relation"] + semantic_weight * parts["object_semantics"]
            + gap_weight * parts["pair_gap"])


def interaction_dynamics_loss(output, relation, object_category,
                              object_features, pair_gap, class_weight=None,
                              semantic_weight=0.25, gap_weight=0.1):
    parts = interaction_dynamics_parts(output, relation, object_category,
                                      object_features, pair_gap, class_weight)
    return (parts["relation"] + semantic_weight * parts["object_semantics"]
            + gap_weight * parts["pair_gap"]).mean()


def model_digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


class InteractionStream:
    """One ordered stream in a stable host/trace namespace.

    reset_entity/reset_pair handle explicit identity lifecycle changes. reset
    starts an independent stream/epoch/split. None is called on a date change.
    State detachment truncates gradients without clearing hidden values.
    """
    def __init__(self, model, namespace="default"):
        self.model = model
        self.namespace = str(namespace)
        self.history_length = model.config["history_length"]
        self.reset()

    def reset(self):
        self.states, self.pair_times, self.counts = {}, {}, {}
        self.pending = set()
        self.histories = defaultdict(lambda: deque(maxlen=self.history_length))
        self.entity_times = {}
        self.last_time = None

    def reset_pair(self, actor, target):
        key = (str(actor), str(target))
        for mapping in (self.states, self.pair_times, self.counts): mapping.pop(key, None)
        self.pending.discard(key)

    def reset_entity(self, entity):
        entity = str(entity)
        for pair in list(self.states):
            if entity in pair: self.reset_pair(*pair)
        self.histories.pop(entity, None)
        self.entity_times.pop(entity, None)

    def detach(self):
        for key in self.pending:
            self.states[key] = self.states[key].detach().clone()
        self.pending.clear()

    def score_then_update(self, events, relation_ids):
        """Batch only pair-independent events; never reorder duplicate pairs.

        The caller splits a chronological segment at each repeated directed
        pair. Endpoint histories are still snapshotted event by event.
        """
        # Import the established tokenization without duplicating its semantics.
        from .data import copy_history, label_tokens, label_id, gap_bucket, TYPE_IDS
        from .semantics import object_semantics
        if not events: raise ValueError("Empty stream block")
        pairs = [(str(e[1]), str(e[2])) for e in events]
        if len(set(pairs)) != len(pairs):
            raise ValueError("A block must contain unique directed pairs; split chronologically")
        times = [int(e[0]) for e in events]
        if any(a>b for a,b in zip(times,times[1:])) or (self.last_time is not None and times[0]<self.last_time):
            raise ValueError("Interaction stream must be chronological; reset explicitly for a new split")
        sh, th, types, sl, tl, rels, cats, features, gaps, tokens = ([] for _ in range(10))
        for event, pair in zip(events, pairs):
            time_ns, actor, target, relation, object_type, object_label = event[:6]
            actor, target = str(actor), str(target)
            actor_label = event[8] if len(event)>8 else ""
            rid = relation_ids.get(relation,0)
            typ = TYPE_IDS.get(object_type,0)
            feat, cat = object_semantics(object_label,object_type)
            pair_gap = gap_bucket(self.pair_times.get(pair),time_ns)
            sh.append(copy_history(self.histories[actor],self.history_length))
            th.append(copy_history(self.histories[target],self.history_length))
            types.append(typ); sl.append(label_tokens(actor_label)); tl.append(label_tokens(object_label))
            rels.append(rid); cats.append(cat); features.append(feat); gaps.append(pair_gap)
            tokens.append((rid,typ,1,pair_gap,label_id(object_label)))
            self.histories[actor].append((rid,typ,1,gap_bucket(self.entity_times.get(actor),time_ns),label_id(object_label)))
            if target!=actor:
                self.histories[target].append((rid,1,2,gap_bucket(self.entity_times.get(target),time_ns),label_id(actor_label)))
            self.entity_times[actor]=time_ns; self.entity_times[target]=time_ns
        device = next(self.model.parameters()).device
        tensor = lambda x: torch.as_tensor(np.asarray(x),device=device)
        zero = next(self.model.parameters()).new_zeros(self.model.config["hidden_dim"])
        seen = tensor([p in self.states for p in pairs])
        prior = torch.stack([self.states.get(p,zero) for p in pairs])
        output = self.model(tensor(sh),tensor(th),torch.ones(len(events),device=device,dtype=torch.long),
            tensor(types),tensor(sl),tensor(tl),prior,seen)
        # Only the update sees the current relation, pair gap and resource token.
        updated = self.model.relationship_memory.observe(prior,tensor(tokens))
        for index,pair in enumerate(pairs):
            self.states[pair]=updated[index]
            self.pair_times[pair]=times[index]
            self.counts[pair]=self.counts.get(pair,0)+1
            if updated.requires_grad: self.pending.add(pair)
        self.last_time=times[-1]
        return output,(tensor(rels),tensor(cats),tensor(features),tensor(gaps))

    def snapshot(self):
        if self.model.training: raise ValueError("Runtime snapshots require a fixed eval model")
        return dict(model_config=dict(self.model.config),
            model_sha256=model_digest(self.model),namespace=self.namespace,
            states={k:v.detach().cpu().clone() for k,v in self.states.items()},pair_times=dict(self.pair_times),counts=dict(self.counts),
            histories={k:list(v) for k,v in self.histories.items()},entity_times=dict(self.entity_times),last_time=self.last_time)

    def restore(self, state):
        if self.model.training: raise ValueError("Restore runtime only with a fixed eval model")
        if (state.get("model_config")!=self.model.config
                or state["model_sha256"]!=model_digest(self.model)):
            raise ValueError("Runtime state belongs to different model weights or configuration")
        if state["namespace"]!=self.namespace: raise ValueError("Runtime namespace mismatch")
        self.reset()
        device=next(self.model.parameters()).device
        self.states={tuple(k):v.clone().to(device) for k,v in state["states"].items()}
        self.pair_times=dict(state["pair_times"]);self.counts=dict(state["counts"])
        self.histories.update({k:deque(v,maxlen=self.history_length) for k,v in state["histories"].items()})
        self.entity_times=dict(state["entity_times"]);self.last_time=state["last_time"]


def chronological_blocks(events, max_size=128):
    """Preserve exact event order and split before any repeated directed pair."""
    if max_size<1: raise ValueError("max_size must be positive")
    block, pairs = [],set()
    for event in events:
        pair=(str(event[1]),str(event[2]))
        if block and (pair in pairs or len(block)>=max_size):
            yield block
            block,pairs=[],set()
        block.append(event);pairs.add(pair)
    if block:yield block
