# SPDX-License-Identifier: Apache-2.0
"""Normal-only optimization and normal-validation checkpoint selection."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .adm import ADM, world_loss
from .data import adm_batch, build_histories, load_stream, read_events
from .eid import EID, InteractionStream, chronological_blocks, interaction_dynamics_loss, interaction_dynamics_scores
from .encoder import LocalEncoder, local_loss


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def checkpoint(path, model, **metadata):
    torch.save(dict(state_dict=model.state_dict(), **metadata), path)


def restore(path, model):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return model


def train_local(config, paths, work):
    """Pretrain the recurrent encoder using normal events only."""
    seed_all(config["seed"])
    model = LocalEncoder(len(config["relations"]), state_dim=16)
    setup = config["training"]["local"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=setup["learning_rate"], weight_decay=1e-4)
    out = Path(work) / "local.pt"
    best = float("inf")

    def epoch(files, train=False):
        model.train(train)
        total, count = 0., 0
        for path in files:
            data = build_histories(read_events(path, config["relations"]), config["relations"])
            indices = np.random.permutation(len(data["history"])) if train else np.arange(len(data["history"]))
            for start in range(0, len(indices), setup["batch_size"]):
                take = indices[start:start + setup["batch_size"]]
                history = torch.as_tensor(data["history"][take])
                state = torch.as_tensor(data["state"][take, :16])
                with torch.set_grad_enabled(train):
                    output = model(history, state, torch.ones(len(take), dtype=torch.long))
                    loss = local_loss(output, torch.as_tensor(data["relation"][take]),
                                      torch.as_tensor(data["type"][take]), torch.as_tensor(data["gap"][take]),
                                      torch.as_tensor(data["next_state"][take]))
                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                total += float(loss.detach()) * len(take)
                count += len(take)
        return total / count

    for number in range(setup["epochs"]):
        epoch(paths["train"], True)
        value = epoch(paths["validation"])
        if value < best:
            best = value
            checkpoint(out, model, epoch=number + 1)
        print(f"Local epoch {number + 1}: normal-validation loss {value:.6f}", flush=True)
    return restore(out, model).eval().requires_grad_(False)


def normalization_tensors(stats):
    return {k: torch.tensor(stats[k], dtype=torch.float32) for k in ("mean", "scale", "active")}


def train_adm(config, cache, stats, work):
    seed_all(config["seed"])
    model = ADM(len(config["relations"]))
    normal = normalization_tensors(stats)
    out = Path(work) / "adm.pt"

    def epoch(files, batch_size, train=False, optimizer=None):
        model.train(train)
        total, count = 0., 0
        for path in files:
            data = load_stream(path)
            selected = np.flatnonzero(data["valid"])
            if train:
                np.random.shuffle(selected)
            for start in range(0, len(selected), batch_size):
                take = selected[start:start + batch_size]
                hidden, state, action, current, future, target = adm_batch(data, take, normal)
                with torch.set_grad_enabled(train):
                    loss, _ = world_loss(model(hidden, state, action), current, future, target,
                                         normal["active"], model.output_sizes, config["future_weight"])
                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                total += float(loss.detach()) * len(take)
                count += len(take)
        if not count:
            raise ValueError("No complete future windows in this split")
        return total / count

    for stage in ("adm_initialization", "adm_finetuning"):
        setup = config["training"][stage]
        optimizer = torch.optim.AdamW(model.parameters(), lr=setup["learning_rate"], weight_decay=1e-4)
        best = float("inf")
        for number in range(setup["epochs"]):
            epoch(cache["train"], setup["batch_size"], True, optimizer)
            value = epoch(cache["validation"], setup["batch_size"])
            if value < best:
                best = value
                checkpoint(out, model, stage=stage, epoch=number + 1)
            print(f"ADM {stage}, epoch {number + 1}: normal-validation loss {value:.6f}", flush=True)
        restore(out, model)
    return model.eval().requires_grad_(False)


def relation_weights(counts, relation_ids):
    values = np.zeros(len(relation_ids) + 1, np.float64)
    for relation, count in counts.items():
        values[relation_ids[relation]] = count
    result = np.zeros_like(values)
    result[1:] = np.clip(np.sqrt(values[1:].sum() / (len(relation_ids) * np.maximum(values[1:], 1))), .25, 4.)
    return torch.as_tensor(result, dtype=torch.float32)


def replay_eid(model, files, config, optimizer=None, class_weight=None):
    """One independent stream; carry memory across its ordered input files."""
    model.train(optimizer is not None)
    stream = InteractionStream(model, config["namespace"])
    setup = config["training"]["eid"]
    total, count = 0., 0
    keys, scores = [], []
    for path in files:
        events = read_events(path, config["relations"])
        for start in range(0, len(events), setup["bptt_events"]):
            segment = events[start:start + setup["bptt_events"]]
            objective = None
            if optimizer is not None:
                optimizer.zero_grad()
            with torch.set_grad_enabled(optimizer is not None):
                for block in chronological_blocks(segment, setup["batch_size"]):
                    output, targets = stream.score_then_update(block, config["relations"])
                    loss = interaction_dynamics_loss(output, *targets, class_weight=class_weight,
                        semantic_weight=config["semantic_weight"], gap_weight=config["gap_weight"])
                    objective = loss * len(block) if objective is None else objective + loss * len(block)
                    values = interaction_dynamics_scores(output, *targets,
                        semantic_weight=config["semantic_weight"], gap_weight=config["gap_weight"])
                    total += float(values.detach().sum())
                    count += len(block)
                    if optimizer is None:
                        keys.extend((str(e[6]), str(e[7]), str(e[0])) for e in block)
                        scores.extend(values.detach().tolist())
                if optimizer is not None:
                    (objective / len(segment)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    stream.detach()
                    optimizer.step()
    return total / count, np.asarray(keys), np.asarray(scores)


def train_eid(config, paths, work):
    seed_all(config["seed"])
    model = EID(len(config["relations"]), history_length=20, hidden_dim=64)
    counts = Counter()
    for path in paths["train"]:
        counts.update(e[3] for e in read_events(path, config["relations"]))
    weights = relation_weights(counts, config["relations"])
    setup = config["training"]["eid"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=setup["learning_rate"], weight_decay=1e-4)
    best, out = float("inf"), Path(work) / "eid.pt"
    for number in range(setup["epochs"]):
        replay_eid(model, paths["train"], config, optimizer, weights)
        value, _, _ = replay_eid(model, paths["validation"], config)
        if value < best:
            best = value
            checkpoint(out, model, epoch=number + 1)
        print(f"EID epoch {number + 1}: normal-validation loss {value:.6f}", flush=True)
    return restore(out, model).eval().requires_grad_(False)
