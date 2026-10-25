# SPDX-License-Identifier: Apache-2.0
"""MUSE input preparation, fitting, and calibrated node scoring."""
import csv
from pathlib import Path

import numpy as np
import torch

from .adm import ADM, future_scores, observation_loss
from .data import (adm_batch, file_digest, fit_normalization, load_stream, prepare_stream,
                   read_events, read_json, save_stream, write_json)
from .eid import EID
from .encoder import LocalEncoder
from .scoring import aggregate_nodes, fuse
from .training import normalization_tensors, replay_eid, restore, train_adm, train_eid, train_local


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    relations = config["relations"]
    if not relations or sorted(relations.values()) != list(range(1, len(relations) + 1)):
        raise ValueError("Relation IDs must be contiguous integers starting at 1")
    paths = {split: [(path.parent / name).resolve() for name in config["splits"][split]]
             for split in ("train", "validation", "test")}
    if any(not items for items in paths.values()):
        raise ValueError("Specify nonempty train, validation, and test file lists")
    if len({p for items in paths.values() for p in items}) != sum(map(len, paths.values())):
        raise ValueError("Input files must be distinct across and within splits")
    for setup in config["training"].values():
        if setup["epochs"] < 1 or setup["batch_size"] < 1 or setup["learning_rate"] <= 0:
            raise ValueError("Training settings must be positive")
    if config["training"]["eid"]["bptt_events"] < 1:
        raise ValueError("bptt_events must be positive")
    return config, paths


def cache_paths(paths, work):
    return {split: [Path(work) / "cache" / split / str(i) for i in range(len(items))]
             for split, items in paths.items()}


def input_signature(config, paths, work):
    return dict(relations=config["relations"], encoder=file_digest(Path(work) / "local.pt"),
                files={split: [file_digest(p) for p in items] for split, items in paths.items()})


def prepare(config, paths, work):
    work = Path(work)
    cache = cache_paths(paths, work)
    marker = work / "inputs.json"
    signature = input_signature(config, paths, work)
    if marker.exists():
        if read_json(marker) != signature:
            raise ValueError("Inputs or encoder changed; use a new work directory")
        if (work / "normalization.json").exists():
            return cache, read_json(work / "normalization.json")
    encoder = restore(work / "local.pt", LocalEncoder(len(config["relations"]), 16)).eval().requires_grad_(False)
    for split, files in paths.items():
        for source, folder in zip(files, cache[split]):
            events = read_events(source, config["relations"])
            save_stream(folder, prepare_stream(events, config["relations"], encoder))
    stats = fit_normalization(load_stream(p) for p in cache["train"])
    write_json(work / "normalization.json", stats)
    write_json(marker, signature)
    return cache, stats


def fit(config, paths, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if any((work / name).exists() for name in ("local.pt", "adm.pt", "eid.pt", "inputs.json")):
        raise ValueError("Training requires a fresh work directory")
    write_json(work / "configuration.json", config)
    train_local(config, paths, work)
    cache, stats = prepare(config, paths, work)
    train_adm(config, cache, stats, work)
    train_eid(config, paths, work)


@torch.no_grad()
def score_adm(config, cache, stats, work):
    model = restore(Path(work) / "adm.pt", ADM(len(config["relations"]))).eval()
    normal = normalization_tensors(stats)
    values = {}
    behavior_scale = transition_scale = None
    for split in ("validation", "test"):
        keys, behaviors, transitions, valid = [], [], [], []
        for folder in cache[split]:
            data = load_stream(folder)
            keys.append(data["keys"])
            valid.append(data["valid"])
            for start in range(0, len(data["valid"]), 2048):
                take = slice(start, start + 2048)
                hidden, state, action, current, _, target = adm_batch(data, take, normal)
                output = model(hidden, state, action)
                behaviors.append(observation_loss(output["policy"], current, model.output_sizes).numpy())
                transitions.append(future_scores(output, target, normal["active"])[0].numpy())
        keys, behavior = np.concatenate(keys), np.concatenate(behaviors)
        transition, complete = np.concatenate(transitions), np.concatenate(valid)
        if split == "validation":
            if not complete.any():
                raise ValueError("Validation data need at least one complete future window")
            behavior_scale = max(float(np.quantile(behavior, .95)), 1e-8)
            transition_scale = max(float(np.quantile(transition[complete], .95)), 1e-8)
        score = behavior / behavior_scale + config["transition_weight"] * np.minimum(
            np.where(complete, transition / transition_scale, 0.), config["transition_cap"])
        nodes, scores = aggregate_nodes(keys, score)
        prefix = "validation_" if split == "validation" else ""
        values.update({prefix + "nodes": nodes, prefix + "scores": scores})
    return values


@torch.no_grad()
def score_eid(config, paths, work):
    model = restore(Path(work) / "eid.pt", EID(len(config["relations"]))).eval()
    values = {}
    for split in ("validation", "test"):
        _, keys, scores = replay_eid(model, paths[split], config)
        nodes, scores = aggregate_nodes(keys, scores)
        prefix = "validation_" if split == "validation" else ""
        values.update({prefix + "nodes": nodes, prefix + "scores": scores})
    return values


def score(config, paths, work):
    work = Path(work)
    if read_json(work / "configuration.json") != config:
        raise ValueError("Use the configuration saved with these trained models")
    cache, stats = prepare(config, paths, work)
    adm, eid = score_adm(config, cache, stats, work), score_eid(config, paths, work)
    result = fuse(adm, eid)
    for name, values in (("adm", adm), ("eid", eid), ("fused", result)):
        np.savez_compressed(work / (name + "_scores.npz"), **values)
    threshold = float(np.quantile(result["validation_scores"], config["alert_quantile"]))
    order = np.argsort(-result["scores"], kind="stable")
    with (work / "ranking.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "node_id", "score", "alert"))
        for rank, index in enumerate(order, 1):
            value = result["scores"][index]
            writer.writerow((rank, result["nodes"][index], float(value), int(value >= threshold)))
    print(str(work / "ranking.csv"))
