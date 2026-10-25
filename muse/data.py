# SPDX-License-Identifier: Apache-2.0
"""Input validation, causal history construction, and frozen future targets."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import re
import zlib

import numpy as np
import torch
from torch.nn import functional as F

from .adm import actor_windows, historical_neighborhood, encode_tokens, action_vector
from .semantics import ProcessSemanticState, action_group, normalize_relation, object_semantics

TYPE_IDS = {"subject": 1, "file": 2, "netflow": 3}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gap_bucket(previous_time, current_time):
    if previous_time is None:
        return 0
    seconds = max(0.0, (current_time - previous_time) / 1e9)
    return min(15, int(math.log2(seconds + 1.0)))


def copy_history(history, length=20):
    output = np.zeros((length, 5), dtype=np.uint16)
    if history:
        values = np.asarray(list(history)[-length:], dtype=np.uint16)
        output[-len(values):] = values
    return output


def label_tokens(value, length=12, vocab_size=8192):
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())[-length:]
    output = np.zeros(length, dtype=np.uint16)
    if tokens:
        output[-len(tokens):] = [1 + zlib.crc32(t.encode("utf-8")) % (vocab_size - 1) for t in tokens]
    return output


def label_id(value, vocab_size=8192):
    normalized = str(value).strip().lower()
    return 1 + zlib.crc32(normalized.encode("utf-8")) % (vocab_size - 1) if normalized else 0


def read_events(path, relation_ids):
    """Read one chronologically ordered host/day stream from normalized JSONL."""
    events = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            actor, target = str(row["actor"]), str(row["object"])
            typ = str(row["object_type"])
            if typ not in TYPE_IDS:
                raise ValueError(f"Unsupported object_type on line {number}: {typ}")
            relation = normalize_relation(row["relation"], typ)
            if relation not in relation_ids:
                raise ValueError(f"Unknown operation on line {number}: {relation}")
            events.append((int(row["timestamp_ns"]), actor, target, relation, typ,
                           str(row.get("object_label", "")), str(row.get("source", actor)),
                           str(row.get("target", target)), str(row.get("actor_label", ""))))
    if not events:
        raise ValueError("Input stream is empty: " + str(path))
    if any(a[0] > b[0] for a, b in zip(events, events[1:])):
        raise ValueError("Input events must be chronological: " + str(path))
    return events


def local_state(counts, count, targets, types):
    vector = np.zeros(16, np.float32)
    if count:
        vector[:12] = counts / count
    vector[12:] = (math.log1p(count), math.log1p(len(targets)), len(types) / 4.0,
                   min(count, 1000) / 1000.0)
    return vector


def neighbor_vector(history, counterpart, time_ns, window=20):
    result = np.zeros(12, np.float32)
    if not history:
        return result
    groups = np.asarray([e[2] for e in history])
    neighbors = Counter(e[1] for e in history)
    result[:6] = np.bincount(groups, minlength=6) / len(history)
    result[6:11] = (len(history) / window, len(neighbors) / window,
                     len(set(groups.tolist())) / 6.0,
                     neighbors.get(counterpart, 0) / len(history),
                     max(neighbors.values()) / len(history))
    age = max(0.0, (time_ns - history[-1][0]) / 1e9)
    result[11] = min(1.0, math.log1p(age) / math.log1p(86400.0))
    return result


def build_histories(events, relation_ids):
    """Read all five views before incorporating the current event."""
    histories = defaultdict(lambda: deque(maxlen=8))
    interaction_history = defaultdict(lambda: deque(maxlen=20))
    last_time, lineage = {}, {}
    counts = defaultdict(lambda: np.zeros(12, np.float32))
    totals, targets, types = Counter(), defaultdict(set), defaultdict(set)
    trackers = defaultdict(ProcessSemanticState)
    endpoint_events = deque(maxlen=2048)
    endpoint_groups = np.zeros(12, np.float32)
    endpoint_actors, endpoint_files, endpoint_registry, endpoint_remotes = (Counter() for _ in range(4))
    endpoint_flags = np.zeros(6, np.float32)
    active_sessions = 0
    output = {k: [] for k in ("history", "state", "next_state", "relation", "type", "gap",
                              "category", "properties", "keys", "actors", "times")}
    neighbors = []

    def endpoint_vector():
        n = len(endpoint_events)
        value = np.zeros(24, np.float32)
        if n:
            value[:12] = endpoint_groups / n
            value[12:17] = (n / 2048.0, math.log1p(len(endpoint_actors)) / math.log1p(2048),
                            math.log1p(len(endpoint_files)) / math.log1p(2048),
                            math.log1p(len(endpoint_registry)) / math.log1p(2048),
                            math.log1p(len(endpoint_remotes)) / math.log1p(2048))
            value[17:21] = endpoint_flags[:4] / n
            value[21] = min(1.0, active_sessions / 64.0)
            value[22:24] = endpoint_flags[4:] / n
        return value

    for event in events:
        time_ns, actor, target, relation, typ, label, source, destination = event[:8]
        rid, tid = relation_ids[relation], TYPE_IDS[typ]
        properties, category = object_semantics(label, typ)
        history = np.zeros((8, 4), np.int64)
        if histories[actor]:
            history[-len(histories[actor]):] = np.asarray(histories[actor])
        before_local = local_state(counts[actor], totals[actor], targets[actor], types[actor])
        before_actor, before_endpoint = trackers[actor].vector(), endpoint_vector()
        inherited = lineage.get(actor, np.zeros(24, np.float32)).copy()
        neighbors.append(np.concatenate((neighbor_vector(interaction_history[actor], target, time_ns),
                                         neighbor_vector(interaction_history[target], actor, time_ns))))
        output["history"].append(history)
        output["state"].append(np.concatenate((before_local, before_actor, before_endpoint, inherited)))
        for key, value in (("relation", rid), ("type", tid), ("gap", gap_bucket(last_time.get(actor), time_ns)),
                           ("category", category), ("properties", properties), ("keys", (source, destination, str(time_ns))),
                           ("actors", actor), ("times", time_ns)):
            output[key].append(value)

        counts[actor][rid % 12] += 1
        totals[actor] += 1
        targets[actor].add(target)
        types[actor].add(tid)
        output["next_state"].append(local_state(counts[actor], totals[actor], targets[actor], types[actor]))
        trackers[actor].update(relation, target, typ, properties)
        handoff = relation == "PROCESS_EXECUTED" and typ == "subject"
        if handoff:
            lineage[target] = (0.75 * trackers[actor].vector() + 0.25 * inherited).astype(np.float32)
        endpoint_type = "registry" if properties[0] else typ
        flags = np.asarray((properties[4], properties[6], properties[1], properties[9], 0., float(handoff)), np.float32)
        if len(endpoint_events) == endpoint_events.maxlen:
            old_group, old_actor, old_target, old_type, old_flags = endpoint_events.popleft()
            endpoint_groups[old_group] -= 1
            for counter, key in ((endpoint_actors, old_actor),
                                 (endpoint_remotes if old_type == "netflow" else
                                  endpoint_registry if old_type == "registry" else endpoint_files, old_target)):
                counter[key] -= 1
                if counter[key] <= 0:
                    del counter[key]
            endpoint_flags[:] -= old_flags
        group = action_group(relation)
        endpoint_events.append((group, actor, target, endpoint_type, flags))
        endpoint_groups[group] += 1
        endpoint_actors[actor] += 1
        counter = endpoint_remotes if typ == "netflow" else endpoint_registry if endpoint_type == "registry" else endpoint_files
        counter[target] += 1
        endpoint_flags[:] += flags
        if relation in {"NETFLOW_CREATED", "NETFLOW_CONNECTED"}:
            active_sessions += 1
        elif relation == "NETFLOW_DISCONNECTED":
            active_sessions = max(0, active_sessions - 1)
        coarse = 0 if group < 2 else 1 if group < 4 else 2 if group < 6 else 3 if group < 8 else 4 if group in (8, 11) else 5
        interaction_history[actor].append((time_ns, target, coarse))
        if target != actor:
            interaction_history[target].append((time_ns, actor, coarse))
        histories[actor].append((rid, 1, tid, output["gap"][-1]))
        last_time[actor] = time_ns
    output = {k: np.asarray(v) for k, v in output.items()}
    for key in ("relation", "type", "gap", "category", "times"):
        output[key] = output[key].astype(np.int64, copy=False)
    output["state"] = np.concatenate((output["state"], historical_neighborhood(neighbors, output["actors"])), -1).astype(np.float32)
    output["properties"] = output["properties"].astype(np.float32)
    return output


def label_tensors(data, selection):
    return tuple(torch.as_tensor(np.array(data[k][selection], copy=True))
                 for k in ("relation", "type", "gap", "category", "properties"))


@torch.no_grad()
def prepare_stream(events, relation_ids, encoder, batch_size=2048):
    data = build_histories(events, relation_ids)
    future, valid = actor_windows(data["actors"], data["times"])
    inputs, targets = [], []
    for start in range(0, len(events), batch_size):
        take = slice(start, start + batch_size)
        hidden = encode_tokens(encoder, torch.as_tensor(data["history"][take]))
        action = action_vector(encoder, *label_tensors(data, take))
        inputs.append(np.concatenate((hidden.numpy(), data["state"][take], action.numpy()), -1).astype(np.float16))
        indices = future[take]
        ok = torch.as_tensor(indices >= 0)
        rel, typ, gap, category, properties = label_tensors(data, np.maximum(indices, 0))
        tokens = torch.stack((rel, torch.ones_like(rel), typ, gap), -1)
        tokens[~ok] = 0
        latent = encode_tokens(encoder, tokens)
        properties = properties * ok[..., None]
        categories = F.one_hot(category.long(), 9).float() * ok[..., None]
        targets.append(torch.cat((latent, properties.float().mean(1), categories.mean(1)), -1).numpy().astype(np.float16))
    return {**{k: data[k] for k in ("relation", "type", "gap", "category", "properties", "keys")},
            "inputs": np.concatenate(inputs), "targets": np.concatenate(targets),
            "future_indices": future, "valid": valid}


def save_stream(folder, data):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        np.save(folder / (key + ".npy"), value, allow_pickle=False)


def load_stream(folder):
    return {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in Path(folder).glob("*.npy")}


def fit_normalization(streams):
    count = 0
    total, square = np.zeros(115, np.float64), np.zeros(115, np.float64)
    for data in streams:
        for start in range(0, len(data["valid"]), 16384):
            target = np.asarray(data["targets"][start:start+16384], np.float64)[data["valid"][start:start+16384]]
            count += len(target)
            total += target.sum(0)
            square += np.square(target).sum(0)
    if count == 0:
        raise ValueError("Training data contain no complete same-actor future windows")
    mean = total / count
    variance = np.maximum(square / count - mean ** 2, 0)
    active = variance > 1e-8
    if not active.any():
        raise ValueError("Future targets have no active dimensions")
    return dict(mean=mean.tolist(), scale=np.maximum(np.sqrt(variance), .01).tolist(), active=active.tolist())


def adm_batch(data, take, normal):
    values = torch.as_tensor(np.array(data["inputs"][take], dtype=np.float32))
    target = torch.as_tensor(np.array(data["targets"][take], dtype=np.float32))
    target = (target - normal["mean"]) / normal["scale"]
    current = label_tensors(data, take)
    future = label_tensors(data, np.maximum(data["future_indices"][take], 0))
    return values[:, :96], values[:, 96:208], values[:, 208:], current, future, target
