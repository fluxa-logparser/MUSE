# SPDX-License-Identifier: Apache-2.0
"""Event-to-node aggregation, normal calibration, and evidence fusion."""
import numpy as np


def aggregate_nodes(keys, scores):
    keys, scores = np.asarray(keys), np.asarray(scores)
    nodes, inverse = np.unique(np.concatenate((keys[:, 0], keys[:, 1])), return_inverse=True)
    values = np.full(len(nodes), -np.inf)
    np.maximum.at(values, inverse[:len(keys)], scores)
    np.maximum.at(values, inverse[len(keys):], scores)
    return nodes, values


def percentile(reference, values):
    reference, values = np.asarray(reference), np.asarray(values)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("Calibration requires finite scores and nonempty validation data")
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


def fuse(adm, eid):
    for key in ("nodes", "validation_nodes"):
        if not np.array_equal(adm[key], eid[key]):
            raise ValueError("Channel node identifiers differ: " + key)
    eps = .5 / (len(adm["validation_nodes"]) + 1)
    result = {"nodes": adm["nodes"], "validation_nodes": adm["validation_nodes"]}
    for prefix in ("", "validation_"):
        a = percentile(adm["validation_scores"], adm[prefix + "scores"])
        b = percentile(eid["validation_scores"], eid[prefix + "scores"])
        result[prefix + "scores"] = np.maximum(a, b) + eps * np.minimum(a, b)
    return result
