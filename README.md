# MUSE

MUSE learns normal operational dynamics from audit events and produces a calibrated ranking of suspicious entities. It consists of five historical views, Agent Dynamics Modeling (ADM), Entity Interaction Dynamics Modeling (EID), and node-level evidence fusion.

## Installation

Use Python 3.10 or 3.11 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root, or install the package with `python -m pip install -e .`.

## Input

Provide normalized audit records as UTF-8 JSONL, one interaction per line. Each file represents one day of a single host or trace, ordered by timestamp. The required fields are:

| Field | Meaning |
|---|---|
| `timestamp_ns` | Integer event timestamp in nanoseconds |
| `actor` | Stable process identifier |
| `object` | Stable object identifier |
| `relation` | Operation in the configured vocabulary, or a supported audit operation name |
| `object_type` | `subject`, `file`, or `netflow` |

Optional fields are `actor_label`, `object_label`, and `source`/`target` for the original directed edge identifiers. The actor/object roles are supplied explicitly. Missing labels become empty strings; omitted source/target identifiers default to actor/object.

Populate the `train`, `validation`, and `test` lists in `config.json` with paths to your files, in chronological order within each split. Paths resolve relative to the configuration file. Training and validation inputs must contain normal activity. Labels and attack annotations are not inputs to fitting or scoring. Use a separate work directory and namespace for each independent host/trace model.

## Run

```bash
python -m muse fit --config config.json --work work
python -m muse score --config config.json --work work
```

`fit` runs Local encoder pretraining, frozen-target preparation, ADM initialization and fine-tuning, and persistent EID training. Each stage selects its checkpoint by normal-validation loss. Fitting requires a fresh work directory. `score` uses those trained models and the same configuration to generate `ranking.csv`, calibrated channel scores, and fused node scores inside the work directory. `prepare` rebuilds the frozen inputs from an existing `local.pt` checkpoint.

## Method

The Local, Actor, Endpoint, Lineage, and Neighborhood views contain pre-event information. Their learned projections are gated and concatenated into a 64-dimensional belief. The Local GRU is pretrained and frozen; the subsequent view projections and prediction heads are trainable.

ADM predicts the current action and a 115-dimensional representation of the next four strictly later interactions of the same actor within the day. The training objective combines behavioral loss, diagonal-Gaussian negative log-likelihood, and weighted future-observation prediction. ADM training uses complete future windows. Target normalization uses normal-training statistics, an active-variance cutoff of `1e-8`, and a standard-deviation floor of `0.01`.

EID evaluates relation, resource semantics, and pair timing using 20-event entity histories and persistent directed-pair hidden states. Events are scored before their pair state is updated. Pair states continue across files in the same stream; independent epochs and splits start from empty state. Truncated backpropagation detaches hidden states without clearing their values.

ADM divides behavioral and transition scores by their normal-validation 95th percentiles and caps transition evidence. Incomplete future windows contribute behavioral evidence only to ADM. Event scores are assigned to both endpoints and aggregated by maximum. Normal-validation empirical percentiles calibrate the two channels, which are fused as `max(p_ADM,p_EID) + epsilon * min(p_ADM,p_EID)`, with `epsilon = 0.5 / (N_validation + 1)`. The normal-validation 99th percentile determines the alert flag.

Scoring processes supplied files in batches. Future evidence is computed from complete future windows, so the final ranking is produced after the supplied stream is available. The output is a node ranking; it does not reconstruct attack paths.

## Source layout

```text
muse/encoder.py      Local recurrent encoder
muse/adm.py          Belief construction and agent dynamics
muse/eid.py          Entity dynamics and persistent pair state
muse/semantics.py    Operation and resource semantics
muse/data.py         Input, causal histories, and future targets
muse/training.py     Optimization and checkpoint selection
muse/scoring.py      Node aggregation, calibration, and fusion
muse/pipeline.py     Integrated fitting and scoring
muse/cli.py          Command-line interface
config.json         Model settings and user-supplied input paths
```

The repository contains source code and configuration only. Runtime models, input files, caches, and outputs are created in the specified work directory. The source is distributed under the accompanying [Apache-2.0 license](LICENSE).
