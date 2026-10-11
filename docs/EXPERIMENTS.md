# Experiment and reproducibility notes

The authoritative submitted outputs are [`artifacts/report.json`](../artifacts/report.json), the four individual topology reports, [`summary.csv`](../artifacts/summary.csv), and [`locked_config.json`](../artifacts/locked_config.json). The latter records the settings locked before confirmatory test evaluation; the full report supplies additional workload and learning parameters.

## Published experiment

| Setting | Value |
|:--|:--|
| Package version | 0.10.0 |
| Development validation seeds recorded in locked config | 21, 22, 23 |
| Confirmatory seed | 29 |
| External requests | 20,000 |
| Chronological split | 15,000 train / 2,500 validation / 2,500 test |
| Network | 8 routers; producer `node-0`; 7 requesters |
| Star center | `node-1` (`--star-center auto`) |
| Catalog / name categories | 300 / 8 |
| Cache capacity / replacement shortlist | 16 packets / 8 residents per router |
| Packet size | 1 fixed unit; no byte-capacity constraint |
| Zipf exponent / requester skew exponent | 1.1 / 0.7 |
| Temporal locality / window / decay | 0.45 / 64 / 0.88 |
| Popularity drift | Every 1,000 requests; fraction 0.05 |
| Request rate | 20 per second |
| Freshness range | 2–300 seconds |
| Vocabulary / embedding dimension | 4,096 / 32 |
| Name LSTM / context LSTM hidden dimension | 48 / 64 |
| Scoring head hidden dimension / numeric features | 64 / 8 |
| Popularity prior / recency prior / residual scale | 1.0 / 0.0 / 0.05 |
| Popularity decay | 32 router-local observations |
| Supervised pretraining | 1 epoch; LRU observation policy; learning rate 0.001 |
| Ranking horizon / pretraining batch length | 32 / 64 |
| RL fine-tuning | 1 epoch; learning rate 0.0001 |
| Discount / action temperature / rollout length | 0.97 / 0.3 / 256 |
| Online auxiliary future-reuse labels | Disabled |
| Bypass / entropy coefficient | Disabled / 0 |
| Device / PyTorch threads | CPU / 1 per process |
| Compared baselines | LRU, LFU, FIFO, Random |

The original run used four independent topology processes. `scripts/reproduce.py` runs the same experiment serially through the CLI, resetting the seed for each topology. It explicitly sets parameters that differ from CLI defaults, including node count, split, freshness, name categories, requester skew, discount, and temperature.

The development-seed experiments themselves are not bundled. The original training environment was not recorded in the supplied artifacts. Local verification during repository preparation used Python 3.12.0, PyTorch 2.5.1, and NumPy 1.26.4 on CPU. Exact training outputs may vary with library versions and hardware.

## Training and checkpoint selection

LRU replays only the training partition to collect the local request sequences that physically reach each router. Pretraining ranks candidates using inverse distance to their next exact-name request within 32 local observations; observations without a complete future window are censored at the tail. Ranking loss is computed before eviction, including the eventual victim.

RL samples eviction actions from negative retention logits. The reward is 1 for a local cache hit and 0 for a miss. An action taken after request `t` receives credit from future rewards, beginning at `t + 1`. Online topology fine-tuning uses no future-reuse auxiliary labels.

Validation can retain the starting policy if training does not improve it. Test requests are excluded from pretraining, RL, and checkpoint selection. Evaluation resets all caches and router state for each policy.

| Topology | Selected stage | Initial validation hit ratio | Selected validation hit ratio | Test hit ratio |
|:--|:--|--:|--:|--:|
| Line | Pretraining, epoch 1 | 73.20% | 73.32% | 74.64% |
| Ring | Initial causal prior | 73.56% | 73.56% | 74.64% |
| Star | RL, epoch 1 | 73.60% | 73.88% | 74.96% |
| Mesh | Initial causal prior | 73.32% | 73.32% | 75.00% |

For ring and mesh, the neural residual remains at its initial zero output. Those selected policies still apply the LRU shortlist and causal decayed-popularity prior. Consequently, the results cannot isolate the contribution of LSTMs or RL. The `best_epoch: 0` in an RL history means that its starting checkpoint won validation; consult the pretraining history to determine whether that starting checkpoint was itself trained.

## Commands

Install from the repository root:

```bash
python -m pip install -e ".[plots]"
```

Check the included checkpoints without retraining:

```bash
python scripts/verify_results.py
```

Train all four topologies from the published configuration:

```bash
python scripts/reproduce.py
```

Run a subset or a different seed:

```bash
python scripts/reproduce.py --topologies line star --output-dir runs/line-star
python scripts/reproduce.py --seed 30
```

Reuse completed, matching checkpoints:

```bash
python scripts/reproduce.py --resume
```

`--resume` reuses completed training stages only when their experiment fingerprints match. It does not resume an interrupted optimizer step. Generated outputs are written beneath `runs/` by default. Choose an output directory outside `artifacts/` to preserve the submitted experiment.

Regenerate README figures from the submitted CSV and topology builder:

```bash
python scripts/make_figures.py
```

## Metrics and artifact provenance

- **Cache hit ratio:** consumer requests satisfied by any router cache, divided by all consumer requests. This differs from router-level hit ratio, whose denominator is router lookups.
- **Byte hit ratio:** bytes satisfied from caches divided by requested bytes. It equals request hit ratio here because all packet sizes are identical.
- **Producer fetches:** requests that reach the producer.
- **Mean retrieval hops:** Interest hops to the answering cache or producer divided by satisfied requests. A hit at the requester contributes zero hops; this is not round-trip latency.

`report.json` combines the four individual reports by reusing completed checkpoints. Its `resumed` flag is therefore true, while the original individual reports record false; their topology results otherwise agree. Checkpoint paths embedded in those files and their metadata retain the original `artifacts/v010_moderate_seed29/models/` location. The repository preserves these records verbatim and stores the checkpoint binaries at `artifacts/models/`. The verification script resolves that location explicitly and validates their recorded SHA-256 hashes before loading them.

The experiment fingerprints and trace partition digests provide provenance. They do not replace evaluation over multiple seeds, real request traces, broader cache sizes, or ablations of the shortlist, prior, and neural residual.
