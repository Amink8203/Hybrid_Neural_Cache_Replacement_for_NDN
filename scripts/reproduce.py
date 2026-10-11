"""Run the published seed-29 experiment without overwriting submitted artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--topologies", nargs="+", choices=["line", "ring", "star", "mesh"],
                        default=["line", "ring", "star", "mesh"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or Path(f"runs/seed{args.seed}")).resolve()
    submitted = (ROOT / "artifacts").resolve()
    if output == submitted or submitted in output.parents or output in submitted.parents:
        parser.error("choose an output directory outside the submitted artifacts tree")
    report = json.loads((submitted / "report.json").read_text(encoding="utf-8"))
    locked = json.loads((submitted / "locked_config.json").read_text(encoding="utf-8"))
    workload = report["trace"]["synthetic_config"]
    training = report["pretraining_config"]
    model = report["model_config"]
    total = workload["num_requests"]
    command = [sys.executable, "-m", "ndn_cache", "topology-train"]
    options = {
        "requests": total,
        "nodes": report["network"]["routers"],
        "catalog-size": workload["catalog_size"],
        "cache-size": report["cache"]["items_per_router"],
        "packet-size": workload["fixed_object_size"],
        "zipf-alpha": workload["zipf_alpha"],
        "temporal-locality": workload["temporal_locality"],
        "locality-window": workload["locality_window"],
        "locality-decay": workload["locality_decay"],
        "drift-interval": workload["drift_interval"],
        "drift-fraction": workload["drift_fraction"],
        "request-rate": workload["request_rate"],
        "min-freshness": workload["min_freshness_period"],
        "max-freshness": workload["max_freshness_period"],
        "name-categories": workload["num_name_categories"],
        "node-skew-alpha": 0.7,
        "train-fraction": report["trace"]["train_requests"] / total,
        "validation-fraction": report["trace"]["validation_requests"] / total,
        "pretrain-epochs": locked["pretrain_epochs"],
        "pretrain-batch-length": 64,
        "pretrain-learning-rate": locked["pretrain_learning_rate"],
        "pretrain-behavior": report["pretraining_behavior_policy"],
        "epochs": locked["rl_epochs"],
        "producer-node": report["network"]["producer_node"],
        "producer-prefix": report["network"]["producer_prefix"],
        "star-center": report["network"]["star_center"],
        "seed": args.seed,
        "device": "cpu",
        "torch-threads": 1,
        "checkpoint-dir": output / "models",
        "output": output / "report.json",
    }
    for key in ("vocab_size", "max_name_components", "embedding_dim", "name_hidden_dim",
                "context_hidden_dim", "head_hidden_dim", "dropout", "popularity_prior_weight",
                "recency_prior_weight", "residual_scale"):
        options[key.replace("_", "-")] = model[key]
    for key in ("learning_rate", "gamma", "rollout_length", "temperature", "entropy_coefficient",
                "auxiliary_coefficient", "prediction_horizon", "calibration_samples",
                "replacement_shortlist_size", "gradient_clip", "recency_scale", "popularity_decay"):
        options[key.replace("_", "-")] = training[key]
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    command.extend(["--topologies", *args.topologies, "--baselines", *locked["baselines"]])
    command.append("--allow-bypass" if training["allow_bypass"] else "--no-allow-bypass")
    if args.resume:
        command.append("--resume")
    print(f"Training {', '.join(args.topologies)} with seed {args.seed}; output: {output}", flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
