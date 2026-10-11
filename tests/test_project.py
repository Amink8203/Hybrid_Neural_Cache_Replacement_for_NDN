"""Regression checks for package imports, NDN behavior, and the public CLI."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ndn_cache.baselines import BaselineCacheController
from ndn_cache.packets import DataPacket, Interest, RequestEvent
from ndn_cache.store import ContentStore
from ndn_cache.topology import TopologySimulator, build_topology

ROOT = Path(__file__).resolve().parents[1]


class ProjectChecks(unittest.TestCase):
    def test_torch_import_does_not_shadow_standard_library_code(self):
        result = subprocess.run([sys.executable, "-c",
            "import code, torch, ndn_cache; assert hasattr(code, 'InteractiveConsole')"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_freshness_boundary_and_packet_capacity(self):
        store = ContentStore(max_items=1, max_bytes=4)
        store.insert(DataPacket("/dataset/item/a", size_bytes=4, freshness_period=2), now=1)
        self.assertIsNotNone(store.lookup(Interest("/dataset/item/a", timestamp=2.9)))
        self.assertIsNone(store.lookup(Interest("/dataset/item/a", timestamp=3)))
        self.assertFalse(store.has_capacity(DataPacket("/dataset/item/b")))
        store.validate()

    def test_star_hub_is_separate_from_producer(self):
        topology = build_topology("star", num_nodes=8, star_center="auto")
        self.assertEqual(topology.route("node-2", "/dataset/item/a").path,
                         ("node-2", "node-1", "node-0"))
        self.assertEqual(len(topology.edges), 7)

    def test_reverse_path_cache_serves_followup_request(self):
        topology = build_topology("line", num_nodes=3)
        events = [RequestEvent(0, "node-2", "/dataset/item/a"),
                  RequestEvent(1, "node-1", "/dataset/item/a")]
        result = TopologySimulator(topology, BaselineCacheController("lru", cache_items=2)).run(events)
        self.assertEqual(result.metrics.producer_fetches, 1)
        self.assertEqual(result.metrics.cache_hit_requests, 1)
        self.assertEqual(result.metrics.interest_hops, 2)

    def test_cli_trains_saves_and_reloads_all_topologies(self):
        from ndn_cache.checkpoint import load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            command = [sys.executable, "-m", "ndn_cache", "topology-train",
                "--topologies", "line", "ring", "star", "mesh", "--nodes", "4",
                "--requests", "160", "--catalog-size", "20", "--cache-size", "3",
                "--replacement-shortlist-size", "2", "--prediction-horizon", "4",
                "--pretrain-epochs", "1", "--epochs", "1", "--packet-size", "1",
                "--checkpoint-dir", str(folder / "models"), "--output", str(folder / "report.json")]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["comparison"]), 20)
            for kind in ["line", "ring", "star", "mesh"]:
                model, _, _ = load_checkpoint(folder / "models" / f"{kind}.pt")
                self.assertEqual(model.config.feature_dim, 8)
                metrics = report["topologies"][kind]["neural_test"]["network_metrics"]
                self.assertEqual(metrics["unsatisfied_requests"], 0)
                self.assertEqual(metrics["cache_hit_ratio"], metrics["byte_cache_hit_ratio"])


if __name__ == "__main__":
    unittest.main()
