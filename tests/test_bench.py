import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench


class UnitTests(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(bench.percentile([1, 2, 3], .5), 2)
        self.assertAlmostEqual(bench.percentile([1, 2, 3], .95), 2.9)
        self.assertIsNone(bench.percentile([None], .95))

    def test_redaction(self):
        self.assertEqual(bench.redact({"api_key": "secret", "x": "a-secret"}, "secret"),
                         {"api_key": "[REDACTED]", "x": "a-[REDACTED]"})

    def test_context_guard(self):
        cfg = bench.load_config(bench.ROOT / "configs/vllm.json")
        bench.validate_run(cfg, ["short", "medium"], True)
        with self.assertRaises(ValueError):
            bench.validate_run(cfg, ["long"], True)
        with self.assertRaises(ValueError):
            bench.validate_run(cfg, ["short"], False)

    def test_profile_never_sweeps(self):
        cfg = bench.load_config(bench.ROOT / "configs/vllm.json")
        spec = bench.scenario_config(cfg, "short", 30, 42, "", 300)["spec"]
        self.assertEqual(spec["profile"]["kind"], "synchronous")
        self.assertEqual(spec["data_loader"]["samples"], 30)
        self.assertEqual(spec["backend"]["extras"]["body"]["temperature"], 0)

    def test_url_guard(self):
        cfg = bench.load_config(bench.ROOT / "configs/vllm.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            cfg["base_url"] = "https://user:password@example.com"
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                bench.load_config(path)

    def test_summary_excludes_errors(self):
        row = {"request_args": json.dumps({"body": {"messages": [{"role": "user", "content": "oi"}], "max_tokens": 128}}),
               "time_to_first_token_ms": 10, "request_latency": .5,
               "inter_token_latency_ms": 20, "output_tokens": 5, "prompt_tokens": 256}
        report = {"benchmarks": [{"requests": {"successful": [row], "errored": [{"request_latency": 100}], "incomplete": []}}]}
        out = bench.summarize(report)
        self.assertEqual(out["errored"], 1)
        self.assertEqual(out["e2e_s_p50"], .5)
        self.assertTrue(out["p95_exploratory"])


if __name__ == "__main__":
    unittest.main()
