import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
from lifecycle import Launch


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
        self.assertEqual(out["errored_request_count"], 1)
        self.assertEqual(out["request_latency_seconds_p50"], .5)
        self.assertTrue(out["percentiles_are_exploratory"])
        self.assertEqual(out["decode_generation_tokens_per_second_p50"], 50)
        self.assertEqual(out["effective_output_tokens_per_second_p50"], 10)
        self.assertEqual(out["time_to_first_token_milliseconds_p99"], 10)

    def test_local_weights_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(ValueError):
                bench.local_model_check(path)
            (path / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(bench.local_model_check(path)["bytes"], 7)
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "Shards"):
                bench.local_model_check(path)

    def test_local_weights_rejects_empty_and_non_weight_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            empty = path / "empty.safetensors"
            empty.touch()
            with self.assertRaisesRegex(ValueError, "ausentes/vazios"):
                bench.local_model_check(empty)

            config = path / "config.json"
            config.write_text("{}")
            with self.assertRaisesRegex(ValueError, "arquivo de pesos"):
                bench.local_model_check(config)

    def test_launch_sets_huggingface_offline_environment(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as free:
            free.bind(("127.0.0.1", 0))
            port = free.getsockname()[1]
            env_file = Path(tmp) / "child-env.json"
            code = (
                "import json, os, pathlib; "
                f"pathlib.Path({str(env_file)!r}).write_text(json.dumps({{'HF_HUB_OFFLINE': os.getenv('HF_HUB_OFFLINE'), 'TRANSFORMERS_OFFLINE': os.getenv('TRANSFORMERS_OFFLINE')}})); "
                "import time; time.sleep(30)"
            )
            argv_file = Path(tmp) / "argv.json"
            argv_file.write_text(json.dumps([sys.executable, "-c", code]))
            launch = Launch({"base_url": f"http://127.0.0.1:{port}"}, argv_file, tmp)
            launch.start()
            try:
                deadline = time.monotonic() + 3
                while not env_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(env_file.exists())
                self.assertEqual(json.loads(env_file.read_text()), {
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            finally:
                launch.close()

    def test_rate_edge_cases_and_bands(self):
        from reporting import derived, context_band, gpu_summary, ratio
        for n, itl in ((1, 10), (3, 0), (3, None), (3, float("nan"))):
            self.assertIsNone(derived({"output_tokens": n, "inter_token_latency_ms": itl})["decode_tokens_s"])
        self.assertIsNone(derived({"output_tokens": 2, "prompt_tokens": None,
                                   "request_latency": 0})["context_end_tokens"])
        self.assertIsNone(ratio(1, float("inf")))
        self.assertIsNone(ratio("1", 2))
        self.assertEqual(context_band(511), "[0, 512)")
        self.assertEqual(context_band(512), "[512, 1024)")
        self.assertEqual(context_band(2077), "[2048, 4096)")
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "gpu.csv").write_text(
                "phase,index,name,used_mib,total_mib,gpu_util_pct,temperature_c,power_w\n"
                "measure,0,GPU,N/A,,nan,N/A,bad\n")
            row = gpu_summary(tmp)[0]
            self.assertIsNone(row["used_mib_mean"])
            self.assertIsNone(row["gpu_util_pct_max"])

    def test_gpu_aggregation_and_html_escaping(self):
        from reporting import gpu_summary, render
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "gpu.csv").write_text("utc,phase,index,name,used_mib,total_mib,gpu_util_pct,temperature_c,power_w\n"
                "now,measure,0,<GPU>,10,24,50,60,N/A\nnow,measure,0,<GPU>,20,24,70,65,100\n"
                "now,warmup,0,<GPU>,5,24,20,50,80\nnow,measure,1,other,8,24,25,40,70\n")
            rows = gpu_summary(p)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["used_mib_max"], 20)
            self.assertEqual(rows[0]["gpu_util_pct_mean"], 60)
            self.assertEqual(rows[0]["power_w_mean"], 100)
            render(p, [])
            self.assertIn("&lt;GPU&gt;", (p / "summary.html").read_text())

    def test_context_estimate(self):
        from reporting import context_summary
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "r1-short-measure-requests.csv").write_text("status,prompt_tokens,output_tokens,inter_token_latency_ms,request_latency,time_to_first_token_ms\nsuccessful,256,128,20,3,460\nerrored,256,0,0,10,0\n")
            row = context_summary(p, 196608)[0]
            self.assertEqual(row["successful_request_count"], 1)
            self.assertEqual(row["scenario"], "short")
            self.assertEqual(row["estimated_start_logical_kv_cache_mebibytes"], 48)
            self.assertEqual(row["decode_generation_tokens_per_second_p50"], 50)


if __name__ == "__main__":
    unittest.main()
