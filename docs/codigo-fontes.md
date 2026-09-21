# Código completo: referência linha a linha

Leia junto com [a explicação detalhada](codigo-explicado.md). Gerado dos arquivos reais; não é um segundo programa. Os números não pertencem ao Python.

## bench.py

SHA-256: `2c60765b7039ab2c035111fda25e79dabc14e5879be14619b7a39e003fc166bb`.

| Função/classe | Linhas |
|---|---|
| `local_model_check` | 29–49 |
| `write_json` | 52–53 |
| `redact` | 56–64 |
| `capture` | 67–72 |
| `load_config` | 75–91 |
| `validate_run` | 94–107 |
| `tokenizer_digest` | 110–119 |
| `prepare_tokenizer` | 122–134 |
| `scenario_config` | 137–156 |
| `Monitor` | 159–222 |
| `percentile` | 225–231 |
| `summarize` | 234–273 |
| `write_requests_csv` | 276–289 |
| `write_summary` | 292–302 |
| `run` | 305–428 |
| `positive` | 431–435 |
| `rebuild_report` | 438–452 |
| `main` | 455–504 |
| `__init__` | 161–167 |
| `start` | 169–174 |
| `kv_loop` | 176–199 |
| `loop` | 201–215 |
| `stop` | 217–222 |

```text
0001 | #!/usr/bin/env python3
0002 | """Um usuário: inicialização, primeira resposta, aquecimento e GuideLLM sequencial."""
0003 | from __future__ import annotations
0004 |
0005 | import argparse
0006 | import asyncio
0007 | import csv
0008 | import hashlib
0009 | import importlib.metadata
0010 | import json
0011 | import math
0012 | import os
0013 | from pathlib import Path
0014 | import platform
0015 | import re
0016 | import subprocess
0017 | import sys
0018 | import threading
0019 | from datetime import datetime, timezone
0020 | from urllib.parse import urlsplit
0021 |
0022 | ROOT = Path(__file__).resolve().parent
0023 | VERSION = "0.3.0"
0024 | GUIDELLM_VERSION = "0.7.4"
0025 | WORKLOADS = {"short": 256, "medium": 2048, "long": 8192}
0026 | OUTPUT_TOKENS = 128
0027 |
0028 |
0029 | def local_model_check(value):
0030 |     """Checa presença, não lê pesos nem aquece o page cache deliberadamente."""
0031 |     path = Path(value).expanduser().resolve()
0032 |     if path.is_file():
0033 |         if path.suffix not in {".safetensors", ".gguf", ".bin", ".pt", ".pth"} and not path.name.startswith("sha256-"):
0034 |             raise ValueError("Informe um arquivo de pesos (não configuração/texto) ou blob sha256- do Ollama.")
0035 |         files = [path]
0036 |     elif path.is_dir():
0037 |         files = [f for f in path.rglob("*") if f.is_file() and f.suffix in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}]
0038 |         for index in path.glob("*.index.json"):
0039 |             data = json.loads(index.read_text())
0040 |             missing = [name for name in set(data.get("weight_map", {}).values()) if not (path / name).is_file()]
0041 |             if missing:
0042 |                 raise ValueError(f"Shards ausentes: {missing}")
0043 |     else:
0044 |         files = []
0045 |     if not files or any(f.stat().st_size == 0 for f in files):
0046 |         raise ValueError("Pesos locais ausentes/vazios. Prepare o modelo antes do benchmark; downloads não são permitidos na medição.")
0047 |     return {"policy": "Pesos locais obrigatórios; download excluído do protocolo.", "path": str(path),
0048 |             "files": len(files), "bytes": sum(f.stat().st_size for f in files),
0049 |             "validation": "presença/tamanho e shards declarados; não verifica conteúdo, SSD físico ou vínculo com API"}
0050 |
0051 |
0052 | def write_json(path, data):
0053 |     Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0054 |
0055 |
0056 | def redact(value, secret):
0057 |     if isinstance(value, dict):
0058 |         return {k: ("[REDACTED]" if k.lower() in {"api_key", "authorization"} else redact(v, secret))
0059 |                 for k, v in value.items()}
0060 |     if isinstance(value, list):
0061 |         return [redact(v, secret) for v in value]
0062 |     if isinstance(value, str) and secret:
0063 |         return value.replace(secret, "[REDACTED]")
0064 |     return value
0065 |
0066 |
0067 | def capture(command):
0068 |     try:
0069 |         proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
0070 |         return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
0071 |     except (OSError, subprocess.TimeoutExpired) as exc:
0072 |         return {"unavailable": str(exc)}
0073 |
0074 |
0075 | def load_config(path):
0076 |     cfg = json.loads(Path(path).read_text(encoding="utf-8"))
0077 |     required = {"runtime", "base_url", "model", "tokenizer", "context_window", "cache_policy",
0078 |                 "runtime_version", "model_artifact", "server_command", "notes"}
0079 |     if set(cfg) != required:
0080 |         raise ValueError(f"Campos da configuração devem ser exatamente: {sorted(required)}")
0081 |     if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in required - {"context_window"}):
0082 |         raise ValueError("Os campos textuais da configuração devem estar preenchidos.")
0083 |     if type(cfg["context_window"]) is not int or cfg["context_window"] < 512:
0084 |         raise ValueError("context_window deve ser um inteiro >= 512.")
0085 |     url = urlsplit(cfg["base_url"])
0086 |     if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
0087 |         raise ValueError("URL inválida: use http(s), sem credenciais, query ou fragmento.")
0088 |     if url.path not in {"", "/", "/v1", "/v1/"}:
0089 |         raise ValueError("Use a raiz do servidor (ex.: http://127.0.0.1:8000), sem endpoint.")
0090 |     cfg["base_url"] = f"{url.scheme}://{url.netloc}"
0091 |     return cfg
0092 |
0093 |
0094 | def validate_run(cfg, scenarios, smoke):
0095 |     for scenario in scenarios:
0096 |         # Margem para o template; o servidor continua sendo a autoridade final.
0097 |         needed = WORKLOADS[scenario] + OUTPUT_TOKENS + 256
0098 |         if cfg["context_window"] < needed:
0099 |             raise ValueError(f"{scenario} precisa de contexto declarado >= {needed}; "
0100 |                              "altere o servidor e depois o JSON, ou retire esse cenário.")
0101 |     if not smoke:
0102 |         if any("PREENCHER" in cfg[k] or "SUBSTITUA" in cfg[k]
0103 |                for k in ("model", "runtime_version", "model_artifact", "server_command")):
0104 |             raise ValueError("Preencha modelo, versão, artefato e comando antes da medição formal; --smoke permite rascunhos.")
0105 |         if cfg["cache_policy"] == "runtime-default-unverified":
0106 |             raise ValueError("Registre cache_policy após verificar o servidor. Ex.: disabled-confirmed ou enabled-recorded. "
0107 |                              "O script NÃO altera nem comprova a política de cache.")
0108 |
0109 |
0110 | def tokenizer_digest(path):
0111 |     path = Path(path)
0112 |     if not path.is_dir() or not (path / "tokenizer_config.json").exists():
0113 |         raise ValueError("Tokenizer local ausente. Execute: python bench.py prepare-tokenizer")
0114 |     hashes = {}
0115 |     for file in sorted(path.rglob("*")):
0116 |         if file.is_file() and not file.name.startswith("."):
0117 |             hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
0118 |     digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
0119 |     return {"sha256": digest, "files": hashes}
0120 |
0121 |
0122 | def prepare_tokenizer(args):
0123 |     from huggingface_hub import HfApi
0124 |     from transformers import AutoTokenizer
0125 |     path = Path(args.output)
0126 |     if path.exists():
0127 |         raise ValueError(f"{path} já existe; não será sobrescrito. Use outro --output.")
0128 |     revision = HfApi().model_info(args.model, revision=args.revision).sha
0129 |     tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
0130 |     path.mkdir(parents=True)
0131 |     tokenizer.save_pretrained(path)
0132 |     write_json(path / "source.json", {"model": args.model, "revision": revision})
0133 |     print(f"Tokenizer salvo em {path}; revisão {revision}. Não foram baixados pesos.")
0134 |     print("Compartilhe esta pasta com o grupo para usar os mesmos arquivos.")
0135 |
0136 |
0137 | def scenario_config(cfg, name, count, seed, secret, timeout):
0138 |     return {
0139 |         "spec": {
0140 |             "backend": {"kind": "openai_http", "target": cfg["base_url"], "model": cfg["model"],
0141 |                         "request_format": "/v1/chat/completions", "stream": True,
0142 |                         "validate_backend": False, "verify": True, "follow_redirects": False,
0143 |                         "timeout": timeout, "api_key": secret or None,
0144 |                         "extras": {"body": {"temperature": 0, "top_p": 1}}},
0145 |             "profile": {"kind": "synchronous", "warmup": 0, "cooldown": 0},
0146 |             "constraints": [{"kind": "max_requests", "count": count}, {"kind": "max_errors", "count": 1}],
0147 |             "tokenizer": {"kind": "huggingface_auto", "model": str(Path(cfg["tokenizer"]).resolve()),
0148 |                           "load_kwargs": {"local_files_only": True, "trust_remote_code": False}},
0149 |             "data": [{"kind": "synthetic_text", "prompt_tokens": WORKLOADS[name],
0150 |                       "output_tokens": OUTPUT_TOKENS}],
0151 |             "data_loader": {"kind": "pytorch", "samples": count, "num_workers": 0, "shuffle": False},
0152 |             "seed": {"kind": "static", "value": seed},
0153 |             "metrics": {"kind": "generative", "sample_size": None, "prefer_response_metrics": True},
0154 |             "outputs": [],
0155 |         }
0156 |     }
0157 |
0158 |
0159 | class Monitor:
0160 |     """Amostragem best effort; CSV local, não um profiler de largura de banda."""
0161 |     def __init__(self, output, cfg=None, secret="", collect_kv=False):
0162 |         self.output = Path(output)
0163 |         self.stop_event = threading.Event()
0164 |         self.thread = None
0165 |         self.phase = "setup"
0166 |         self.cfg, self.secret, self.collect_kv = cfg, secret, collect_kv
0167 |         self.kv_thread = None
0168 |
0169 |     def start(self):
0170 |         self.thread = threading.Thread(target=self.loop, daemon=True)
0171 |         self.thread.start()
0172 |         if self.collect_kv:
0173 |             self.kv_thread = threading.Thread(target=self.kv_loop, daemon=True)
0174 |             self.kv_thread.start()
0175 |
0176 |     def kv_loop(self):
0177 |         import httpx
0178 |         headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
0179 |         pattern = re.compile(r'^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc))(\{[^}]*\})?\s+([0-9.eE+\-]+)(?:\s|$)')
0180 |         with (self.output / "kv-cache.csv").open("w", newline="") as handle, httpx.Client(timeout=1, headers=headers, follow_redirects=False) as client:
0181 |             writer = csv.writer(handle)
0182 |             writer.writerow(["utc", "phase", "series", "fraction"])
0183 |             while not self.stop_event.is_set():
0184 |                 phase = self.phase
0185 |                 try:
0186 |                     response = client.get(self.cfg["base_url"] + "/metrics")
0187 |                     response.raise_for_status()
0188 |                     matches = [m for line in response.text.splitlines() if (m := pattern.match(line))]
0189 |                     modern = any(m[1] == "vllm:kv_cache_usage_perc" for m in matches)
0190 |                     for m in matches:
0191 |                         if modern and m[1] != "vllm:kv_cache_usage_perc":
0192 |                             continue
0193 |                         value = float(m[3])
0194 |                         if math.isfinite(value) and 0 <= value <= 1:
0195 |                             writer.writerow([datetime.now(timezone.utc).isoformat(), phase, redact(m[1] + (m[2] or ""), self.secret), value])
0196 |                     handle.flush()
0197 |                 except (httpx.HTTPError, ValueError):
0198 |                     pass  # Serveur inicializando/endpoint ausente: nunca inventar zeros.
0199 |                 self.stop_event.wait(1)
0200 |
0201 |     def loop(self):
0202 |         with (self.output / "gpu.csv").open("w", newline="", encoding="utf-8") as handle:
0203 |             writer = csv.writer(handle)
0204 |             writer.writerow(["utc", "phase", "index", "name", "used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"])
0205 |             while not self.stop_event.is_set():
0206 |                 phase = self.phase
0207 |                 result = capture(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
0208 |                                   "--format=csv,noheader,nounits"])
0209 |                 if result.get("returncode") != 0:
0210 |                     write_json(self.output / "gpu-unavailable.json", result)
0211 |                     return
0212 |                 for row in csv.reader(result["stdout"].splitlines(), skipinitialspace=True):
0213 |                     writer.writerow([datetime.now(timezone.utc).isoformat(), phase, *row])
0214 |                 handle.flush()
0215 |                 self.stop_event.wait(1)
0216 |
0217 |     def stop(self):
0218 |         self.stop_event.set()
0219 |         if self.thread:
0220 |             self.thread.join(timeout=17)
0221 |         if self.kv_thread:
0222 |             self.kv_thread.join(timeout=3)
0223 |
0224 |
0225 | def percentile(values, q):
0226 |     values = sorted(v for v in values if v is not None and math.isfinite(v))
0227 |     if not values:
0228 |         return None
0229 |     pos = (len(values) - 1) * q
0230 |     lo, hi = math.floor(pos), math.ceil(pos)
0231 |     return values[lo] + (values[hi] - values[lo]) * (pos - lo)
0232 |
0233 |
0234 | def summarize(report):
0235 |     """Métricas por requisição, sem misturar warmup ou erros com sucessos."""
0236 |     benchmarks = report["benchmarks"]
0237 |     if len(benchmarks) != 1:
0238 |         raise ValueError("Esperado exatamente um benchmark sequencial.")
0239 |     requests = benchmarks[0]["requests"]
0240 |     from reporting import derived
0241 |     good = [{**r, **derived(r)} for r in requests["successful"]]
0242 |     result = {
0243 |         "successful_request_count": len(good),
0244 |         "errored_request_count": len(requests["errored"]),
0245 |         "incomplete_request_count": len(requests["incomplete"]),
0246 |     }
0247 |     # Os nomes são deliberadamente longos: summary.json é um artefato de
0248 |     # análise, e não uma API em que economizar alguns bytes melhora algo.
0249 |     metrics = {
0250 |         "request_first_token_latency_milliseconds": "time_to_first_token_ms",
0251 |         "request_latency_seconds": "request_latency",
0252 |         "within_response_next_token_latency_milliseconds": "inter_token_latency_ms",
0253 |         "output_completion_token_count": "output_tokens",
0254 |         "input_prompt_token_count": "prompt_tokens",
0255 |         "decode_generation_tokens_per_second": "decode_tokens_s",
0256 |         "effective_output_tokens_per_second": "effective_tokens_s",
0257 |     }
0258 |     for label, key in metrics.items():
0259 |         vals = [r.get(key) for r in good]
0260 |         result[label + "_sample_count"] = sum(v is not None for v in vals)
0261 |         result[label + "_p50"] = percentile(vals, .5)
0262 |         result[label + "_p95"] = percentile(vals, .95)
0263 |         result[label + "_p99"] = percentile(vals, .99)
0264 |     # O hash exclui aliases do modelo e chaves: apenas carga de entrada e limite de saída.
0265 |     bodies = []
0266 |     for row in good:
0267 |         args = json.loads(row["request_args"])
0268 |         body = args.get("body", {})
0269 |         bodies.append({k: body.get(k) for k in ("messages", "max_tokens")})
0270 |     result["requests_sha256"] = hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
0271 |     result["percentiles_are_exploratory"] = len(good) < 100
0272 |     result["percentile_definition"] = "Empirical linear interpolation over successful requests in this phase/scenario/repetition block."
0273 |     return result
0274 |
0275 |
0276 | def write_requests_csv(path, report):
0277 |     """Amostras individuais para análise no R/Python, incluindo status de erro."""
0278 |     from reporting import derived
0279 |     fields = ["status", "request_id", "request_start_time", "request_latency", "time_to_first_token_ms",
0280 |               "inter_token_latency_ms", "prompt_tokens", "output_tokens", "decode_tokens_s", "effective_tokens_s",
0281 |               "context_start_tokens", "context_end_tokens", "context_band"]
0282 |     with Path(path).open("w", newline="", encoding="utf-8") as handle:
0283 |         writer = csv.DictWriter(handle, fieldnames=fields)
0284 |         writer.writeheader()
0285 |         for status in ("successful", "errored", "incomplete"):
0286 |             for row in report["benchmarks"][0]["requests"][status]:
0287 |                 if status == "successful":
0288 |                     row = {**row, **derived(row)}
0289 |                 writer.writerow({"status": status, **{key: row.get(key) for key in fields[1:]}})
0290 |
0291 |
0292 | def write_summary(output, rows):
0293 |     from reporting import render
0294 |     write_json(output / "summary.json", rows)
0295 |     if not rows:
0296 |         render(output, rows)
0297 |         return
0298 |     with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
0299 |         writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
0300 |         writer.writeheader()
0301 |         writer.writerows(rows)
0302 |     render(output, rows)
0303 |
0304 |
0305 | def run(args):
0306 |     import fcntl
0307 |     from lifecycle import DEFAULT_PROMPT, Launch, lifecycle_report, timed_request, wait_models
0308 |     if importlib.metadata.version("guidellm") != GUIDELLM_VERSION:
0309 |         raise ValueError(f"Este projeto exige guidellm=={GUIDELLM_VERSION}; reinstale requirements.txt.")
0310 |     cfg = load_config(args.config)
0311 |     validate_run(cfg, args.scenarios, args.smoke)
0312 |     availability = local_model_check(args.local_model_path)
0313 |     availability["kv_bytes_per_token"] = args.kv_bytes_per_token
0314 |     availability["launch_hf_offline"] = bool(args.launch)
0315 |     digest = tokenizer_digest(cfg["tokenizer"])
0316 |     count, repetitions = (3, 1) if args.smoke else (args.requests, args.repetitions)
0317 |     secret = os.environ.get("BENCH_API_KEY", "")
0318 |     prompt = Path(args.first_prompt_file).read_text(encoding="utf-8") if args.first_prompt_file else DEFAULT_PROMPT
0319 |     if not prompt.strip():
0320 |         raise ValueError("O prompt inicial não pode estar vazio.")
0321 |     base = Path(args.results)
0322 |     base.mkdir(parents=True, exist_ok=True)
0323 |     with (base / ".benchmark.lock").open("a") as lock:
0324 |         try:
0325 |             fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
0326 |         except BlockingIOError:
0327 |             raise ValueError("Já existe um benchmark usando esta pasta results. Não execute dois ao mesmo tempo.") from None
0328 |         stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
0329 |         output = base / stamp
0330 |         output.mkdir()
0331 |         manifest = {"project_version": VERSION, "guidellm_version": GUIDELLM_VERSION, "started_utc": stamp,
0332 |                     "config": cfg, "tokenizer": digest, "smoke": args.smoke, "requests": count,
0333 |                     "repetitions": repetitions, "warmup_requests_per_case": args.warmup,
0334 |                     "scenarios": args.scenarios, "seed": args.seed, "profile": "synchronous",
0335 |                     "model_availability": availability,
0336 |                     "guidellm_compat": "0.7.4 bounded drain of real late completion updates (5s); no request retry",
0337 |                     "telemetry": {"gpu_source": "local nvidia-smi", "kv_metrics_requested": args.collect_kv_metrics,
0338 |                                   "sampling": "approximately 1 Hz; not per-token"},
0339 |                     "python": sys.version, "platform": platform.platform(),
0340 |                     "git": capture(["git", "rev-parse", "HEAD"]), "status": "running"}
0341 |         write_json(output / "manifest.json", redact(manifest, secret))
0342 |         write_json(output / "client-packages.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
0343 |         write_json(output / "gpu-before.json", capture(["nvidia-smi"]))
0344 |         monitor, rows = Monitor(output, cfg, secret, args.collect_kv_metrics), []
0345 |         launch, origin, cleanup_error = None, None, None
0346 |         lifecycle = {"mode": "new-process" if args.launch else "existing-server-state-unknown",
0347 |                      "status": "running", "initial_state_note": args.initial_state,
0348 |                      "startup_timeout_s": args.startup_timeout}
0349 |         print(f"Resultados: {output.resolve()}", flush=True)
0350 |         try:
0351 |             monitor.start()
0352 |             if args.launch:
0353 |                 launch = Launch(cfg, args.launch, output)
0354 |                 lifecycle["argv"] = redact(launch.argv, secret)
0355 |                 monitor.phase = "process_startup"
0356 |                 origin = launch.start()
0357 |                 lifecycle["pid"] = launch.process.pid
0358 |             lifecycle_report(output, redact(lifecycle, secret))
0359 |             lifecycle["readiness"] = wait_models(cfg, secret, args.startup_timeout, launch)
0360 |             lifecycle_report(output, redact(lifecycle, secret))
0361 |             monitor.phase = "first_request"
0362 |             print("Primeiro POST: medição da primeira resposta (nenhuma geração prévia enviada pelo cliente).", flush=True)
0363 |             lifecycle["first_request"] = timed_request(cfg, secret, args.timeout, prompt,
0364 |                                                        output / "first-request.json", origin)
0365 |             lifecycle_report(output, redact(lifecycle, secret))
0366 |             from guidellm.benchmark import BenchmarkScenario, benchmark_generative_text
0367 |             from guidellm_compat import completion_drain
0368 |             for rep in range(repetitions):
0369 |                 # Rotação balanceia parcialmente a posição dos cenários entre repetições.
0370 |                 names = args.scenarios[rep % len(args.scenarios):] + args.scenarios[:rep % len(args.scenarios)]
0371 |                 for name in names:
0372 |                     for phase, n in (("warmup", args.warmup), ("measure", count)):
0373 |                         if not n:
0374 |                             continue
0375 |                         seed = args.seed + rep * 100 + list(WORKLOADS).index(name)
0376 |                         if phase == "warmup":
0377 |                             seed += 1_000_000
0378 |                         prefix = f"r{rep+1}-{name}-{phase}"
0379 |                         monitor.phase = prefix
0380 |                         config = scenario_config(cfg, name, n, seed, secret, args.timeout)
0381 |                         write_json(output / f"{prefix}-config.json", redact(config, secret))
0382 |                         print(f"{prefix}: {n} requisições, uma por vez", flush=True)
0383 |                         with completion_drain():
0384 |                             report, _ = asyncio.run(benchmark_generative_text(BenchmarkScenario.model_validate(config)))
0385 |                         raw = redact(report.model_dump(mode="json"), secret)
0386 |                         write_json(output / f"{prefix}.json", raw)
0387 |                         write_requests_csv(output / f"{prefix}-requests.csv", raw)
0388 |                         summary = summarize(raw)
0389 |                         summary["expected"] = n
0390 |                         summary["missing_request_count"] = max(0, n - sum(summary[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count")))
0391 |                         rows.append({"runtime": cfg["runtime"], "model": cfg["model"],
0392 |                                      "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
0393 |                                      "phase": phase, "scenario": name, "repetition": rep+1, **summary})
0394 |                         write_summary(output, rows)
0395 |                         if summary["successful_request_count"] != n or summary["errored_request_count"] or summary["incomplete_request_count"]:
0396 |                             raise RuntimeError(f"{prefix}: requisições falharam ou execução incompleta. Veja o JSON; não compare como sucesso.")
0397 |             monitor.phase = "warm_reference"
0398 |             lifecycle["warm_reference"] = timed_request(cfg, secret, args.timeout, prompt,
0399 |                                                          output / "warm-reference.json")
0400 |             manifest["status"] = lifecycle["status"] = "complete"
0401 |         except BaseException as exc:
0402 |             manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0403 |             manifest["error"] = redact(str(exc), secret)
0404 |             lifecycle["status"] = manifest["status"]
0405 |             lifecycle["error"] = manifest["error"]
0406 |             raise
0407 |         finally:
0408 |             for key, filename in (("first_request", "first-request.json"), ("warm_reference", "warm-reference.json")):
0409 |                 if (output / filename).exists():
0410 |                     lifecycle[key] = json.loads((output / filename).read_text(encoding="utf-8"))
0411 |             if launch is not None:
0412 |                 monitor.phase = "server_shutdown"
0413 |                 try:
0414 |                     launch.close()
0415 |                     lifecycle["server_cleanup"] = "stopped owned process group only"
0416 |                 except Exception as exc:
0417 |                     cleanup_error = redact(str(exc), secret)
0418 |                     lifecycle["server_cleanup_error"] = cleanup_error
0419 |                     lifecycle["status"] = manifest["status"] = "failed"
0420 |             lifecycle_report(output, redact(lifecycle, secret))
0421 |             monitor.stop()
0422 |             manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
0423 |             write_json(output / "manifest.json", redact(manifest, secret))
0424 |             write_json(output / "gpu-after.json", capture(["nvidia-smi"]))
0425 |             write_summary(output, rows)
0426 |         if cleanup_error:
0427 |             raise RuntimeError(f"Falha ao encerrar processo criado: {cleanup_error}. Confira o PID no lifecycle.json.")
0428 |         print(f"Concluído. Abra {output / 'lifecycle.html'} e {output / 'summary.html'}")
0429 |
0430 |
0431 | def positive(value):
0432 |     number = int(value)
0433 |     if number <= 0:
0434 |         raise argparse.ArgumentTypeError("Use um inteiro positivo.")
0435 |     return number
0436 |
0437 |
0438 | def rebuild_report(args):
0439 |     """Atualiza apenas derivados, preservando relatórios brutos e manifesto original."""
0440 |     output = Path(args.output)
0441 |     rows = json.loads((output / "summary.json").read_text())
0442 |     manifest = json.loads((output / "manifest.json").read_text())
0443 |     for row in rows:
0444 |         prefix = f"r{row['repetition']}-{row['scenario']}-{row['phase']}"
0445 |         raw = json.loads((output / f"{prefix}.json").read_text())
0446 |         row.update(summarize(raw))
0447 |         expected = manifest.get("warmup_requests_per_case") if row["phase"] == "warmup" else manifest.get("requests")
0448 |         row["expected"] = expected
0449 |         row["missing_request_count"] = max(0, expected - sum(row[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count"))) if expected is not None else None
0450 |         write_requests_csv(output / f"{prefix}-requests.csv", raw)
0451 |     write_summary(output, rows)
0452 |     print(f"Relatório atualizado: {output / 'summary.html'}; dados brutos preservados.")
0453 |
0454 |
0455 | def main():
0456 |     parser = argparse.ArgumentParser(description=__doc__)
0457 |     commands = parser.add_subparsers(dest="command", required=True)
0458 |     report = commands.add_parser("report", help="Regenera derivados de uma execução existente, sem nova inferência.")
0459 |     report.add_argument("--output", required=True)
0460 |     report.set_defaults(func=rebuild_report)
0461 |     prep = commands.add_parser("prepare-tokenizer", help="Baixa apenas tokenizer; fixa revisão e guarda origem.")
0462 |     prep.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
0463 |     prep.add_argument("--revision", default="main")
0464 |     prep.add_argument("--output", default="tokenizer")
0465 |     prep.set_defaults(func=prepare_tokenizer)
0466 |     cmd = commands.add_parser("run", help="Mede primeiro acesso, aquecimento e GuideLLM; lançamento do servidor é opcional.")
0467 |     cmd.add_argument("--config", required=True)
0468 |     cmd.add_argument("--local-model-path", required=True, help="Pesos já no SSD: pasta HF, arquivo GGUF ou blob local do Ollama. Não baixa arquivos.")
0469 |     cmd.add_argument("--collect-kv-metrics", action="store_true", help="Amostra /metrics do vLLM (~1 Hz); ocupação do pool KV, não bytes.")
0470 |     cmd.add_argument("--kv-bytes-per-token", type=positive, help="Opcional: bytes de KV lógico por token, calculados para arquitetura/dtype reais. Estimativa, não VRAM medida.")
0471 |     cmd.add_argument("--input-tokens", nargs="+", type=positive, help="Substitui --scenarios por uma grade de comprimentos sintéticos, ex.: 256 512 1024 2048 3072.")
0472 |     cmd.add_argument("--scenarios", nargs="+", choices=list(WORKLOADS), default=["short", "medium"])
0473 |     cmd.add_argument("--requests", type=positive, default=30)
0474 |     cmd.add_argument("--repetitions", type=positive, default=3)
0475 |     cmd.add_argument("--warmup", type=positive, default=3)
0476 |     cmd.add_argument("--seed", type=positive, default=42)
0477 |     cmd.add_argument("--timeout", type=positive, default=300)
0478 |     cmd.add_argument("--results", default="results")
0479 |     cmd.add_argument("--smoke", action="store_true", help="3 medições e 1 repetição; não vale como resultado final.")
0480 |     cmd.add_argument("--launch", help="Arquivo JSON com argv para iniciar um runtime LOCAL; encerra só esse processo ao final.")
0481 |     cmd.add_argument("--startup-timeout", type=positive, default=1800, help="Limite da espera pela API com --launch, em segundos.")
0482 |     cmd.add_argument("--first-prompt-file", help="Texto UTF-8 para a primeira requisição e referência final; default: pergunta sobre RAM/VRAM.")
0483 |     cmd.add_argument("--initial-state", default="weights local; OS/compilation caches not controlled", help="Descreva SSD e caches existentes; apenas registra, não limpa.")
0484 |     cmd.set_defaults(func=run)
0485 |     args = parser.parse_args()
0486 |     if getattr(args, "input_tokens", None):
0487 |         if len(set(args.input_tokens)) != len(args.input_tokens):
0488 |             parser.error("Não repita comprimentos em --input-tokens.")
0489 |         args.scenarios = []
0490 |         for size in args.input_tokens:
0491 |             name = f"ctx{size}"
0492 |             WORKLOADS[name] = size
0493 |             args.scenarios.append(name)
0494 |     if hasattr(args, "scenarios") and len(set(args.scenarios)) != len(args.scenarios):
0495 |         parser.error("Não repita cenários na lista.")
0496 |     try:
0497 |         args.func(args)
0498 |     except KeyboardInterrupt:
0499 |         print("Interrompido; resultados já concluídos foram preservados.", file=sys.stderr)
0500 |         return 130
0501 |     except Exception as exc:
0502 |         print(f"ERRO: {redact(str(exc), os.environ.get('BENCH_API_KEY', ''))}", file=sys.stderr)
0503 |         return 1
0504 |     return 0
0505 |
0506 |
0507 | if __name__ == "__main__":
0508 |     raise SystemExit(main())
```

## lifecycle.py

SHA-256: `fe67f97a9c2a4f14371c0b42c928dbe8ce429068e8111f94dee2378d9b91477c`.

| Função/classe | Linhas |
|---|---|
| `Launch` | 23–75 |
| `wait_models` | 78–109 |
| `timed_request` | 112–183 |
| `lifecycle_report` | 186–201 |
| `__init__` | 24–35 |
| `start` | 37–55 |
| `close` | 57–75 |

```text
0001 | """Cronometria de inicialização/primeiro stream, fora das fases GuideLLM.
0002 |
0003 | Nenhum POST de inferência é enviado antes da primeira requisição medida.
0004 | O lançamento é opt-in e usa argv sem shell. Só o processo criado é encerrado.
0005 | """
0006 | import json
0007 | import os
0008 | from pathlib import Path
0009 | import signal
0010 | import socket
0011 | import subprocess
0012 | import time
0013 | from urllib.parse import urlsplit
0014 |
0015 | import httpx
0016 |
0017 | DEFAULT_PROMPT = (
0018 |     "Explique para um estudante de estatística a diferença entre memória RAM e VRAM. "
0019 |     "Inclua um exemplo de uso de um chatbot e conclua com um resumo em três itens."
0020 | )
0021 |
0022 |
0023 | class Launch:
0024 |     def __init__(self, cfg, command_file, output):
0025 |         url = urlsplit(cfg["base_url"])
0026 |         if url.hostname not in {"127.0.0.1", "localhost", "::1"}:
0027 |             raise ValueError("--launch só aceita servidor local (localhost).")
0028 |         self.host, self.port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
0029 |         self.argv = json.loads(Path(command_file).read_text(encoding="utf-8"))
0030 |         if not isinstance(self.argv, list) or not self.argv or any(not isinstance(x, str) or not x for x in self.argv):
0031 |             raise ValueError("O arquivo --launch deve conter um array JSON não vazio de strings (argv).")
0032 |         self.output = Path(output)
0033 |         self.process = None
0034 |         self.log = None
0035 |         self.started = None
0036 |
0037 |     def start(self):
0038 |         # Não interrompe servidores existentes nem tenta tomar uma porta ocupada.
0039 |         try:
0040 |             connection = socket.create_connection((self.host, self.port), timeout=1)
0041 |         except OSError:
0042 |             pass
0043 |         else:
0044 |             connection.close()
0045 |             raise ValueError("A porta já está em uso. Pare o seu servidor manualmente antes de usar --launch.")
0046 |         self.log = (self.output / "server.log").open("w", encoding="utf-8")
0047 |         self.started = time.perf_counter()
0048 |         try:
0049 |             self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
0050 |                                             start_new_session=True, shell=False,
0051 |                                             env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
0052 |         except BaseException:
0053 |             self.log.close()
0054 |             raise
0055 |         return self.started
0056 |
0057 |     def close(self):
0058 |         """Encerra exclusivamente o grupo criado por este objeto, inclusive filhos."""
0059 |         if self.process is not None:
0060 |             try:
0061 |                 os.killpg(self.process.pid, signal.SIGTERM)
0062 |             except ProcessLookupError:
0063 |                 pass
0064 |             try:
0065 |                 self.process.wait(timeout=20)
0066 |             except subprocess.TimeoutExpired:
0067 |                 os.killpg(self.process.pid, signal.SIGKILL)
0068 |                 self.process.wait(timeout=5)
0069 |             # Alguns workers podem sobreviver ao encerramento do processo líder.
0070 |             try:
0071 |                 os.killpg(self.process.pid, signal.SIGKILL)
0072 |             except ProcessLookupError:
0073 |                 pass
0074 |         if self.log is not None:
0075 |             self.log.close()
0076 |
0077 |
0078 | def wait_models(cfg, secret, timeout, launch=None):
0079 |     """Apenas GET. Modelo listado não comprova que seus pesos estão na GPU."""
0080 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0081 |     started = time.perf_counter()
0082 |     probes, last, next_update = 0, None, started + 15
0083 |     with httpx.Client(timeout=2, headers=headers, follow_redirects=False) as client:
0084 |         while True:
0085 |             probes += 1
0086 |             if launch is not None and launch.process.poll() is not None:
0087 |                 raise RuntimeError(f"Servidor encerrou antes de ficar disponível (exit={launch.process.returncode}). Veja server.log.")
0088 |             try:
0089 |                 response = client.get(cfg["base_url"] + "/v1/models")
0090 |                 if response.status_code in {401, 403}:
0091 |                     raise ValueError("API recusou autenticação; confira BENCH_API_KEY.")
0092 |                 response.raise_for_status()
0093 |                 models = [m["id"] for m in response.json().get("data", [])]
0094 |                 if cfg["model"] in models:
0095 |                     observed = time.perf_counter()
0096 |                     return {"models": models, "get_probes": probes,
0097 |                             "wait_wall_s": observed - started,
0098 |                             "process_to_api_observed_s": observed - launch.started if launch else None,
0099 |                             "criterion": "GET /v1/models retornou o ID; não é prova de pesos residentes"}
0100 |                 last = f"ID ausente; disponíveis: {models}"
0101 |             except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
0102 |                 last = str(exc)
0103 |             elapsed = time.perf_counter() - started
0104 |             if not launch or elapsed >= timeout:
0105 |                 raise RuntimeError(f"API/modelo não disponível após {elapsed:.1f}s: {last}")
0106 |             if time.perf_counter() >= next_update:
0107 |                 print(f"Aguardando API do processo iniciado: {elapsed:.0f}s...", flush=True)
0108 |                 next_update = time.perf_counter() + 15
0109 |             time.sleep(min(.5, max(0, timeout - elapsed)))
0110 |
0111 |
0112 | def timed_request(cfg, secret, timeout, prompt, output, process_origin=None):
0113 |     """Primeiro POST é simultaneamente medição e validação, sem pré-aquecimento oculto.
0114 |
0115 |     Grava resultado parcial inclusive em timeout, stream inválido ou usage ausente.
0116 |     TTFT aqui é primeiro conteúdo não vazio recebido (não mero cabeçalho/role).
0117 |     """
0118 |     path = Path(output)
0119 |     body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
0120 |             "temperature": 0, "top_p": 1, "max_tokens": 128, "stream": True,
0121 |             "stream_options": {"include_usage": True}}
0122 |     result = {"status": "running", "body": body, "stream_usage": None,
0123 |               "content_event_offsets_s": [], "output": "", "done": False,
0124 |               "ttft_ms": None, "e2e_s": None, "mean_itl_ms": None,
0125 |               "process_to_first_content_s": None, "process_to_response_end_s": None}
0126 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0127 |     started = None
0128 |     try:
0129 |         with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
0130 |             started = time.perf_counter()
0131 |             with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as stream:
0132 |                 result["headers_ms"] = (time.perf_counter() - started) * 1000
0133 |                 stream.raise_for_status()
0134 |                 if "text/event-stream" not in stream.headers.get("content-type", ""):
0135 |                     raise ValueError("A API não respondeu com SSE.")
0136 |                 for line in stream.iter_lines():
0137 |                     if not line.startswith("data:"):
0138 |                         continue
0139 |                     value = line[5:].strip()
0140 |                     if value == "[DONE]":
0141 |                         result["done"] = True
0142 |                         break
0143 |                     event = json.loads(value)
0144 |                     if "error" in event:
0145 |                         raise ValueError(f"Erro no stream: {event['error']}")
0146 |                     result["stream_usage"] = event.get("usage") or result["stream_usage"]
0147 |                     text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
0148 |                     if text:
0149 |                         now = time.perf_counter()
0150 |                         result["content_event_offsets_s"].append(now - started)
0151 |                         result["output"] += text
0152 |                         if result["ttft_ms"] is None:
0153 |                             result["ttft_ms"] = (now - started) * 1000
0154 |                             if process_origin is not None:
0155 |                                 result["process_to_first_content_s"] = now - process_origin
0156 |             ended = time.perf_counter()
0157 |             result["e2e_s"] = ended - started
0158 |             if process_origin is not None:
0159 |                 result["process_to_response_end_s"] = ended - process_origin
0160 |             if not result["done"] or not result["output"]:
0161 |                 raise ValueError("Stream incompleto ou sem conteúdo.")
0162 |             usage = result["stream_usage"]
0163 |             if not usage or not all(type(usage.get(k)) is int and usage[k] > 0 for k in ("prompt_tokens", "completion_tokens")):
0164 |                 raise ValueError("usage ausente/inválido no stream; tempos parciais preservados, contagens não estimadas.")
0165 |             offsets = result["content_event_offsets_s"]
0166 |             if usage["completion_tokens"] > 1:
0167 |                 result["mean_itl_ms"] = 1000 * (offsets[-1] - offsets[0]) / (usage["completion_tokens"] - 1)
0168 |             from reporting import derived
0169 |             result.update(derived({"output_tokens": usage["completion_tokens"], "prompt_tokens": usage["prompt_tokens"],
0170 |                                    "inter_token_latency_ms": result["mean_itl_ms"], "request_latency": result["e2e_s"]}))
0171 |             result["status"] = "complete"
0172 |     except BaseException as exc:
0173 |         result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0174 |         result["error"] = str(exc)
0175 |         if started is not None:
0176 |             result["elapsed_until_exit_s"] = time.perf_counter() - started
0177 |         raise
0178 |     finally:
0179 |         serialized = json.dumps(result, ensure_ascii=False, indent=2)
0180 |         if secret:
0181 |             serialized = serialized.replace(secret, "[REDACTED]")
0182 |         path.write_text(serialized + "\n", encoding="utf-8")
0183 |     return result
0184 |
0185 |
0186 | def lifecycle_report(output, lifecycle):
0187 |     import html
0188 |     output = Path(output)
0189 |     (output / "lifecycle.json").write_text(json.dumps(lifecycle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0190 |     rows = []
0191 |     readiness = lifecycle.get("readiness", {})
0192 |     rows.append(("Processo → API observada (s)", readiness.get("process_to_api_observed_s")))
0193 |     for phase in ("first_request", "warm_reference"):
0194 |         req = lifecycle.get(phase, {})
0195 |         for metric in ("ttft_ms", "decode_tokens_s", "effective_tokens_s", "e2e_s", "mean_itl_ms", "process_to_first_content_s", "process_to_response_end_s"):
0196 |             if phase == "warm_reference" and metric.startswith("process_"):
0197 |                 continue
0198 |             rows.append((phase + " · " + metric, req.get(metric)))
0199 |     table = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value)) if value is not None else 'Não medido'}</td></tr>" for label, value in rows)
0200 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ciclo de vida</title><style>body{{font:17px/1.7 system-ui;background:#f6f3ec;color:#193835;margin:25px}}td,th{{padding:12px;border-bottom:1px solid #ccd6cc;text-align:left}}table{{width:100%;overflow-wrap:anywhere}}a{{color:#136d58}}</style><h1>Inicialização e primeira resposta</h1><p>Modo: {html.escape(lifecycle['mode'])}. Status: {html.escape(lifecycle['status'])}.</p><p>API disponível não implica modelo na GPU. O primeiro POST é cronometrado, sem teste de geração anterior. Processo novo não implica caches de disco/CUDA frios. A referência final repete o prompt e pode aproveitar prefix caching.</p><table>{table}</table><p><a href="summary.html">Aquecimento e blocos GuideLLM</a> · <a href="lifecycle.json">Dados do ciclo de vida</a></p></html>'''
0201 |     (output / "lifecycle.html").write_text(page, encoding="utf-8")
```

## reporting.py

SHA-256: `00c798d6a5f5687dee1b8b3463f582aaf8c9771c96f32f32612aed44e0125a81`.

| Função/classe | Linhas |
|---|---|
| `ratio` | 10–13 |
| `percentile` | 16–23 |
| `derived` | 26–36 |
| `context_band` | 39–46 |
| `table` | 49–56 |
| `context_summary` | 59–96 |
| `gpu_summary` | 99–121 |
| `render` | 124–188 |
| `fmt` | 50–53 |

```text
0001 | """Métricas derivadas e relatório offline; não confunde contexto com VRAM."""
0002 | import csv
0003 | import html
0004 | import json
0005 | import math
0006 | from collections import defaultdict
0007 | from pathlib import Path
0008 |
0009 |
0010 | def ratio(a, b):
0011 |     if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
0012 |         return None
0013 |     return a / b if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0 else None
0014 |
0015 |
0016 | def percentile(values, q):
0017 |     """Percentil empírico com interpolação linear; ausências não viram zero."""
0018 |     values = sorted(v for v in values if v is not None and math.isfinite(v))
0019 |     if not values:
0020 |         return None
0021 |     position = (len(values) - 1) * q
0022 |     lower, upper = math.floor(position), math.ceil(position)
0023 |     return values[lower] + (values[upper] - values[lower]) * (position - lower)
0024 |
0025 |
0026 | def derived(row):
0027 |     n, p = row.get("output_tokens"), row.get("prompt_tokens")
0028 |     itl = row.get("inter_token_latency_ms")
0029 |     return {
0030 |         "decode_tokens_s": ratio(1000, itl) if n is not None and n > 1 else None,
0031 |         "effective_tokens_s": ratio(n, row.get("request_latency")),
0032 |         "context_start_tokens": p,
0033 |         # Comprimento lógico final; não é o número exato de posições materializadas.
0034 |         "context_end_tokens": p + n if p is not None and n is not None else None,
0035 |         "context_band": context_band(p),
0036 |     }
0037 |
0038 |
0039 | def context_band(p):
0040 |     if p is None:
0041 |         return "não informado"
0042 |     for limit in (512, 1024, 2048, 4096, 8192, 16384):
0043 |         if p < limit:
0044 |             lower = 0 if limit == 512 else limit // 2
0045 |             return f"[{lower}, {limit})"
0046 |     return "[16384, +∞)"
0047 |
0048 |
0049 | def table(rows, columns):
0050 |     def fmt(v):
0051 |         if v is None:
0052 |             return "Não disponível"
0053 |         return f"{v:.3f}" if isinstance(v, float) else str(v)
0054 |     head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
0055 |     body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(r.get(k)))}</td>" for k, _ in columns) + "</tr>" for r in rows)
0056 |     return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
0057 |
0058 |
0059 | def context_summary(output, kv_bytes_per_token=None):
0060 |     groups = defaultdict(list)
0061 |     for file in sorted(Path(output).glob("r*-*-requests.csv")):
0062 |         rep, scenario, phase, _ = file.stem.split("-", 3)
0063 |         with file.open() as handle:
0064 |             for row in csv.DictReader(handle):
0065 |                 if row["status"] != "successful":
0066 |                     continue
0067 |                 for key in ("prompt_tokens", "output_tokens", "inter_token_latency_ms", "request_latency", "time_to_first_token_ms"):
0068 |                     row[key] = float(row[key]) if row.get(key) else None
0069 |                 row.update(derived(row))
0070 |                 # Preservamos o cenário mesmo quando dois cenários caem na
0071 |                 # mesma faixa de contexto. Isso evita que short/medium/long
0072 |                 # apareçam como uma única população no JSON derivado.
0073 |                 groups[(phase, rep, scenario, row["context_band"])].append(row)
0074 |     result = []
0075 |     for (phase, rep, scenario, band), rows in groups.items():
0076 |         entry = {"phase": phase, "repetition": rep, "scenario": scenario,
0077 |                  "context_band": band, "successful_request_count": len(rows)}
0078 |         context_metrics = {
0079 |             "decode_generation_tokens_per_second": "decode_tokens_s",
0080 |             "effective_output_tokens_per_second": "effective_tokens_s",
0081 |             "request_first_token_latency_milliseconds": "time_to_first_token_ms",
0082 |             "initial_context_input_token_count": "context_start_tokens",
0083 |             "final_logical_context_token_count": "context_end_tokens",
0084 |         }
0085 |         for label, key in context_metrics.items():
0086 |             values = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
0087 |             entry[label + "_sample_count"] = len(values)
0088 |             entry[label + "_p50"] = percentile(values, .50)
0089 |             entry[label + "_p95"] = percentile(values, .95)
0090 |             entry[label + "_p99"] = percentile(values, .99)
0091 |         result.append(entry)
0092 |         for edge in ("start", "end"):
0093 |             tokens_key = "initial_context_input_token_count_p50" if edge == "start" else "final_logical_context_token_count_p50"
0094 |             tokens = entry[tokens_key]
0095 |             entry[f"estimated_{edge}_logical_kv_cache_mebibytes"] = tokens * kv_bytes_per_token / 1048576 if tokens is not None and kv_bytes_per_token else None
0096 |     return result
0097 |
0098 |
0099 | def gpu_summary(output):
0100 |     groups = defaultdict(list)
0101 |     path = Path(output) / "gpu.csv"
0102 |     if path.exists():
0103 |         with path.open() as handle:
0104 |             for row in csv.DictReader(handle):
0105 |                 groups[(row["phase"], row["index"], row["name"])].append(row)
0106 |     result = []
0107 |     for (phase, index, name), rows in groups.items():
0108 |         entry = {"phase": phase, "gpu": index, "name": name, "samples": len(rows)}
0109 |         for key in ("used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"):
0110 |             values = []
0111 |             for row in rows:
0112 |                 try:
0113 |                     value = float(row[key])
0114 |                     if math.isfinite(value):
0115 |                         values.append(value)
0116 |                 except (ValueError, TypeError, KeyError):
0117 |                     pass
0118 |             entry[key + "_mean"] = sum(values) / len(values) if values else None
0119 |             entry[key + "_max"] = max(values) if values else None
0120 |         result.append(entry)
0121 |     return result
0122 |
0123 |
0124 | def render(output, rows):
0125 |     output = Path(output)
0126 |     lifecycle = json.loads((output / "lifecycle.json").read_text()) if (output / "lifecycle.json").exists() else {}
0127 |     manifest = json.loads((output / "manifest.json").read_text()) if (output / "manifest.json").exists() else {}
0128 |     kv_bytes = manifest.get("model_availability", {}).get("kv_bytes_per_token")
0129 |     context, gpu = context_summary(output, kv_bytes), gpu_summary(output)
0130 |     (output / "context-summary.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
0131 |     (output / "gpu-summary.json").write_text(json.dumps(gpu, ensure_ascii=False, indent=2) + "\n")
0132 |     startup = lifecycle.get("readiness", {}).get("process_to_api_observed_s")
0133 |     initial = []
0134 |     for key, label in (("first_request", "Primeira resposta"), ("warm_reference", "Referência final")):
0135 |         req = lifecycle.get(key, {})
0136 |         usage = req.get("stream_usage") or {}
0137 |         d = derived({"output_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
0138 |                      "inter_token_latency_ms": req.get("mean_itl_ms"), "request_latency": req.get("e2e_s")})
0139 |         initial.append({"phase": label, "ttft": req.get("ttft_ms"), "e2e": req.get("e2e_s"), **d})
0140 |     metrics = table(rows, [("phase", "Fase"), ("scenario", "Cenário"), ("repetition", "Repetição"),
0141 |         ("expected", "Requisições previstas"), ("successful_request_count", "Requisições bem-sucedidas"),
0142 |         ("errored_request_count", "Requisições com erro"), ("incomplete_request_count", "Requisições incompletas"),
0143 |         ("missing_request_count", "Requisições ausentes do relatório bruto"),
0144 |         ("request_first_token_latency_milliseconds_p50", "Latência da requisição até primeiro token p50 (ms)"),
0145 |         ("request_first_token_latency_milliseconds_p95", "Latência da requisição até primeiro token p95 (ms)"),
0146 |         ("request_first_token_latency_milliseconds_p99", "Latência da requisição até primeiro token p99 (ms)"),
0147 |         ("decode_generation_tokens_per_second_p50", "Velocidade de geração p50 (tokens/s)"),
0148 |         ("effective_output_tokens_per_second_p50", "Velocidade efetiva de saída p50 (tokens/s)"),
0149 |         ("request_latency_seconds_p50", "Latência total p50 (s)"),
0150 |         ("input_prompt_token_count_p50", "Tokens de entrada p50"),
0151 |         ("output_completion_token_count_p50", "Tokens de saída p50")])
0152 |     by_context = table(context, [("phase", "Fase"), ("repetition", "Repetição"), ("scenario", "Cenário"), ("context_band", "Faixa de entrada (tokens)"),
0153 |         ("successful_request_count", "Requisições bem-sucedidas"),
0154 |         ("initial_context_input_token_count_p50", "Tokens de entrada inicial p50"),
0155 |         ("final_logical_context_token_count_p50", "Tokens de contexto lógico final p50"),
0156 |         ("estimated_start_logical_kv_cache_mebibytes", "KV lógico inicial estimado (MiB)"),
0157 |         ("estimated_end_logical_kv_cache_mebibytes", "KV lógico final estimado (MiB)"),
0158 |         ("request_first_token_latency_milliseconds_p50", "Latência da requisição até primeiro token p50 (ms)"),
0159 |         ("decode_generation_tokens_per_second_p50", "Velocidade de geração p50 (tokens/s)"),
0160 |         ("effective_output_tokens_per_second_p50", "Velocidade efetiva p50 (tokens/s)")])
0161 |     hardware = table(gpu, [("phase", "Fase"), ("gpu", "GPU"), ("name", "Nome"), ("samples", "Amostras"),
0162 |         ("used_mib_max", "Memória máx. (MiB)"), ("total_mib_max", "Memória total (MiB)"),
0163 |         ("gpu_util_pct_mean", "Utilização média (%)"), ("gpu_util_pct_max", "Utilização máx. (%)"),
0164 |         ("temperature_c_max", "Temperatura máx. (°C)"), ("power_w_mean", "Potência média (W)"), ("power_w_max", "Potência máx. (W)")]) if gpu else "<p>Não disponível: nenhuma amostra NVIDIA válida. Em Apple/Metal este coletor não mede GPU; isso não significa utilização zero.</p>"
0165 |     kvfile = output / "kv-cache.csv"
0166 |     kvrows = []
0167 |     if kvfile.exists():
0168 |         with kvfile.open() as handle:
0169 |             groups = defaultdict(list)
0170 |             for r in csv.DictReader(handle):
0171 |                 groups[(r["phase"], r["series"])].append(float(r["fraction"]) * 100)
0172 |             kvrows = [{"phase": p, "series": s, "n": len(v), "mean": sum(v)/len(v), "max": max(v)} for (p, s), v in groups.items()]
0173 |     kv = table(kvrows, [("phase", "Fase"), ("series", "Série do servidor"), ("n", "Amostras"), ("mean", "Ocupação média (%)"), ("max", "Ocupação máx. (%)")]) if kvrows else "<p>Ocupação real de KV não disponível nesta execução. Não foi estimada a partir da VRAM.</p>"
0174 |     first = table(initial, [("phase", "Fase"), ("ttft", "TTFT (ms)"), ("decode_tokens_s", "Geração (tokens/s)"), ("effective_tokens_s", "Efetiva (tokens/s)"), ("e2e", "Total (s)")])
0175 |     policy = manifest.get("model_availability", {}).get("policy", "Execução anterior: veja o estado inicial; ausência de download não verificada por esta versão.")
0176 |     failure = f'<p class="note">Execução não concluída: {html.escape(str(manifest["error"]))}. Dados parciais não constituem uma bateria válida.</p>' if manifest.get("error") else ""
0177 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark · latência, geração e GPU</title>
0178 | <style>body{{font:16px/1.7 system-ui;margin:32px;background:#f6f3ec;color:#193835}}main{{max-width:1400px;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border:1px solid #ccd6cc;text-align:left}}th{{background:#e0e9df}}.scroll{{overflow:auto}}h2{{margin-top:38px}}a{{color:#136d58}}.note{{padding:16px;background:#fff0de;border-left:4px solid #b54e27}}</style><main>
0179 | <h1>Um usuário · latência, geração e GPU</h1><p>Status: <strong>{html.escape(manifest.get('status', 'desconhecido'))}</strong>. {html.escape(policy)}</p>{failure}
0180 | <p class="note">TTFT = espera pelo primeiro token/conteúdo observado. Geração = (tokens de saída − 1)/(tempo entre primeiro e último token). Efetiva = tokens de saída/tempo total da requisição, incluindo TTFT. São taxas por requisição, não throughput agregado de usuários.</p>
0181 | <h2>1. Inicialização e primeira resposta</h2><p>Processo → API disponível: {html.escape(str(startup)) if startup is not None else 'não medido'} s. <a href="lifecycle.html">Ver ciclo de vida completo</a>.</p>{first}
0182 | <h2>2. Aquecimento e operação posterior</h2><p>Percentis entre requisições bem-sucedidas. Warmup e measure separados; p95 com menos de 100 sucessos é exploratório. Geração indisponível com menos de dois tokens ou intervalo não positivo.</p>{metrics}
0183 | <h2>3. Tokens/s por faixa de contexto — proxy da carga de KV</h2><p>Faixa definida pela entrada real, incluindo template, antes do decode. O contexto cresce durante a saída; mostramos também seu comprimento lógico final. Esta é uma comparação de velocidades médias de respostas iniciadas em cada faixa, não uma medição token a token dentro de faixas de ocupação física do cache.</p>{by_context}
0184 | <p>Para atenção completa, mantendo modelo, dtype de KV e uma sequência: KV lógico ≈ 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento × tokens. Pesos 4/8 bits não determinam o dtype do KV. Blocos, reserva, prefix caching e sliding window impedem tratar essa fórmula como medição de VRAM. MiB estimados só aparecem com --kv-bytes-per-token informado e verificado pelo operador; caso contrário, ficam indisponíveis.</p>
0185 | <h2>4. GPU por fase</h2><p>Host do cliente; execute no mesmo pod do servidor. Aproximadamente 1 amostra/s, todas as GPUs visíveis, sem atribuição por processo. Máximos amostrados podem perder picos. N/A é ausência de dado, não zero.</p>{hardware}
0186 | <h2>5. Ocupação real do pool KV — vLLM</h2><p>Coleta opcional de /metrics via --collect-kv-metrics. Percentual de blocos ocupados do pool, não percentual de VRAM nem bytes. Séries/engines separados. Amostragem e atualização do servidor podem perder transientes; não sincronizada por token.</p>{kv}
0187 | <p><a href="summary.json">Resumo JSON</a> · <a href="context-summary.json">Faixas JSON</a> · <a href="gpu-summary.json">GPU JSON</a> · <a href="manifest.json">Manifesto</a></p></main></html>'''
0188 |     (output / "summary.html").write_text(page, encoding="utf-8")
```

## guidellm_compat.py

SHA-256: `44ca9b6f443b04d7021323eb8584a6d29721d50c32b5f39bc549f371be70a5a7`.

| Função/classe | Linhas |
|---|---|
| `_is_incomplete` | 16–22 |
| `completion_drain` | 26–87 |

```text
0001 | """Narrow compatibility workarounds for the pinned GuideLLM release."""
0002 |
0003 | from __future__ import annotations
0004 |
0005 | import asyncio
0006 | from collections.abc import Iterator
0007 | from contextlib import contextmanager
0008 | from importlib.metadata import version
0009 | import time
0010 | from typing import Any
0011 |
0012 |
0013 | GUIDELLM_VERSION = "0.7.4"
0014 |
0015 |
0016 | def _is_incomplete(state: Any) -> bool:
0017 |     """Return whether a scheduler snapshot still has non-terminal requests."""
0018 |     return (
0019 |         state is not None
0020 |         and state.created_requests > state.processed_requests
0021 |         and state.processing_requests > 0
0022 |     )
0023 |
0024 |
0025 | @contextmanager
0026 | def completion_drain(timeout: float = 5.0) -> Iterator[None]:
0027 |     """Drain a late final scheduler update affected by GuideLLM 0.7.4's race.
0028 |
0029 |     GuideLLM 0.7.4 may set ``shutdown_event`` in its receive callback immediately
0030 |     before that callback's final update reaches the local receive buffer.  The
0031 |     stock iterator can observe the event during its timeout and return first.
0032 |
0033 |     This context manager temporarily wraps ``WorkerProcessGroup.request_updates``.
0034 |     After the stock iterator returns, the wrapper reads only genuine messages from
0035 |     GuideLLM's receive queue for at most ``timeout`` seconds, and only if the last
0036 |     yielded snapshot is demonstrably incomplete and shutdown has been signalled.
0037 |     If no update arrives, it returns the incomplete result unchanged so the
0038 |     caller's existing completeness guard remains authoritative.
0039 |
0040 |     This is a process-global monkey patch; do not overlap benchmark runs or use the
0041 |     context manager concurrently from multiple threads.
0042 |     """
0043 |     if timeout <= 0:
0044 |         raise ValueError("timeout must be greater than zero")
0045 |     installed = version("guidellm")
0046 |     if installed != GUIDELLM_VERSION:
0047 |         raise RuntimeError(
0048 |             f"completion_drain supports guidellm=={GUIDELLM_VERSION}, found {installed}"
0049 |         )
0050 |
0051 |     from guidellm.scheduler.worker_group import WorkerProcessGroup
0052 |
0053 |     original = WorkerProcessGroup.request_updates
0054 |
0055 |     async def request_updates_with_completion_drain(self):
0056 |         last_state = None
0057 |         async for update in original(self):
0058 |             last_state = update[3]
0059 |             yield update
0060 |
0061 |         shutdown_event = self.shutdown_event
0062 |         messaging = self.messaging
0063 |         if (
0064 |             not _is_incomplete(last_state)
0065 |             or shutdown_event is None
0066 |             or not shutdown_event.is_set()
0067 |             or messaging is None
0068 |         ):
0069 |             return
0070 |
0071 |         deadline = time.monotonic() + timeout
0072 |         while _is_incomplete(last_state):
0073 |             remaining = deadline - time.monotonic()
0074 |             if remaining <= 0:
0075 |                 return
0076 |             try:
0077 |                 update = await messaging.get(timeout=remaining)
0078 |             except asyncio.TimeoutError:
0079 |                 return
0080 |             last_state = update[3]
0081 |             yield update
0082 |
0083 |     WorkerProcessGroup.request_updates = request_updates_with_completion_drain
0084 |     try:
0085 |         yield
0086 |     finally:
0087 |         WorkerProcessGroup.request_updates = original
```
