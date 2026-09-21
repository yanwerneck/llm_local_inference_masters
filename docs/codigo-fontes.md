# Código completo: referência linha a linha

Leia junto com [a explicação detalhada](codigo-explicado.md). Gerado dos arquivos reais; não é um segundo programa. Os números não pertencem ao Python.

## bench.py

SHA-256: `df1a27e27ab687cc2bcaa64a4687c17e6a81f3205b2a7ddac8a918795114daba`.

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
| `summarize` | 234–259 |
| `write_requests_csv` | 262–275 |
| `write_summary` | 278–288 |
| `run` | 291–414 |
| `positive` | 417–421 |
| `rebuild_report` | 424–438 |
| `main` | 441–490 |
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
0242 |     result = {"successful": len(good), "errored": len(requests["errored"]), "incomplete": len(requests["incomplete"])}
0243 |     metrics = {"ttft_ms": "time_to_first_token_ms", "e2e_s": "request_latency",
0244 |                "mean_itl_ms": "inter_token_latency_ms", "output_tokens": "output_tokens", "prompt_tokens": "prompt_tokens",
0245 |                "decode_tokens_s": "decode_tokens_s", "effective_tokens_s": "effective_tokens_s"}
0246 |     for label, key in metrics.items():
0247 |         vals = [r.get(key) for r in good]
0248 |         result[label + "_n"] = sum(v is not None for v in vals)
0249 |         result[label + "_p50"] = percentile(vals, .5)
0250 |         result[label + "_p95"] = percentile(vals, .95)
0251 |     # O hash exclui aliases do modelo e chaves: apenas carga de entrada e limite de saída.
0252 |     bodies = []
0253 |     for row in good:
0254 |         args = json.loads(row["request_args"])
0255 |         body = args.get("body", {})
0256 |         bodies.append({k: body.get(k) for k in ("messages", "max_tokens")})
0257 |     result["requests_sha256"] = hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
0258 |     result["p95_exploratory"] = len(good) < 100
0259 |     return result
0260 |
0261 |
0262 | def write_requests_csv(path, report):
0263 |     """Amostras individuais para análise no R/Python, incluindo status de erro."""
0264 |     from reporting import derived
0265 |     fields = ["status", "request_id", "request_start_time", "request_latency", "time_to_first_token_ms",
0266 |               "inter_token_latency_ms", "prompt_tokens", "output_tokens", "decode_tokens_s", "effective_tokens_s",
0267 |               "context_start_tokens", "context_end_tokens", "context_band"]
0268 |     with Path(path).open("w", newline="", encoding="utf-8") as handle:
0269 |         writer = csv.DictWriter(handle, fieldnames=fields)
0270 |         writer.writeheader()
0271 |         for status in ("successful", "errored", "incomplete"):
0272 |             for row in report["benchmarks"][0]["requests"][status]:
0273 |                 if status == "successful":
0274 |                     row = {**row, **derived(row)}
0275 |                 writer.writerow({"status": status, **{key: row.get(key) for key in fields[1:]}})
0276 |
0277 |
0278 | def write_summary(output, rows):
0279 |     from reporting import render
0280 |     write_json(output / "summary.json", rows)
0281 |     if not rows:
0282 |         render(output, rows)
0283 |         return
0284 |     with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
0285 |         writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
0286 |         writer.writeheader()
0287 |         writer.writerows(rows)
0288 |     render(output, rows)
0289 |
0290 |
0291 | def run(args):
0292 |     import fcntl
0293 |     from lifecycle import DEFAULT_PROMPT, Launch, lifecycle_report, timed_request, wait_models
0294 |     if importlib.metadata.version("guidellm") != GUIDELLM_VERSION:
0295 |         raise ValueError(f"Este projeto exige guidellm=={GUIDELLM_VERSION}; reinstale requirements.txt.")
0296 |     cfg = load_config(args.config)
0297 |     validate_run(cfg, args.scenarios, args.smoke)
0298 |     availability = local_model_check(args.local_model_path)
0299 |     availability["kv_bytes_per_token"] = args.kv_bytes_per_token
0300 |     availability["launch_hf_offline"] = bool(args.launch)
0301 |     digest = tokenizer_digest(cfg["tokenizer"])
0302 |     count, repetitions = (3, 1) if args.smoke else (args.requests, args.repetitions)
0303 |     secret = os.environ.get("BENCH_API_KEY", "")
0304 |     prompt = Path(args.first_prompt_file).read_text(encoding="utf-8") if args.first_prompt_file else DEFAULT_PROMPT
0305 |     if not prompt.strip():
0306 |         raise ValueError("O prompt inicial não pode estar vazio.")
0307 |     base = Path(args.results)
0308 |     base.mkdir(parents=True, exist_ok=True)
0309 |     with (base / ".benchmark.lock").open("a") as lock:
0310 |         try:
0311 |             fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
0312 |         except BlockingIOError:
0313 |             raise ValueError("Já existe um benchmark usando esta pasta results. Não execute dois ao mesmo tempo.") from None
0314 |         stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
0315 |         output = base / stamp
0316 |         output.mkdir()
0317 |         manifest = {"project_version": VERSION, "guidellm_version": GUIDELLM_VERSION, "started_utc": stamp,
0318 |                     "config": cfg, "tokenizer": digest, "smoke": args.smoke, "requests": count,
0319 |                     "repetitions": repetitions, "warmup_requests_per_case": args.warmup,
0320 |                     "scenarios": args.scenarios, "seed": args.seed, "profile": "synchronous",
0321 |                     "model_availability": availability,
0322 |                     "guidellm_compat": "0.7.4 bounded drain of real late completion updates (5s); no request retry",
0323 |                     "telemetry": {"gpu_source": "local nvidia-smi", "kv_metrics_requested": args.collect_kv_metrics,
0324 |                                   "sampling": "approximately 1 Hz; not per-token"},
0325 |                     "python": sys.version, "platform": platform.platform(),
0326 |                     "git": capture(["git", "rev-parse", "HEAD"]), "status": "running"}
0327 |         write_json(output / "manifest.json", redact(manifest, secret))
0328 |         write_json(output / "client-packages.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
0329 |         write_json(output / "gpu-before.json", capture(["nvidia-smi"]))
0330 |         monitor, rows = Monitor(output, cfg, secret, args.collect_kv_metrics), []
0331 |         launch, origin, cleanup_error = None, None, None
0332 |         lifecycle = {"mode": "new-process" if args.launch else "existing-server-state-unknown",
0333 |                      "status": "running", "initial_state_note": args.initial_state,
0334 |                      "startup_timeout_s": args.startup_timeout}
0335 |         print(f"Resultados: {output.resolve()}", flush=True)
0336 |         try:
0337 |             monitor.start()
0338 |             if args.launch:
0339 |                 launch = Launch(cfg, args.launch, output)
0340 |                 lifecycle["argv"] = redact(launch.argv, secret)
0341 |                 monitor.phase = "process_startup"
0342 |                 origin = launch.start()
0343 |                 lifecycle["pid"] = launch.process.pid
0344 |             lifecycle_report(output, redact(lifecycle, secret))
0345 |             lifecycle["readiness"] = wait_models(cfg, secret, args.startup_timeout, launch)
0346 |             lifecycle_report(output, redact(lifecycle, secret))
0347 |             monitor.phase = "first_request"
0348 |             print("Primeiro POST: medição da primeira resposta (nenhuma geração prévia enviada pelo cliente).", flush=True)
0349 |             lifecycle["first_request"] = timed_request(cfg, secret, args.timeout, prompt,
0350 |                                                        output / "first-request.json", origin)
0351 |             lifecycle_report(output, redact(lifecycle, secret))
0352 |             from guidellm.benchmark import BenchmarkScenario, benchmark_generative_text
0353 |             from guidellm_compat import completion_drain
0354 |             for rep in range(repetitions):
0355 |                 # Rotação balanceia parcialmente a posição dos cenários entre repetições.
0356 |                 names = args.scenarios[rep % len(args.scenarios):] + args.scenarios[:rep % len(args.scenarios)]
0357 |                 for name in names:
0358 |                     for phase, n in (("warmup", args.warmup), ("measure", count)):
0359 |                         if not n:
0360 |                             continue
0361 |                         seed = args.seed + rep * 100 + list(WORKLOADS).index(name)
0362 |                         if phase == "warmup":
0363 |                             seed += 1_000_000
0364 |                         prefix = f"r{rep+1}-{name}-{phase}"
0365 |                         monitor.phase = prefix
0366 |                         config = scenario_config(cfg, name, n, seed, secret, args.timeout)
0367 |                         write_json(output / f"{prefix}-config.json", redact(config, secret))
0368 |                         print(f"{prefix}: {n} requisições, uma por vez", flush=True)
0369 |                         with completion_drain():
0370 |                             report, _ = asyncio.run(benchmark_generative_text(BenchmarkScenario.model_validate(config)))
0371 |                         raw = redact(report.model_dump(mode="json"), secret)
0372 |                         write_json(output / f"{prefix}.json", raw)
0373 |                         write_requests_csv(output / f"{prefix}-requests.csv", raw)
0374 |                         summary = summarize(raw)
0375 |                         summary["expected"] = n
0376 |                         summary["missing"] = max(0, n - sum(summary[k] for k in ("successful", "errored", "incomplete")))
0377 |                         rows.append({"runtime": cfg["runtime"], "model": cfg["model"],
0378 |                                      "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
0379 |                                      "phase": phase, "scenario": name, "repetition": rep+1, **summary})
0380 |                         write_summary(output, rows)
0381 |                         if summary["successful"] != n or summary["errored"] or summary["incomplete"]:
0382 |                             raise RuntimeError(f"{prefix}: requisições falharam ou execução incompleta. Veja o JSON; não compare como sucesso.")
0383 |             monitor.phase = "warm_reference"
0384 |             lifecycle["warm_reference"] = timed_request(cfg, secret, args.timeout, prompt,
0385 |                                                          output / "warm-reference.json")
0386 |             manifest["status"] = lifecycle["status"] = "complete"
0387 |         except BaseException as exc:
0388 |             manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0389 |             manifest["error"] = redact(str(exc), secret)
0390 |             lifecycle["status"] = manifest["status"]
0391 |             lifecycle["error"] = manifest["error"]
0392 |             raise
0393 |         finally:
0394 |             for key, filename in (("first_request", "first-request.json"), ("warm_reference", "warm-reference.json")):
0395 |                 if (output / filename).exists():
0396 |                     lifecycle[key] = json.loads((output / filename).read_text(encoding="utf-8"))
0397 |             if launch is not None:
0398 |                 monitor.phase = "server_shutdown"
0399 |                 try:
0400 |                     launch.close()
0401 |                     lifecycle["server_cleanup"] = "stopped owned process group only"
0402 |                 except Exception as exc:
0403 |                     cleanup_error = redact(str(exc), secret)
0404 |                     lifecycle["server_cleanup_error"] = cleanup_error
0405 |                     lifecycle["status"] = manifest["status"] = "failed"
0406 |             lifecycle_report(output, redact(lifecycle, secret))
0407 |             monitor.stop()
0408 |             manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
0409 |             write_json(output / "manifest.json", redact(manifest, secret))
0410 |             write_json(output / "gpu-after.json", capture(["nvidia-smi"]))
0411 |             write_summary(output, rows)
0412 |         if cleanup_error:
0413 |             raise RuntimeError(f"Falha ao encerrar processo criado: {cleanup_error}. Confira o PID no lifecycle.json.")
0414 |         print(f"Concluído. Abra {output / 'lifecycle.html'} e {output / 'summary.html'}")
0415 |
0416 |
0417 | def positive(value):
0418 |     number = int(value)
0419 |     if number <= 0:
0420 |         raise argparse.ArgumentTypeError("Use um inteiro positivo.")
0421 |     return number
0422 |
0423 |
0424 | def rebuild_report(args):
0425 |     """Atualiza apenas derivados, preservando relatórios brutos e manifesto original."""
0426 |     output = Path(args.output)
0427 |     rows = json.loads((output / "summary.json").read_text())
0428 |     manifest = json.loads((output / "manifest.json").read_text())
0429 |     for row in rows:
0430 |         prefix = f"r{row['repetition']}-{row['scenario']}-{row['phase']}"
0431 |         raw = json.loads((output / f"{prefix}.json").read_text())
0432 |         row.update(summarize(raw))
0433 |         expected = manifest.get("warmup_requests_per_case") if row["phase"] == "warmup" else manifest.get("requests")
0434 |         row["expected"] = expected
0435 |         row["missing"] = max(0, expected - sum(row[k] for k in ("successful", "errored", "incomplete"))) if expected is not None else None
0436 |         write_requests_csv(output / f"{prefix}-requests.csv", raw)
0437 |     write_summary(output, rows)
0438 |     print(f"Relatório atualizado: {output / 'summary.html'}; dados brutos preservados.")
0439 |
0440 |
0441 | def main():
0442 |     parser = argparse.ArgumentParser(description=__doc__)
0443 |     commands = parser.add_subparsers(dest="command", required=True)
0444 |     report = commands.add_parser("report", help="Regenera derivados de uma execução existente, sem nova inferência.")
0445 |     report.add_argument("--output", required=True)
0446 |     report.set_defaults(func=rebuild_report)
0447 |     prep = commands.add_parser("prepare-tokenizer", help="Baixa apenas tokenizer; fixa revisão e guarda origem.")
0448 |     prep.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
0449 |     prep.add_argument("--revision", default="main")
0450 |     prep.add_argument("--output", default="tokenizer")
0451 |     prep.set_defaults(func=prepare_tokenizer)
0452 |     cmd = commands.add_parser("run", help="Mede primeiro acesso, aquecimento e GuideLLM; lançamento do servidor é opcional.")
0453 |     cmd.add_argument("--config", required=True)
0454 |     cmd.add_argument("--local-model-path", required=True, help="Pesos já no SSD: pasta HF, arquivo GGUF ou blob local do Ollama. Não baixa arquivos.")
0455 |     cmd.add_argument("--collect-kv-metrics", action="store_true", help="Amostra /metrics do vLLM (~1 Hz); ocupação do pool KV, não bytes.")
0456 |     cmd.add_argument("--kv-bytes-per-token", type=positive, help="Opcional: bytes de KV lógico por token, calculados para arquitetura/dtype reais. Estimativa, não VRAM medida.")
0457 |     cmd.add_argument("--input-tokens", nargs="+", type=positive, help="Substitui --scenarios por uma grade de comprimentos sintéticos, ex.: 256 512 1024 2048 3072.")
0458 |     cmd.add_argument("--scenarios", nargs="+", choices=list(WORKLOADS), default=["short", "medium"])
0459 |     cmd.add_argument("--requests", type=positive, default=30)
0460 |     cmd.add_argument("--repetitions", type=positive, default=3)
0461 |     cmd.add_argument("--warmup", type=positive, default=3)
0462 |     cmd.add_argument("--seed", type=positive, default=42)
0463 |     cmd.add_argument("--timeout", type=positive, default=300)
0464 |     cmd.add_argument("--results", default="results")
0465 |     cmd.add_argument("--smoke", action="store_true", help="3 medições e 1 repetição; não vale como resultado final.")
0466 |     cmd.add_argument("--launch", help="Arquivo JSON com argv para iniciar um runtime LOCAL; encerra só esse processo ao final.")
0467 |     cmd.add_argument("--startup-timeout", type=positive, default=1800, help="Limite da espera pela API com --launch, em segundos.")
0468 |     cmd.add_argument("--first-prompt-file", help="Texto UTF-8 para a primeira requisição e referência final; default: pergunta sobre RAM/VRAM.")
0469 |     cmd.add_argument("--initial-state", default="weights local; OS/compilation caches not controlled", help="Descreva SSD e caches existentes; apenas registra, não limpa.")
0470 |     cmd.set_defaults(func=run)
0471 |     args = parser.parse_args()
0472 |     if getattr(args, "input_tokens", None):
0473 |         if len(set(args.input_tokens)) != len(args.input_tokens):
0474 |             parser.error("Não repita comprimentos em --input-tokens.")
0475 |         args.scenarios = []
0476 |         for size in args.input_tokens:
0477 |             name = f"ctx{size}"
0478 |             WORKLOADS[name] = size
0479 |             args.scenarios.append(name)
0480 |     if hasattr(args, "scenarios") and len(set(args.scenarios)) != len(args.scenarios):
0481 |         parser.error("Não repita cenários na lista.")
0482 |     try:
0483 |         args.func(args)
0484 |     except KeyboardInterrupt:
0485 |         print("Interrompido; resultados já concluídos foram preservados.", file=sys.stderr)
0486 |         return 130
0487 |     except Exception as exc:
0488 |         print(f"ERRO: {redact(str(exc), os.environ.get('BENCH_API_KEY', ''))}", file=sys.stderr)
0489 |         return 1
0490 |     return 0
0491 |
0492 |
0493 | if __name__ == "__main__":
0494 |     raise SystemExit(main())
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

SHA-256: `1e3bd5b3440038950df1473732650c0b72ad99e8ca867aba135cc692d010d4b0`.

| Função/classe | Linhas |
|---|---|
| `ratio` | 11–14 |
| `derived` | 17–27 |
| `context_band` | 30–37 |
| `table` | 40–47 |
| `context_summary` | 50–72 |
| `gpu_summary` | 75–97 |
| `render` | 100–153 |
| `fmt` | 41–44 |

```text
0001 | """Métricas derivadas e relatório offline; não confunde contexto com VRAM."""
0002 | import csv
0003 | import html
0004 | import json
0005 | import math
0006 | from collections import defaultdict
0007 | from pathlib import Path
0008 | from statistics import median
0009 |
0010 |
0011 | def ratio(a, b):
0012 |     if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
0013 |         return None
0014 |     return a / b if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0 else None
0015 |
0016 |
0017 | def derived(row):
0018 |     n, p = row.get("output_tokens"), row.get("prompt_tokens")
0019 |     itl = row.get("inter_token_latency_ms")
0020 |     return {
0021 |         "decode_tokens_s": ratio(1000, itl) if n is not None and n > 1 else None,
0022 |         "effective_tokens_s": ratio(n, row.get("request_latency")),
0023 |         "context_start_tokens": p,
0024 |         # Comprimento lógico final; não é o número exato de posições materializadas.
0025 |         "context_end_tokens": p + n if p is not None and n is not None else None,
0026 |         "context_band": context_band(p),
0027 |     }
0028 |
0029 |
0030 | def context_band(p):
0031 |     if p is None:
0032 |         return "não informado"
0033 |     for limit in (512, 1024, 2048, 4096, 8192, 16384):
0034 |         if p < limit:
0035 |             lower = 0 if limit == 512 else limit // 2
0036 |             return f"[{lower}, {limit})"
0037 |     return "[16384, +∞)"
0038 |
0039 |
0040 | def table(rows, columns):
0041 |     def fmt(v):
0042 |         if v is None:
0043 |             return "Não disponível"
0044 |         return f"{v:.3f}" if isinstance(v, float) else str(v)
0045 |     head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
0046 |     body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(r.get(k)))}</td>" for k, _ in columns) + "</tr>" for r in rows)
0047 |     return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
0048 |
0049 |
0050 | def context_summary(output, kv_bytes_per_token=None):
0051 |     groups = defaultdict(list)
0052 |     for file in sorted(Path(output).glob("r*-*-requests.csv")):
0053 |         rep, scenario, phase, _ = file.stem.split("-", 3)
0054 |         with file.open() as handle:
0055 |             for row in csv.DictReader(handle):
0056 |                 if row["status"] != "successful":
0057 |                     continue
0058 |                 for key in ("prompt_tokens", "output_tokens", "inter_token_latency_ms", "request_latency", "time_to_first_token_ms"):
0059 |                     row[key] = float(row[key]) if row.get(key) else None
0060 |                 row.update(derived(row))
0061 |                 groups[(phase, rep, row["context_band"])].append(row)
0062 |     result = []
0063 |     for (phase, rep, band), rows in groups.items():
0064 |         entry = {"phase": phase, "repetition": rep, "context_band": band, "n": len(rows)}
0065 |         for key in ("decode_tokens_s", "effective_tokens_s", "time_to_first_token_ms", "context_start_tokens", "context_end_tokens"):
0066 |             values = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
0067 |             entry[key + "_p50"] = median(values) if values else None
0068 |         result.append(entry)
0069 |         for edge in ("start", "end"):
0070 |             tokens = entry[f"context_{edge}_tokens_p50"]
0071 |             entry[f"kv_{edge}_mib_estimate"] = tokens * kv_bytes_per_token / 1048576 if tokens is not None and kv_bytes_per_token else None
0072 |     return result
0073 |
0074 |
0075 | def gpu_summary(output):
0076 |     groups = defaultdict(list)
0077 |     path = Path(output) / "gpu.csv"
0078 |     if path.exists():
0079 |         with path.open() as handle:
0080 |             for row in csv.DictReader(handle):
0081 |                 groups[(row["phase"], row["index"], row["name"])].append(row)
0082 |     result = []
0083 |     for (phase, index, name), rows in groups.items():
0084 |         entry = {"phase": phase, "gpu": index, "name": name, "samples": len(rows)}
0085 |         for key in ("used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"):
0086 |             values = []
0087 |             for row in rows:
0088 |                 try:
0089 |                     value = float(row[key])
0090 |                     if math.isfinite(value):
0091 |                         values.append(value)
0092 |                 except (ValueError, TypeError, KeyError):
0093 |                     pass
0094 |             entry[key + "_mean"] = sum(values) / len(values) if values else None
0095 |             entry[key + "_max"] = max(values) if values else None
0096 |         result.append(entry)
0097 |     return result
0098 |
0099 |
0100 | def render(output, rows):
0101 |     output = Path(output)
0102 |     lifecycle = json.loads((output / "lifecycle.json").read_text()) if (output / "lifecycle.json").exists() else {}
0103 |     manifest = json.loads((output / "manifest.json").read_text()) if (output / "manifest.json").exists() else {}
0104 |     kv_bytes = manifest.get("model_availability", {}).get("kv_bytes_per_token")
0105 |     context, gpu = context_summary(output, kv_bytes), gpu_summary(output)
0106 |     (output / "context-summary.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
0107 |     (output / "gpu-summary.json").write_text(json.dumps(gpu, ensure_ascii=False, indent=2) + "\n")
0108 |     startup = lifecycle.get("readiness", {}).get("process_to_api_observed_s")
0109 |     initial = []
0110 |     for key, label in (("first_request", "Primeira resposta"), ("warm_reference", "Referência final")):
0111 |         req = lifecycle.get(key, {})
0112 |         usage = req.get("stream_usage") or {}
0113 |         d = derived({"output_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
0114 |                      "inter_token_latency_ms": req.get("mean_itl_ms"), "request_latency": req.get("e2e_s")})
0115 |         initial.append({"phase": label, "ttft": req.get("ttft_ms"), "e2e": req.get("e2e_s"), **d})
0116 |     metrics = table(rows, [("phase", "Fase"), ("scenario", "Cenário"), ("repetition", "Repetição"),
0117 |         ("expected", "Previstas"), ("successful", "Sucessos"), ("errored", "Erros"), ("incomplete", "Incompletas"), ("missing", "Ausentes do relatório bruto"),
0118 |         ("ttft_ms_p50", "TTFT p50 (ms)"), ("ttft_ms_p95", "TTFT p95 (ms)"),
0119 |         ("decode_tokens_s_p50", "Geração p50 (tokens/s)"), ("effective_tokens_s_p50", "Efetiva p50 (tokens/s)"),
0120 |         ("e2e_s_p50", "Total p50 (s)"), ("prompt_tokens_p50", "Entrada p50 (tokens)"), ("output_tokens_p50", "Saída p50 (tokens)")])
0121 |     by_context = table(context, [("phase", "Fase"), ("repetition", "Repetição"), ("context_band", "Faixa de entrada (tokens)"),
0122 |         ("n", "n"), ("context_start_tokens_p50", "Contexto inicial p50"), ("context_end_tokens_p50", "Contexto final p50"),
0123 |         ("kv_start_mib_estimate", "KV inicial estimado (MiB)"), ("kv_end_mib_estimate", "KV final estimado (MiB)"),
0124 |         ("time_to_first_token_ms_p50", "TTFT p50 (ms)"), ("decode_tokens_s_p50", "Geração p50 (tokens/s)"),
0125 |         ("effective_tokens_s_p50", "Efetiva p50 (tokens/s)")])
0126 |     hardware = table(gpu, [("phase", "Fase"), ("gpu", "GPU"), ("name", "Nome"), ("samples", "Amostras"),
0127 |         ("used_mib_max", "Memória máx. (MiB)"), ("total_mib_max", "Memória total (MiB)"),
0128 |         ("gpu_util_pct_mean", "Utilização média (%)"), ("gpu_util_pct_max", "Utilização máx. (%)"),
0129 |         ("temperature_c_max", "Temperatura máx. (°C)"), ("power_w_mean", "Potência média (W)"), ("power_w_max", "Potência máx. (W)")]) if gpu else "<p>Não disponível: nenhuma amostra NVIDIA válida. Em Apple/Metal este coletor não mede GPU; isso não significa utilização zero.</p>"
0130 |     kvfile = output / "kv-cache.csv"
0131 |     kvrows = []
0132 |     if kvfile.exists():
0133 |         with kvfile.open() as handle:
0134 |             groups = defaultdict(list)
0135 |             for r in csv.DictReader(handle):
0136 |                 groups[(r["phase"], r["series"])].append(float(r["fraction"]) * 100)
0137 |             kvrows = [{"phase": p, "series": s, "n": len(v), "mean": sum(v)/len(v), "max": max(v)} for (p, s), v in groups.items()]
0138 |     kv = table(kvrows, [("phase", "Fase"), ("series", "Série do servidor"), ("n", "Amostras"), ("mean", "Ocupação média (%)"), ("max", "Ocupação máx. (%)")]) if kvrows else "<p>Ocupação real de KV não disponível nesta execução. Não foi estimada a partir da VRAM.</p>"
0139 |     first = table(initial, [("phase", "Fase"), ("ttft", "TTFT (ms)"), ("decode_tokens_s", "Geração (tokens/s)"), ("effective_tokens_s", "Efetiva (tokens/s)"), ("e2e", "Total (s)")])
0140 |     policy = manifest.get("model_availability", {}).get("policy", "Execução anterior: veja o estado inicial; ausência de download não verificada por esta versão.")
0141 |     failure = f'<p class="note">Execução não concluída: {html.escape(str(manifest["error"]))}. Dados parciais não constituem uma bateria válida.</p>' if manifest.get("error") else ""
0142 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark · latência, geração e GPU</title>
0143 | <style>body{{font:16px/1.7 system-ui;margin:32px;background:#f6f3ec;color:#193835}}main{{max-width:1400px;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border:1px solid #ccd6cc;text-align:left}}th{{background:#e0e9df}}.scroll{{overflow:auto}}h2{{margin-top:38px}}a{{color:#136d58}}.note{{padding:16px;background:#fff0de;border-left:4px solid #b54e27}}</style><main>
0144 | <h1>Um usuário · latência, geração e GPU</h1><p>Status: <strong>{html.escape(manifest.get('status', 'desconhecido'))}</strong>. {html.escape(policy)}</p>{failure}
0145 | <p class="note">TTFT = espera pelo primeiro token/conteúdo observado. Geração = (tokens de saída − 1)/(tempo entre primeiro e último token). Efetiva = tokens de saída/tempo total da requisição, incluindo TTFT. São taxas por requisição, não throughput agregado de usuários.</p>
0146 | <h2>1. Inicialização e primeira resposta</h2><p>Processo → API disponível: {html.escape(str(startup)) if startup is not None else 'não medido'} s. <a href="lifecycle.html">Ver ciclo de vida completo</a>.</p>{first}
0147 | <h2>2. Aquecimento e operação posterior</h2><p>Percentis entre requisições bem-sucedidas. Warmup e measure separados; p95 com menos de 100 sucessos é exploratório. Geração indisponível com menos de dois tokens ou intervalo não positivo.</p>{metrics}
0148 | <h2>3. Tokens/s por faixa de contexto — proxy da carga de KV</h2><p>Faixa definida pela entrada real, incluindo template, antes do decode. O contexto cresce durante a saída; mostramos também seu comprimento lógico final. Esta é uma comparação de velocidades médias de respostas iniciadas em cada faixa, não uma medição token a token dentro de faixas de ocupação física do cache.</p>{by_context}
0149 | <p>Para atenção completa, mantendo modelo, dtype de KV e uma sequência: KV lógico ≈ 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento × tokens. Pesos 4/8 bits não determinam o dtype do KV. Blocos, reserva, prefix caching e sliding window impedem tratar essa fórmula como medição de VRAM. MiB estimados só aparecem com --kv-bytes-per-token informado e verificado pelo operador; caso contrário, ficam indisponíveis.</p>
0150 | <h2>4. GPU por fase</h2><p>Host do cliente; execute no mesmo pod do servidor. Aproximadamente 1 amostra/s, todas as GPUs visíveis, sem atribuição por processo. Máximos amostrados podem perder picos. N/A é ausência de dado, não zero.</p>{hardware}
0151 | <h2>5. Ocupação real do pool KV — vLLM</h2><p>Coleta opcional de /metrics via --collect-kv-metrics. Percentual de blocos ocupados do pool, não percentual de VRAM nem bytes. Séries/engines separados. Amostragem e atualização do servidor podem perder transientes; não sincronizada por token.</p>{kv}
0152 | <p><a href="summary.json">Resumo JSON</a> · <a href="context-summary.json">Faixas JSON</a> · <a href="gpu-summary.json">GPU JSON</a> · <a href="manifest.json">Manifesto</a></p></main></html>'''
0153 |     (output / "summary.html").write_text(page, encoding="utf-8")
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
