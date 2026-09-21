# Código completo: referência linha a linha

Leia junto com [a explicação detalhada](codigo-explicado.md). Gerado dos arquivos reais; não é um segundo programa. Os números não pertencem ao Python.

## bench.py

SHA-256: `17db50411629499e6b377fa891cc86c0e9274a2545e5125065219f7a23f5574f`.

| Função/classe | Linhas |
|---|---|
| `local_model_check` | 30–50 |
| `write_json` | 53–54 |
| `redact` | 57–65 |
| `capture` | 68–73 |
| `load_config` | 76–92 |
| `validate_run` | 95–108 |
| `tokenizer_digest` | 111–120 |
| `prepare_tokenizer` | 123–135 |
| `scenario_config` | 138–157 |
| `Monitor` | 160–308 |
| `percentile` | 311–317 |
| `summarize` | 320–372 |
| `write_requests_csv` | 375–388 |
| `write_summary` | 391–401 |
| `run` | 404–528 |
| `positive` | 531–535 |
| `rebuild_report` | 538–552 |
| `main` | 555–604 |
| `__init__` | 162–169 |
| `set_phase` | 171–177 |
| `start` | 179–188 |
| `kv_loop` | 190–213 |
| `loop` | 215–255 |
| `stop` | 257–266 |
| `write_telemetry_summary` | 268–308 |
| `read_rows` | 270–275 |

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
0019 | import time
0020 | from datetime import datetime, timezone
0021 | from urllib.parse import urlsplit
0022 |
0023 | ROOT = Path(__file__).resolve().parent
0024 | VERSION = "0.4.0"
0025 | GUIDELLM_VERSION = "0.7.4"
0026 | WORKLOADS = {"short": 256, "medium": 2048, "long": 8192}
0027 | OUTPUT_TOKENS = 128
0028 |
0029 |
0030 | def local_model_check(value):
0031 |     """Checa presença, não lê pesos nem aquece o page cache deliberadamente."""
0032 |     path = Path(value).expanduser().resolve()
0033 |     if path.is_file():
0034 |         if path.suffix not in {".safetensors", ".gguf", ".bin", ".pt", ".pth"} and not path.name.startswith("sha256-"):
0035 |             raise ValueError("Informe um arquivo de pesos (não configuração/texto) ou blob sha256- do Ollama.")
0036 |         files = [path]
0037 |     elif path.is_dir():
0038 |         files = [f for f in path.rglob("*") if f.is_file() and f.suffix in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}]
0039 |         for index in path.glob("*.index.json"):
0040 |             data = json.loads(index.read_text())
0041 |             missing = [name for name in set(data.get("weight_map", {}).values()) if not (path / name).is_file()]
0042 |             if missing:
0043 |                 raise ValueError(f"Shards ausentes: {missing}")
0044 |     else:
0045 |         files = []
0046 |     if not files or any(f.stat().st_size == 0 for f in files):
0047 |         raise ValueError("Pesos locais ausentes/vazios. Prepare o modelo antes do benchmark; downloads não são permitidos na medição.")
0048 |     return {"policy": "Pesos locais obrigatórios; download excluído do protocolo.", "path": str(path),
0049 |             "files": len(files), "bytes": sum(f.stat().st_size for f in files),
0050 |             "validation": "presença/tamanho e shards declarados; não verifica conteúdo, SSD físico ou vínculo com API"}
0051 |
0052 |
0053 | def write_json(path, data):
0054 |     Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0055 |
0056 |
0057 | def redact(value, secret):
0058 |     if isinstance(value, dict):
0059 |         return {k: ("[REDACTED]" if k.lower() in {"api_key", "authorization"} else redact(v, secret))
0060 |                 for k, v in value.items()}
0061 |     if isinstance(value, list):
0062 |         return [redact(v, secret) for v in value]
0063 |     if isinstance(value, str) and secret:
0064 |         return value.replace(secret, "[REDACTED]")
0065 |     return value
0066 |
0067 |
0068 | def capture(command):
0069 |     try:
0070 |         proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
0071 |         return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
0072 |     except (OSError, subprocess.TimeoutExpired) as exc:
0073 |         return {"unavailable": str(exc)}
0074 |
0075 |
0076 | def load_config(path):
0077 |     cfg = json.loads(Path(path).read_text(encoding="utf-8"))
0078 |     required = {"runtime", "base_url", "model", "tokenizer", "context_window", "cache_policy",
0079 |                 "runtime_version", "model_artifact", "server_command", "notes"}
0080 |     if set(cfg) != required:
0081 |         raise ValueError(f"Campos da configuração devem ser exatamente: {sorted(required)}")
0082 |     if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in required - {"context_window"}):
0083 |         raise ValueError("Os campos textuais da configuração devem estar preenchidos.")
0084 |     if type(cfg["context_window"]) is not int or cfg["context_window"] < 512:
0085 |         raise ValueError("context_window deve ser um inteiro >= 512.")
0086 |     url = urlsplit(cfg["base_url"])
0087 |     if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
0088 |         raise ValueError("URL inválida: use http(s), sem credenciais, query ou fragmento.")
0089 |     if url.path not in {"", "/", "/v1", "/v1/"}:
0090 |         raise ValueError("Use a raiz do servidor (ex.: http://127.0.0.1:8000), sem endpoint.")
0091 |     cfg["base_url"] = f"{url.scheme}://{url.netloc}"
0092 |     return cfg
0093 |
0094 |
0095 | def validate_run(cfg, scenarios, smoke):
0096 |     for scenario in scenarios:
0097 |         # Margem para o template; o servidor continua sendo a autoridade final.
0098 |         needed = WORKLOADS[scenario] + OUTPUT_TOKENS + 256
0099 |         if cfg["context_window"] < needed:
0100 |             raise ValueError(f"{scenario} precisa de contexto declarado >= {needed}; "
0101 |                              "altere o servidor e depois o JSON, ou retire esse cenário.")
0102 |     if not smoke:
0103 |         if any("PREENCHER" in cfg[k] or "SUBSTITUA" in cfg[k]
0104 |                for k in ("model", "runtime_version", "model_artifact", "server_command")):
0105 |             raise ValueError("Preencha modelo, versão, artefato e comando antes da medição formal; --smoke permite rascunhos.")
0106 |         if cfg["cache_policy"] == "runtime-default-unverified":
0107 |             raise ValueError("Registre cache_policy após verificar o servidor. Ex.: disabled-confirmed ou enabled-recorded. "
0108 |                              "O script NÃO altera nem comprova a política de cache.")
0109 |
0110 |
0111 | def tokenizer_digest(path):
0112 |     path = Path(path)
0113 |     if not path.is_dir() or not (path / "tokenizer_config.json").exists():
0114 |         raise ValueError("Tokenizer local ausente. Execute: python bench.py prepare-tokenizer")
0115 |     hashes = {}
0116 |     for file in sorted(path.rglob("*")):
0117 |         if file.is_file() and not file.name.startswith("."):
0118 |             hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
0119 |     digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
0120 |     return {"sha256": digest, "files": hashes}
0121 |
0122 |
0123 | def prepare_tokenizer(args):
0124 |     from huggingface_hub import HfApi
0125 |     from transformers import AutoTokenizer
0126 |     path = Path(args.output)
0127 |     if path.exists():
0128 |         raise ValueError(f"{path} já existe; não será sobrescrito. Use outro --output.")
0129 |     revision = HfApi().model_info(args.model, revision=args.revision).sha
0130 |     tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
0131 |     path.mkdir(parents=True)
0132 |     tokenizer.save_pretrained(path)
0133 |     write_json(path / "source.json", {"model": args.model, "revision": revision})
0134 |     print(f"Tokenizer salvo em {path}; revisão {revision}. Não foram baixados pesos.")
0135 |     print("Compartilhe esta pasta com o grupo para usar os mesmos arquivos.")
0136 |
0137 |
0138 | def scenario_config(cfg, name, count, seed, secret, timeout):
0139 |     return {
0140 |         "spec": {
0141 |             "backend": {"kind": "openai_http", "target": cfg["base_url"], "model": cfg["model"],
0142 |                         "request_format": "/v1/chat/completions", "stream": True,
0143 |                         "validate_backend": False, "verify": True, "follow_redirects": False,
0144 |                         "timeout": timeout, "api_key": secret or None,
0145 |                         "extras": {"body": {"temperature": 0, "top_p": 1}}},
0146 |             "profile": {"kind": "synchronous", "warmup": 0, "cooldown": 0},
0147 |             "constraints": [{"kind": "max_requests", "count": count}, {"kind": "max_errors", "count": 1}],
0148 |             "tokenizer": {"kind": "huggingface_auto", "model": str(Path(cfg["tokenizer"]).resolve()),
0149 |                           "load_kwargs": {"local_files_only": True, "trust_remote_code": False}},
0150 |             "data": [{"kind": "synthetic_text", "prompt_tokens": WORKLOADS[name],
0151 |                       "output_tokens": OUTPUT_TOKENS}],
0152 |             "data_loader": {"kind": "pytorch", "samples": count, "num_workers": 0, "shuffle": False},
0153 |             "seed": {"kind": "static", "value": seed},
0154 |             "metrics": {"kind": "generative", "sample_size": None, "prefer_response_metrics": True},
0155 |             "outputs": [],
0156 |         }
0157 |     }
0158 |
0159 |
0160 | class Monitor:
0161 |     """Amostragem contínua e marcação de fases; não é um profiler PCIe."""
0162 |     def __init__(self, output, cfg=None, secret="", collect_kv=False):
0163 |         self.output = Path(output)
0164 |         self.stop_event = threading.Event()
0165 |         self.thread = None
0166 |         self.phase = "setup"
0167 |         self.cfg, self.secret, self.collect_kv = cfg, secret, collect_kv
0168 |         self.kv_thread = None
0169 |         self.started_monotonic = None
0170 |
0171 |     def set_phase(self, phase, event=None):
0172 |         self.phase = phase
0173 |         print(f"[telemetria] fase={phase} evento={event or 'phase_change'}", flush=True)
0174 |         if hasattr(self, "events_handle"):
0175 |             writer = csv.writer(self.events_handle)
0176 |             writer.writerow([datetime.now(timezone.utc).isoformat(), time.monotonic(), phase, event or "phase_change"])
0177 |             self.events_handle.flush()
0178 |
0179 |     def start(self):
0180 |         self.started_monotonic = time.monotonic()
0181 |         self.events_handle = (self.output / "events.csv").open("w", newline="", encoding="utf-8")
0182 |         csv.writer(self.events_handle).writerow(["utc", "monotonic_s", "phase", "event"])
0183 |         self.set_phase(self.phase, "monitor_started")
0184 |         self.thread = threading.Thread(target=self.loop, daemon=True)
0185 |         self.thread.start()
0186 |         if self.collect_kv:
0187 |             self.kv_thread = threading.Thread(target=self.kv_loop, daemon=True)
0188 |             self.kv_thread.start()
0189 |
0190 |     def kv_loop(self):
0191 |         import httpx
0192 |         headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
0193 |         pattern = re.compile(r'^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc))(\{[^}]*\})?\s+([0-9.eE+\-]+)(?:\s|$)')
0194 |         with (self.output / "kv-cache.csv").open("w", newline="") as handle, httpx.Client(timeout=1, headers=headers, follow_redirects=False) as client:
0195 |             writer = csv.writer(handle)
0196 |             writer.writerow(["utc", "phase", "series", "fraction"])
0197 |             while not self.stop_event.is_set():
0198 |                 phase = self.phase
0199 |                 try:
0200 |                     response = client.get(self.cfg["base_url"] + "/metrics")
0201 |                     response.raise_for_status()
0202 |                     matches = [m for line in response.text.splitlines() if (m := pattern.match(line))]
0203 |                     modern = any(m[1] == "vllm:kv_cache_usage_perc" for m in matches)
0204 |                     for m in matches:
0205 |                         if modern and m[1] != "vllm:kv_cache_usage_perc":
0206 |                             continue
0207 |                         value = float(m[3])
0208 |                         if math.isfinite(value) and 0 <= value <= 1:
0209 |                             writer.writerow([datetime.now(timezone.utc).isoformat(), phase, redact(m[1] + (m[2] or ""), self.secret), value])
0210 |                     handle.flush()
0211 |                 except (httpx.HTTPError, ValueError):
0212 |                     pass  # Serveur inicializando/endpoint ausente: nunca inventar zeros.
0213 |                 self.stop_event.wait(1)
0214 |
0215 |     def loop(self):
0216 |         try:
0217 |             import psutil
0218 |         except ImportError:
0219 |             psutil = None
0220 |         with (self.output / "gpu.csv").open("w", newline="", encoding="utf-8") as handle, \
0221 |              (self.output / "system.csv").open("w", newline="", encoding="utf-8") as system_handle:
0222 |             writer = csv.writer(handle)
0223 |             writer.writerow(["utc", "phase", "index", "name", "used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"])
0224 |             system_writer = csv.writer(system_handle)
0225 |             system_writer.writerow(["utc", "phase", "cpu_util_pct", "ram_used_mib", "ram_available_mib", "ram_total_mib",
0226 |                                     "load1", "root_disk_used_mib", "root_disk_free_mib", "disk_read_bytes", "disk_write_bytes"])
0227 |             if psutil:
0228 |                 psutil.cpu_percent(interval=None)
0229 |             sample_number = 0
0230 |             while not self.stop_event.is_set():
0231 |                 sample_number += 1
0232 |                 phase = self.phase
0233 |                 utc = datetime.now(timezone.utc).isoformat()
0234 |                 result = capture(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
0235 |                                   "--format=csv,noheader,nounits"])
0236 |                 if result.get("returncode") != 0:
0237 |                     write_json(self.output / "gpu-unavailable.json", result)
0238 |                 else:
0239 |                     gpu_rows = list(csv.reader(result["stdout"].splitlines(), skipinitialspace=True))
0240 |                     for row in gpu_rows:
0241 |                         writer.writerow([utc, phase, *row])
0242 |                     if sample_number == 1 or sample_number % 10 == 0:
0243 |                         compact = "; ".join(f"GPU{row[0]} VRAM={row[2]}/{row[3]} MiB uso={row[4]}%" for row in gpu_rows)
0244 |                         print(f"[telemetria] fase={phase} {compact or 'GPU sem amostra'}", flush=True)
0245 |                 handle.flush()
0246 |                 if psutil:
0247 |                     vm = psutil.virtual_memory()
0248 |                     du = psutil.disk_usage(str(self.output.anchor or "/"))
0249 |                     io = psutil.disk_io_counters()
0250 |                     system_writer.writerow([utc, phase, psutil.cpu_percent(interval=None), vm.used / 1048576,
0251 |                                              vm.available / 1048576, vm.total / 1048576, os.getloadavg()[0],
0252 |                                              du.used / 1048576, du.free / 1048576,
0253 |                                              getattr(io, "read_bytes", None), getattr(io, "write_bytes", None)])
0254 |                     system_handle.flush()
0255 |                 self.stop_event.wait(1)
0256 |
0257 |     def stop(self):
0258 |         self.stop_event.set()
0259 |         if self.thread:
0260 |             self.thread.join(timeout=17)
0261 |         if self.kv_thread:
0262 |             self.kv_thread.join(timeout=3)
0263 |         self.set_phase("stopped", "monitor_stopped")
0264 |         if hasattr(self, "events_handle"):
0265 |             self.events_handle.close()
0266 |         self.write_telemetry_summary()
0267 |
0268 |     def write_telemetry_summary(self):
0269 |         """Agrega telemetria por fase para relacionar picos com eventos do benchmark."""
0270 |         def read_rows(name):
0271 |             path = self.output / name
0272 |             if not path.exists():
0273 |                 return []
0274 |             with path.open() as handle:
0275 |                 return list(csv.DictReader(handle))
0276 |         sources = {"gpu": read_rows("gpu.csv"), "system": read_rows("system.csv"), "kv": read_rows("kv-cache.csv")}
0277 |         phases = sorted({r.get("phase") for rows in sources.values() for r in rows if r.get("phase")})
0278 |         output = {"definition": "Amostras observadas por fase; não são bytes nem tempos de transferência PCIe.", "phases": {}}
0279 |         for phase in phases:
0280 |             entry = {"gpu_samples": 0, "system_samples": 0, "kv_samples": 0}
0281 |             grows = [r for r in sources["gpu"] if r.get("phase") == phase]
0282 |             for key in ("used_mib", "gpu_util_pct", "temperature_c", "power_w"):
0283 |                 vals = []
0284 |                 for r in grows:
0285 |                     try: vals.append(float(r[key]))
0286 |                     except (ValueError, TypeError, KeyError): pass
0287 |                 entry[f"gpu_{key}_mean"] = sum(vals) / len(vals) if vals else None
0288 |                 entry[f"gpu_{key}_max"] = max(vals) if vals else None
0289 |             entry["gpu_samples"] = len(grows)
0290 |             srows = [r for r in sources["system"] if r.get("phase") == phase]
0291 |             for key in ("cpu_util_pct", "ram_used_mib", "ram_available_mib", "root_disk_used_mib", "root_disk_free_mib"):
0292 |                 vals = []
0293 |                 for r in srows:
0294 |                     try: vals.append(float(r[key]))
0295 |                     except (ValueError, TypeError, KeyError): pass
0296 |                 entry[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
0297 |                 entry[f"{key}_max"] = max(vals) if vals else None
0298 |             entry["system_samples"] = len(srows)
0299 |             krows = [r for r in sources["kv"] if r.get("phase") == phase]
0300 |             vals = []
0301 |             for r in krows:
0302 |                 try: vals.append(float(r["fraction"]) * 100)
0303 |                 except (ValueError, TypeError, KeyError): pass
0304 |             entry["kv_occupancy_pct_mean"] = sum(vals) / len(vals) if vals else None
0305 |             entry["kv_occupancy_pct_max"] = max(vals) if vals else None
0306 |             entry["kv_samples"] = len(vals)
0307 |             output["phases"][phase] = entry
0308 |         write_json(self.output / "telemetry-summary.json", output)
0309 |
0310 |
0311 | def percentile(values, q):
0312 |     values = sorted(v for v in values if v is not None and math.isfinite(v))
0313 |     if not values:
0314 |         return None
0315 |     pos = (len(values) - 1) * q
0316 |     lo, hi = math.floor(pos), math.ceil(pos)
0317 |     return values[lo] + (values[hi] - values[lo]) * (pos - lo)
0318 |
0319 |
0320 | def summarize(report):
0321 |     """Métricas por requisição, sem misturar warmup ou erros com sucessos."""
0322 |     benchmarks = report["benchmarks"]
0323 |     if len(benchmarks) != 1:
0324 |         raise ValueError("Esperado exatamente um benchmark sequencial.")
0325 |     requests = benchmarks[0]["requests"]
0326 |     from reporting import derived
0327 |     good = [{**r, **derived(r)} for r in requests["successful"]]
0328 |     result = {
0329 |         "successful_request_count": len(good),
0330 |         "errored_request_count": len(requests["errored"]),
0331 |         "incomplete_request_count": len(requests["incomplete"]),
0332 |     }
0333 |     # Os nomes são deliberadamente longos: summary.json é um artefato de
0334 |     # análise, e não uma API em que economizar alguns bytes melhora algo.
0335 |     metrics = {
0336 |         "request_first_token_latency_milliseconds": "time_to_first_token_ms",
0337 |         "request_latency_seconds": "request_latency",
0338 |         "within_response_next_token_latency_milliseconds": "inter_token_latency_ms",
0339 |         "output_completion_token_count": "output_tokens",
0340 |         "input_prompt_token_count": "prompt_tokens",
0341 |         "decode_generation_tokens_per_second": "decode_tokens_s",
0342 |         "effective_output_tokens_per_second": "effective_tokens_s",
0343 |     }
0344 |     for label, key in metrics.items():
0345 |         vals = [r.get(key) for r in good]
0346 |         result[label + "_sample_count"] = sum(v is not None for v in vals)
0347 |         result[label + "_p50"] = percentile(vals, .5)
0348 |         result[label + "_p95"] = percentile(vals, .95)
0349 |         result[label + "_p99"] = percentile(vals, .99)
0350 |     # Nomes canônicos para a comparação entre runtimes. Os campos históricos
0351 |     # acima permanecem para compatibilidade com relatórios já gerados.
0352 |     ttft = [r.get("time_to_first_token_ms") for r in good]
0353 |     tokens_s = [r.get("decode_tokens_s") for r in good]
0354 |     result["time_to_first_token_milliseconds_sample_count"] = sum(v is not None for v in ttft)
0355 |     result["tokens_per_second_sample_count"] = sum(v is not None for v in tokens_s)
0356 |     for suffix, q in (("p50", .50), ("p95", .95), ("p99", .99)):
0357 |         result[f"time_to_first_token_milliseconds_{suffix}"] = percentile(ttft, q)
0358 |         result[f"tokens_per_second_{suffix}"] = percentile(tokens_s, q)
0359 |     result["metric_definitions"] = {
0360 |         "Time To First Token": "milissegundos entre o envio da requisição e o primeiro token/conteúdo observado; há um valor por requisição.",
0361 |         "Tokens/s": "tokens de saída por segundo durante o decode, calculado por requisição a partir do intervalo entre tokens; não inclui TTFT.",
0362 |     }
0363 |     # O hash exclui aliases do modelo e chaves: apenas carga de entrada e limite de saída.
0364 |     bodies = []
0365 |     for row in good:
0366 |         args = json.loads(row["request_args"])
0367 |         body = args.get("body", {})
0368 |         bodies.append({k: body.get(k) for k in ("messages", "max_tokens")})
0369 |     result["requests_sha256"] = hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
0370 |     result["percentiles_are_exploratory"] = len(good) < 100
0371 |     result["percentile_definition"] = "Empirical linear interpolation over successful requests in this phase/scenario/repetition block."
0372 |     return result
0373 |
0374 |
0375 | def write_requests_csv(path, report):
0376 |     """Amostras individuais para análise no R/Python, incluindo status de erro."""
0377 |     from reporting import derived
0378 |     fields = ["status", "request_id", "request_start_time", "request_latency", "time_to_first_token_ms",
0379 |               "inter_token_latency_ms", "prompt_tokens", "output_tokens", "decode_tokens_s", "effective_tokens_s",
0380 |               "context_start_tokens", "context_end_tokens", "context_band"]
0381 |     with Path(path).open("w", newline="", encoding="utf-8") as handle:
0382 |         writer = csv.DictWriter(handle, fieldnames=fields)
0383 |         writer.writeheader()
0384 |         for status in ("successful", "errored", "incomplete"):
0385 |             for row in report["benchmarks"][0]["requests"][status]:
0386 |                 if status == "successful":
0387 |                     row = {**row, **derived(row)}
0388 |                 writer.writerow({"status": status, **{key: row.get(key) for key in fields[1:]}})
0389 |
0390 |
0391 | def write_summary(output, rows):
0392 |     from reporting import render
0393 |     write_json(output / "summary.json", rows)
0394 |     if not rows:
0395 |         render(output, rows)
0396 |         return
0397 |     with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
0398 |         writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
0399 |         writer.writeheader()
0400 |         writer.writerows(rows)
0401 |     render(output, rows)
0402 |
0403 |
0404 | def run(args):
0405 |     import fcntl
0406 |     from lifecycle import DEFAULT_PROMPT, Launch, lifecycle_report, timed_request, wait_models
0407 |     if importlib.metadata.version("guidellm") != GUIDELLM_VERSION:
0408 |         raise ValueError(f"Este projeto exige guidellm=={GUIDELLM_VERSION}; reinstale requirements.txt.")
0409 |     cfg = load_config(args.config)
0410 |     validate_run(cfg, args.scenarios, args.smoke)
0411 |     availability = local_model_check(args.local_model_path)
0412 |     availability["kv_bytes_per_token"] = args.kv_bytes_per_token
0413 |     availability["launch_hf_offline"] = bool(args.launch)
0414 |     digest = tokenizer_digest(cfg["tokenizer"])
0415 |     count, repetitions = (3, 1) if args.smoke else (args.requests, args.repetitions)
0416 |     secret = os.environ.get("BENCH_API_KEY", "")
0417 |     prompt = Path(args.first_prompt_file).read_text(encoding="utf-8") if args.first_prompt_file else DEFAULT_PROMPT
0418 |     if not prompt.strip():
0419 |         raise ValueError("O prompt inicial não pode estar vazio.")
0420 |     base = Path(args.results)
0421 |     base.mkdir(parents=True, exist_ok=True)
0422 |     with (base / ".benchmark.lock").open("a") as lock:
0423 |         try:
0424 |             fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
0425 |         except BlockingIOError:
0426 |             raise ValueError("Já existe um benchmark usando esta pasta results. Não execute dois ao mesmo tempo.") from None
0427 |         stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
0428 |         output = base / stamp
0429 |         output.mkdir()
0430 |         manifest = {"project_version": VERSION, "guidellm_version": GUIDELLM_VERSION, "started_utc": stamp,
0431 |                     "config": cfg, "tokenizer": digest, "smoke": args.smoke, "requests": count,
0432 |                     "repetitions": repetitions, "warmup_requests_per_case": args.warmup,
0433 |                     "scenarios": args.scenarios, "seed": args.seed, "profile": "synchronous",
0434 |                     "model_availability": availability,
0435 |                     "guidellm_compat": "0.7.4 bounded drain of real late completion updates (5s); no request retry",
0436 |                     "telemetry": {"gpu_source": "local nvidia-smi", "system_source": "psutil host CPU/RAM/disk counters",
0437 |                                   "event_source": "events.csv phase markers", "kv_metrics_requested": args.collect_kv_metrics,
0438 |                                   "sampling": "approximately 1 Hz; not per-token; PCIe copy time is not directly measured"},
0439 |                     "python": sys.version, "platform": platform.platform(),
0440 |                     "git": capture(["git", "rev-parse", "HEAD"]), "status": "running"}
0441 |         write_json(output / "manifest.json", redact(manifest, secret))
0442 |         write_json(output / "client-packages.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
0443 |         write_json(output / "gpu-before.json", capture(["nvidia-smi"]))
0444 |         monitor, rows = Monitor(output, cfg, secret, args.collect_kv_metrics), []
0445 |         launch, origin, cleanup_error = None, None, None
0446 |         lifecycle = {"mode": "new-process" if args.launch else "existing-server-state-unknown",
0447 |                      "status": "running", "initial_state_note": args.initial_state,
0448 |                      "startup_timeout_s": args.startup_timeout}
0449 |         print(f"Resultados: {output.resolve()}", flush=True)
0450 |         try:
0451 |             monitor.start()
0452 |             if args.launch:
0453 |                 launch = Launch(cfg, args.launch, output)
0454 |                 lifecycle["argv"] = redact(launch.argv, secret)
0455 |                 monitor.set_phase("process_startup")
0456 |                 origin = launch.start()
0457 |                 lifecycle["pid"] = launch.process.pid
0458 |             lifecycle_report(output, redact(lifecycle, secret))
0459 |             lifecycle["readiness"] = wait_models(cfg, secret, args.startup_timeout, launch)
0460 |             lifecycle_report(output, redact(lifecycle, secret))
0461 |             monitor.set_phase("first_request")
0462 |             print("Primeiro POST: medição da primeira resposta (nenhuma geração prévia enviada pelo cliente).", flush=True)
0463 |             lifecycle["first_request"] = timed_request(cfg, secret, args.timeout, prompt,
0464 |                                                        output / "first-request.json", origin)
0465 |             lifecycle_report(output, redact(lifecycle, secret))
0466 |             from guidellm.benchmark import BenchmarkScenario, benchmark_generative_text
0467 |             from guidellm_compat import completion_drain
0468 |             for rep in range(repetitions):
0469 |                 # Rotação balanceia parcialmente a posição dos cenários entre repetições.
0470 |                 names = args.scenarios[rep % len(args.scenarios):] + args.scenarios[:rep % len(args.scenarios)]
0471 |                 for name in names:
0472 |                     for phase, n in (("warmup", args.warmup), ("measure", count)):
0473 |                         if not n:
0474 |                             continue
0475 |                         seed = args.seed + rep * 100 + list(WORKLOADS).index(name)
0476 |                         if phase == "warmup":
0477 |                             seed += 1_000_000
0478 |                         prefix = f"r{rep+1}-{name}-{phase}"
0479 |                         monitor.set_phase(prefix)
0480 |                         config = scenario_config(cfg, name, n, seed, secret, args.timeout)
0481 |                         write_json(output / f"{prefix}-config.json", redact(config, secret))
0482 |                         print(f"{prefix}: {n} requisições, uma por vez", flush=True)
0483 |                         with completion_drain():
0484 |                             report, _ = asyncio.run(benchmark_generative_text(BenchmarkScenario.model_validate(config)))
0485 |                         raw = redact(report.model_dump(mode="json"), secret)
0486 |                         write_json(output / f"{prefix}.json", raw)
0487 |                         write_requests_csv(output / f"{prefix}-requests.csv", raw)
0488 |                         summary = summarize(raw)
0489 |                         summary["expected"] = n
0490 |                         summary["missing_request_count"] = max(0, n - sum(summary[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count")))
0491 |                         rows.append({"runtime": cfg["runtime"], "model": cfg["model"],
0492 |                                      "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
0493 |                                      "phase": phase, "scenario": name, "repetition": rep+1, **summary})
0494 |                         write_summary(output, rows)
0495 |                         if summary["successful_request_count"] != n or summary["errored_request_count"] or summary["incomplete_request_count"]:
0496 |                             raise RuntimeError(f"{prefix}: requisições falharam ou execução incompleta. Veja o JSON; não compare como sucesso.")
0497 |             monitor.set_phase("warm_reference")
0498 |             lifecycle["warm_reference"] = timed_request(cfg, secret, args.timeout, prompt,
0499 |                                                          output / "warm-reference.json")
0500 |             manifest["status"] = lifecycle["status"] = "complete"
0501 |         except BaseException as exc:
0502 |             manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0503 |             manifest["error"] = redact(str(exc), secret)
0504 |             lifecycle["status"] = manifest["status"]
0505 |             lifecycle["error"] = manifest["error"]
0506 |             raise
0507 |         finally:
0508 |             for key, filename in (("first_request", "first-request.json"), ("warm_reference", "warm-reference.json")):
0509 |                 if (output / filename).exists():
0510 |                     lifecycle[key] = json.loads((output / filename).read_text(encoding="utf-8"))
0511 |             if launch is not None:
0512 |                 monitor.set_phase("server_shutdown")
0513 |                 try:
0514 |                     launch.close()
0515 |                     lifecycle["server_cleanup"] = "stopped owned process group only"
0516 |                 except Exception as exc:
0517 |                     cleanup_error = redact(str(exc), secret)
0518 |                     lifecycle["server_cleanup_error"] = cleanup_error
0519 |                     lifecycle["status"] = manifest["status"] = "failed"
0520 |             lifecycle_report(output, redact(lifecycle, secret))
0521 |             monitor.stop()
0522 |             manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
0523 |             write_json(output / "manifest.json", redact(manifest, secret))
0524 |             write_json(output / "gpu-after.json", capture(["nvidia-smi"]))
0525 |             write_summary(output, rows)
0526 |         if cleanup_error:
0527 |             raise RuntimeError(f"Falha ao encerrar processo criado: {cleanup_error}. Confira o PID no lifecycle.json.")
0528 |         print(f"Concluído. Abra {output / 'lifecycle.html'} e {output / 'summary.html'}")
0529 |
0530 |
0531 | def positive(value):
0532 |     number = int(value)
0533 |     if number <= 0:
0534 |         raise argparse.ArgumentTypeError("Use um inteiro positivo.")
0535 |     return number
0536 |
0537 |
0538 | def rebuild_report(args):
0539 |     """Atualiza apenas derivados, preservando relatórios brutos e manifesto original."""
0540 |     output = Path(args.output)
0541 |     rows = json.loads((output / "summary.json").read_text())
0542 |     manifest = json.loads((output / "manifest.json").read_text())
0543 |     for row in rows:
0544 |         prefix = f"r{row['repetition']}-{row['scenario']}-{row['phase']}"
0545 |         raw = json.loads((output / f"{prefix}.json").read_text())
0546 |         row.update(summarize(raw))
0547 |         expected = manifest.get("warmup_requests_per_case") if row["phase"] == "warmup" else manifest.get("requests")
0548 |         row["expected"] = expected
0549 |         row["missing_request_count"] = max(0, expected - sum(row[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count"))) if expected is not None else None
0550 |         write_requests_csv(output / f"{prefix}-requests.csv", raw)
0551 |     write_summary(output, rows)
0552 |     print(f"Relatório atualizado: {output / 'summary.html'}; dados brutos preservados.")
0553 |
0554 |
0555 | def main():
0556 |     parser = argparse.ArgumentParser(description=__doc__)
0557 |     commands = parser.add_subparsers(dest="command", required=True)
0558 |     report = commands.add_parser("report", help="Regenera derivados de uma execução existente, sem nova inferência.")
0559 |     report.add_argument("--output", required=True)
0560 |     report.set_defaults(func=rebuild_report)
0561 |     prep = commands.add_parser("prepare-tokenizer", help="Baixa apenas tokenizer; fixa revisão e guarda origem.")
0562 |     prep.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
0563 |     prep.add_argument("--revision", default="main")
0564 |     prep.add_argument("--output", default="tokenizer")
0565 |     prep.set_defaults(func=prepare_tokenizer)
0566 |     cmd = commands.add_parser("run", help="Mede primeiro acesso, aquecimento e GuideLLM; lançamento do servidor é opcional.")
0567 |     cmd.add_argument("--config", required=True)
0568 |     cmd.add_argument("--local-model-path", required=True, help="Pesos já no SSD: pasta HF, arquivo GGUF ou blob local do Ollama. Não baixa arquivos.")
0569 |     cmd.add_argument("--collect-kv-metrics", action="store_true", help="Amostra /metrics do vLLM (~1 Hz); ocupação do pool KV, não bytes.")
0570 |     cmd.add_argument("--kv-bytes-per-token", type=positive, help="Opcional: bytes de KV lógico por token, calculados para arquitetura/dtype reais. Estimativa, não VRAM medida.")
0571 |     cmd.add_argument("--input-tokens", nargs="+", type=positive, help="Substitui --scenarios por uma grade de comprimentos sintéticos, ex.: 256 512 1024 2048 3072.")
0572 |     cmd.add_argument("--scenarios", nargs="+", choices=list(WORKLOADS), default=["short", "medium", "long"])
0573 |     cmd.add_argument("--requests", type=positive, default=50, help="Requisições de medição por cenário e repetição; 50×3 dá 150 sucessos para percentis.")
0574 |     cmd.add_argument("--repetitions", type=positive, default=3)
0575 |     cmd.add_argument("--warmup", type=positive, default=3)
0576 |     cmd.add_argument("--seed", type=positive, default=42)
0577 |     cmd.add_argument("--timeout", type=positive, default=300)
0578 |     cmd.add_argument("--results", default="results")
0579 |     cmd.add_argument("--smoke", action="store_true", help="3 medições e 1 repetição; não vale como resultado final.")
0580 |     cmd.add_argument("--launch", help="Arquivo JSON com argv para iniciar um runtime LOCAL; encerra só esse processo ao final.")
0581 |     cmd.add_argument("--startup-timeout", type=positive, default=1800, help="Limite da espera pela API com --launch, em segundos.")
0582 |     cmd.add_argument("--first-prompt-file", help="Texto UTF-8 para a primeira requisição e referência final; default: pergunta sobre RAM/VRAM.")
0583 |     cmd.add_argument("--initial-state", default="weights local; OS/compilation caches not controlled", help="Descreva SSD e caches existentes; apenas registra, não limpa.")
0584 |     cmd.set_defaults(func=run)
0585 |     args = parser.parse_args()
0586 |     if getattr(args, "input_tokens", None):
0587 |         if len(set(args.input_tokens)) != len(args.input_tokens):
0588 |             parser.error("Não repita comprimentos em --input-tokens.")
0589 |         args.scenarios = []
0590 |         for size in args.input_tokens:
0591 |             name = f"ctx{size}"
0592 |             WORKLOADS[name] = size
0593 |             args.scenarios.append(name)
0594 |     if hasattr(args, "scenarios") and len(set(args.scenarios)) != len(args.scenarios):
0595 |         parser.error("Não repita cenários na lista.")
0596 |     try:
0597 |         args.func(args)
0598 |     except KeyboardInterrupt:
0599 |         print("Interrompido; resultados já concluídos foram preservados.", file=sys.stderr)
0600 |         return 130
0601 |     except Exception as exc:
0602 |         print(f"ERRO: {redact(str(exc), os.environ.get('BENCH_API_KEY', ''))}", file=sys.stderr)
0603 |         return 1
0604 |     return 0
0605 |
0606 |
0607 | if __name__ == "__main__":
0608 |     raise SystemExit(main())
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

SHA-256: `a85ffdf7d86d66414aa098fac38cc90cc663b2d8e8eda5587091ec6d1dacebc0`.

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
0144 |         ("time_to_first_token_milliseconds_p50", "Time To First Token p50 (ms)"),
0145 |         ("time_to_first_token_milliseconds_p95", "Time To First Token p95 (ms)"),
0146 |         ("time_to_first_token_milliseconds_p99", "Time To First Token p99 (ms)"),
0147 |         ("tokens_per_second_p50", "Tokens/s p50"),
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
0185 | <h2>4. GPU, CPU, RAM e SSD por fase</h2><p>O monitor amostra aproximadamente 1 vez/s e relaciona cada amostra a eventos/fases. GPU vem do nvidia-smi; CPU, RAM e I/O de disco vêm do host. Máximos amostrados podem perder picos e não há atribuição por processo. N/A é ausência de dado, não zero. <a href="telemetry-summary.json">Resumo de telemetria</a> · <a href="events.csv">Eventos</a> · <a href="system.csv">CPU/RAM/SSD</a></p>{hardware}
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
