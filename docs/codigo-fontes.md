# Código completo: referência linha a linha

Leia junto com [a explicação detalhada](codigo-explicado.md). Gerado dos arquivos reais; não é um segundo programa. Os números não pertencem ao Python.

## bench.py

SHA-256: `8c8eada26672759637ef4cb3adf0f09f031161faf23e56ed1f31c6a5f4a83920`.

| Função/classe | Linhas |
|---|---|
| `_synthetic_prompt` | 30–37 |
| `stream_request` | 40–86 |
| `stream_metrics` | 89–108 |
| `run_stream_batch` | 111–121 |
| `local_model_check` | 124–144 |
| `write_json` | 147–148 |
| `redact` | 151–159 |
| `capture` | 162–167 |
| `load_config` | 170–186 |
| `validate_run` | 189–202 |
| `tokenizer_digest` | 205–214 |
| `prepare_tokenizer` | 217–229 |
| `scenario_config` | 232–251 |
| `Monitor` | 254–429 |
| `percentile` | 432–438 |
| `summarize` | 441–505 |
| `write_requests_csv` | 508–524 |
| `write_summary` | 527–537 |
| `run` | 540–664 |
| `positive` | 667–671 |
| `rebuild_report` | 674–688 |
| `main` | 691–749 |
| `__init__` | 256–263 |
| `set_phase` | 265–273 |
| `start` | 275–285 |
| `log` | 287–291 |
| `kv_loop` | 293–317 |
| `loop` | 319–367 |
| `stop` | 369–380 |
| `write_telemetry_summary` | 382–429 |
| `read_rows` | 384–389 |

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
0030 | def _synthetic_prompt(tokenizer, target_tokens: int) -> str:
0031 |     """Cria texto localmente; nenhuma chamada de servidor participa do relógio."""
0032 |     seed = "Explique de forma objetiva este conceito para um estudante de estatística. "
0033 |     text = seed
0034 |     while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
0035 |         text += seed
0036 |     ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
0037 |     return tokenizer.decode(ids, skip_special_tokens=False)
0038 |
0039 |
0040 | def stream_request(cfg, prompt, timeout, secret=""):
0041 |     """Mede uma requisição SSE com ``time.perf_counter`` por requisição.
0042 |
0043 |     ``first_token_time``/``last_token_time`` são os instantes do primeiro/último
0044 |     evento SSE com conteúdo. A API OpenAI não promete um evento por token; por
0045 |     isso a resolução é a do evento de conteúdo, enquanto ``completion_tokens``
0046 |     vem exclusivamente de ``usage`` quando o servidor o fornece.
0047 |     """
0048 |     import httpx
0049 |     body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
0050 |             "temperature": 0, "top_p": 1, "max_tokens": OUTPUT_TOKENS,
0051 |             "stream": True, "stream_options": {"include_usage": True}}
0052 |     started = time.perf_counter()
0053 |     first = last = None
0054 |     chunks = 0
0055 |     content = ""
0056 |     usage = None
0057 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0058 |     with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
0059 |         with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as response:
0060 |             response.raise_for_status()
0061 |             if "text/event-stream" not in response.headers.get("content-type", ""):
0062 |                 raise ValueError("A API não respondeu com SSE.")
0063 |             for line in response.iter_lines():
0064 |                 if not line.startswith("data:"):
0065 |                     continue
0066 |                 value = line[5:].strip()
0067 |                 if value == "[DONE]":
0068 |                     break
0069 |                 event = json.loads(value)
0070 |                 if "error" in event:
0071 |                     raise ValueError(f"Erro no stream: {event['error']}")
0072 |                 if event.get("usage"):
0073 |                     usage = event["usage"]
0074 |                 text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
0075 |                 if text:
0076 |                     now = time.perf_counter()
0077 |                     first = first or now
0078 |                     last = now
0079 |                     chunks += 1
0080 |                     content += text
0081 |     ended = time.perf_counter()
0082 |     e2e = ended - started
0083 |     metrics = stream_metrics(started, first, last, ended, usage)
0084 |     return {**metrics,
0085 |             "stream_content_event_count": chunks, "output": content, "usage_observed": usage is not None,
0086 |             "request_args": json.dumps({"body": body}, ensure_ascii=False)}
0087 |
0088 |
0089 | def stream_metrics(request_start_time, first_token_time, last_token_time, request_end_time, usage):
0090 |     """Calcula métricas exclusivamente de timestamps monotônicos do cliente."""
0091 |     usage = usage if isinstance(usage, dict) else {}
0092 |     completion = usage.get("completion_tokens") if type(usage.get("completion_tokens")) is int else None
0093 |     prompt_tokens = usage.get("prompt_tokens") if type(usage.get("prompt_tokens")) is int else None
0094 |     total = usage.get("total_tokens") if type(usage.get("total_tokens")) is int else None
0095 |     ttft = first_token_time - request_start_time if first_token_time is not None else None
0096 |     e2e = request_end_time - request_start_time
0097 |     generation = (last_token_time - first_token_time) if first_token_time is not None and last_token_time is not None and completion is not None and completion > 1 else None
0098 |     inter = generation / (completion - 1) if generation is not None else None
0099 |     decode = completion / generation if completion is not None and generation and generation > 0 else None
0100 |     effective = completion / e2e if completion is not None and e2e > 0 else None
0101 |     return {"request_start_time": request_start_time, "first_token_time": first_token_time,
0102 |             "request_end_time": request_end_time, "time_to_first_token_seconds": ttft,
0103 |             "generation_time_seconds": generation, "end_to_end_latency_seconds": e2e,
0104 |             "completion_tokens": completion, "prompt_tokens": prompt_tokens, "total_tokens": total,
0105 |             "decode_tokens_per_second": decode, "end_to_end_tokens_per_second": effective,
0106 |             "inter_token_latency_seconds": inter, "time_to_first_token_ms": ttft * 1000 if ttft is not None else None,
0107 |             "request_latency": e2e, "inter_token_latency_ms": inter * 1000 if inter is not None else None,
0108 |             "output_tokens": completion, "decode_tokens_s": decode, "effective_tokens_s": effective}
0109 |
0110 |
0111 | def run_stream_batch(cfg, tokenizer, scenario, count, timeout, secret=""):
0112 |     prompt = _synthetic_prompt(tokenizer, WORKLOADS[scenario])
0113 |     successful, errored = [], []
0114 |     for _ in range(count):
0115 |         try:
0116 |             row = stream_request(cfg, prompt, timeout, secret)
0117 |             row["status"] = "successful"
0118 |             successful.append(row)
0119 |         except Exception as exc:
0120 |             errored.append({"status": "errored", "error": str(exc)})
0121 |     return {"benchmarks": [{"requests": {"successful": successful, "errored": errored, "incomplete": []}}]}
0122 |
0123 |
0124 | def local_model_check(value):
0125 |     """Checa presença, não lê pesos nem aquece o page cache deliberadamente."""
0126 |     path = Path(value).expanduser().resolve()
0127 |     if path.is_file():
0128 |         if path.suffix not in {".safetensors", ".gguf", ".bin", ".pt", ".pth"} and not path.name.startswith("sha256-"):
0129 |             raise ValueError("Informe um arquivo de pesos (não configuração/texto) ou blob sha256- do Ollama.")
0130 |         files = [path]
0131 |     elif path.is_dir():
0132 |         files = [f for f in path.rglob("*") if f.is_file() and f.suffix in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}]
0133 |         for index in path.glob("*.index.json"):
0134 |             data = json.loads(index.read_text())
0135 |             missing = [name for name in set(data.get("weight_map", {}).values()) if not (path / name).is_file()]
0136 |             if missing:
0137 |                 raise ValueError(f"Shards ausentes: {missing}")
0138 |     else:
0139 |         files = []
0140 |     if not files or any(f.stat().st_size == 0 for f in files):
0141 |         raise ValueError("Pesos locais ausentes/vazios. Prepare o modelo antes do benchmark; downloads não são permitidos na medição.")
0142 |     return {"policy": "Pesos locais obrigatórios; download excluído do protocolo.", "path": str(path),
0143 |             "files": len(files), "bytes": sum(f.stat().st_size for f in files),
0144 |             "validation": "presença/tamanho e shards declarados; não verifica conteúdo, SSD físico ou vínculo com API"}
0145 |
0146 |
0147 | def write_json(path, data):
0148 |     Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0149 |
0150 |
0151 | def redact(value, secret):
0152 |     if isinstance(value, dict):
0153 |         return {k: ("[REDACTED]" if k.lower() in {"api_key", "authorization"} else redact(v, secret))
0154 |                 for k, v in value.items()}
0155 |     if isinstance(value, list):
0156 |         return [redact(v, secret) for v in value]
0157 |     if isinstance(value, str) and secret:
0158 |         return value.replace(secret, "[REDACTED]")
0159 |     return value
0160 |
0161 |
0162 | def capture(command):
0163 |     try:
0164 |         proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
0165 |         return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
0166 |     except (OSError, subprocess.TimeoutExpired) as exc:
0167 |         return {"unavailable": str(exc)}
0168 |
0169 |
0170 | def load_config(path):
0171 |     cfg = json.loads(Path(path).read_text(encoding="utf-8"))
0172 |     required = {"runtime", "base_url", "model", "tokenizer", "context_window", "cache_policy",
0173 |                 "runtime_version", "model_artifact", "server_command", "notes"}
0174 |     if set(cfg) != required:
0175 |         raise ValueError(f"Campos da configuração devem ser exatamente: {sorted(required)}")
0176 |     if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in required - {"context_window"}):
0177 |         raise ValueError("Os campos textuais da configuração devem estar preenchidos.")
0178 |     if type(cfg["context_window"]) is not int or cfg["context_window"] < 512:
0179 |         raise ValueError("context_window deve ser um inteiro >= 512.")
0180 |     url = urlsplit(cfg["base_url"])
0181 |     if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
0182 |         raise ValueError("URL inválida: use http(s), sem credenciais, query ou fragmento.")
0183 |     if url.path not in {"", "/", "/v1", "/v1/"}:
0184 |         raise ValueError("Use a raiz do servidor (ex.: http://127.0.0.1:8000), sem endpoint.")
0185 |     cfg["base_url"] = f"{url.scheme}://{url.netloc}"
0186 |     return cfg
0187 |
0188 |
0189 | def validate_run(cfg, scenarios, smoke):
0190 |     for scenario in scenarios:
0191 |         # Margem para o template; o servidor continua sendo a autoridade final.
0192 |         needed = WORKLOADS[scenario] + OUTPUT_TOKENS + 256
0193 |         if cfg["context_window"] < needed:
0194 |             raise ValueError(f"{scenario} precisa de contexto declarado >= {needed}; "
0195 |                              "altere o servidor e depois o JSON, ou retire esse cenário.")
0196 |     if not smoke:
0197 |         if any("PREENCHER" in cfg[k] or "SUBSTITUA" in cfg[k]
0198 |                for k in ("model", "runtime_version", "model_artifact", "server_command")):
0199 |             raise ValueError("Preencha modelo, versão, artefato e comando antes da medição formal; --smoke permite rascunhos.")
0200 |         if cfg["cache_policy"] == "runtime-default-unverified":
0201 |             raise ValueError("Registre cache_policy após verificar o servidor. Ex.: disabled-confirmed ou enabled-recorded. "
0202 |                              "O script NÃO altera nem comprova a política de cache.")
0203 |
0204 |
0205 | def tokenizer_digest(path):
0206 |     path = Path(path)
0207 |     if not path.is_dir() or not (path / "tokenizer_config.json").exists():
0208 |         raise ValueError("Tokenizer local ausente. Execute: python bench.py prepare-tokenizer")
0209 |     hashes = {}
0210 |     for file in sorted(path.rglob("*")):
0211 |         if file.is_file() and not file.name.startswith("."):
0212 |             hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
0213 |     digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
0214 |     return {"sha256": digest, "files": hashes}
0215 |
0216 |
0217 | def prepare_tokenizer(args):
0218 |     from huggingface_hub import HfApi
0219 |     from transformers import AutoTokenizer
0220 |     path = Path(args.output)
0221 |     if path.exists():
0222 |         raise ValueError(f"{path} já existe; não será sobrescrito. Use outro --output.")
0223 |     revision = HfApi().model_info(args.model, revision=args.revision).sha
0224 |     tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
0225 |     path.mkdir(parents=True)
0226 |     tokenizer.save_pretrained(path)
0227 |     write_json(path / "source.json", {"model": args.model, "revision": revision})
0228 |     print(f"Tokenizer salvo em {path}; revisão {revision}. Não foram baixados pesos.")
0229 |     print("Compartilhe esta pasta com o grupo para usar os mesmos arquivos.")
0230 |
0231 |
0232 | def scenario_config(cfg, name, count, seed, secret, timeout):
0233 |     return {
0234 |         "spec": {
0235 |             "backend": {"kind": "openai_http", "target": cfg["base_url"], "model": cfg["model"],
0236 |                         "request_format": "/v1/chat/completions", "stream": True,
0237 |                         "validate_backend": False, "verify": True, "follow_redirects": False,
0238 |                         "timeout": timeout, "api_key": secret or None,
0239 |                         "extras": {"body": {"temperature": 0, "top_p": 1}}},
0240 |             "profile": {"kind": "synchronous", "warmup": 0, "cooldown": 0},
0241 |             "constraints": [{"kind": "max_requests", "count": count}, {"kind": "max_errors", "count": 1}],
0242 |             "tokenizer": {"kind": "huggingface_auto", "model": str(Path(cfg["tokenizer"]).resolve()),
0243 |                           "load_kwargs": {"local_files_only": True, "trust_remote_code": False}},
0244 |             "data": [{"kind": "synthetic_text", "prompt_tokens": WORKLOADS[name],
0245 |                       "output_tokens": OUTPUT_TOKENS}],
0246 |             "data_loader": {"kind": "pytorch", "samples": count, "num_workers": 0, "shuffle": False},
0247 |             "seed": {"kind": "static", "value": seed},
0248 |             "metrics": {"kind": "generative", "sample_size": None, "prefer_response_metrics": True},
0249 |             "outputs": [],
0250 |         }
0251 |     }
0252 |
0253 |
0254 | class Monitor:
0255 |     """Amostragem contínua e marcação de fases; não é um profiler PCIe."""
0256 |     def __init__(self, output, cfg=None, secret="", collect_kv=False):
0257 |         self.output = Path(output)
0258 |         self.stop_event = threading.Event()
0259 |         self.thread = None
0260 |         self.phase = "setup"
0261 |         self.cfg, self.secret, self.collect_kv = cfg, secret, collect_kv
0262 |         self.kv_thread = None
0263 |         self.started_monotonic = None
0264 |
0265 |     def set_phase(self, phase, event=None):
0266 |         self.phase = phase
0267 |         utc = datetime.now(timezone.utc).isoformat()
0268 |         elapsed = (time.monotonic() - self.started_monotonic) if self.started_monotonic else 0.0
0269 |         self.log(f"[telemetria] utc={utc} decorrido={elapsed:.3f}s fase={phase} evento={event or 'phase_change'}")
0270 |         if hasattr(self, "events_handle"):
0271 |             writer = csv.writer(self.events_handle)
0272 |             writer.writerow([utc, time.monotonic(), elapsed, phase, event or "phase_change"])
0273 |             self.events_handle.flush()
0274 |
0275 |     def start(self):
0276 |         self.started_monotonic = time.monotonic()
0277 |         self.telemetry_handle = (self.output / "telemetry.log").open("a", encoding="utf-8")
0278 |         self.events_handle = (self.output / "events.csv").open("w", newline="", encoding="utf-8")
0279 |         csv.writer(self.events_handle).writerow(["utc", "monotonic_s", "elapsed_s", "phase", "event"])
0280 |         self.set_phase(self.phase, "monitor_started")
0281 |         self.thread = threading.Thread(target=self.loop, daemon=True)
0282 |         self.thread.start()
0283 |         if self.collect_kv:
0284 |             self.kv_thread = threading.Thread(target=self.kv_loop, daemon=True)
0285 |             self.kv_thread.start()
0286 |
0287 |     def log(self, message):
0288 |         print(message, flush=True)
0289 |         if hasattr(self, "telemetry_handle"):
0290 |             self.telemetry_handle.write(message + "\n")
0291 |             self.telemetry_handle.flush()
0292 |
0293 |     def kv_loop(self):
0294 |         import httpx
0295 |         headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
0296 |         pattern = re.compile(r'^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc))(\{[^}]*\})?\s+([0-9.eE+\-]+)(?:\s|$)')
0297 |         with (self.output / "kv-cache.csv").open("w", newline="") as handle, httpx.Client(timeout=1, headers=headers, follow_redirects=False) as client:
0298 |             writer = csv.writer(handle)
0299 |             writer.writerow(["utc", "elapsed_s", "phase", "series", "fraction"])
0300 |             while not self.stop_event.is_set():
0301 |                 phase = self.phase
0302 |                 try:
0303 |                     response = client.get(self.cfg["base_url"] + "/metrics")
0304 |                     response.raise_for_status()
0305 |                     matches = [m for line in response.text.splitlines() if (m := pattern.match(line))]
0306 |                     modern = any(m[1] == "vllm:kv_cache_usage_perc" for m in matches)
0307 |                     for m in matches:
0308 |                         if modern and m[1] != "vllm:kv_cache_usage_perc":
0309 |                             continue
0310 |                         value = float(m[3])
0311 |                         if math.isfinite(value) and 0 <= value <= 1:
0312 |                             writer.writerow([datetime.now(timezone.utc).isoformat(), time.monotonic() - self.started_monotonic,
0313 |                                              phase, redact(m[1] + (m[2] or ""), self.secret), value])
0314 |                     handle.flush()
0315 |                 except (httpx.HTTPError, ValueError):
0316 |                     pass  # Serveur inicializando/endpoint ausente: nunca inventar zeros.
0317 |                 self.stop_event.wait(1)
0318 |
0319 |     def loop(self):
0320 |         try:
0321 |             import psutil
0322 |         except ImportError:
0323 |             psutil = None
0324 |         with (self.output / "gpu.csv").open("w", newline="", encoding="utf-8") as handle, \
0325 |              (self.output / "system.csv").open("w", newline="", encoding="utf-8") as system_handle:
0326 |             writer = csv.writer(handle)
0327 |             writer.writerow(["utc", "elapsed_s", "sample_index", "phase", "index", "name", "used_mib", "total_mib", "used_gib", "total_gib",
0328 |                              "vram_used_pct", "gpu_util_pct", "memory_util_pct", "temperature_c", "power_w"])
0329 |             system_writer = csv.writer(system_handle)
0330 |             system_writer.writerow(["utc", "elapsed_s", "sample_index", "phase", "cpu_util_pct", "ram_used_mib", "ram_available_mib", "ram_total_mib",
0331 |                                     "load1", "root_disk_used_mib", "root_disk_free_mib", "disk_read_bytes", "disk_write_bytes"])
0332 |             if psutil:
0333 |                 psutil.cpu_percent(interval=None)
0334 |             sample_number = 0
0335 |             while not self.stop_event.is_set():
0336 |                 sample_number += 1
0337 |                 phase = self.phase
0338 |                 utc = datetime.now(timezone.utc).isoformat()
0339 |                 result = capture(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
0340 |                                   "--format=csv,noheader,nounits"])
0341 |                 if result.get("returncode") != 0:
0342 |                     write_json(self.output / "gpu-unavailable.json", result)
0343 |                 else:
0344 |                     gpu_rows = list(csv.reader(result["stdout"].splitlines(), skipinitialspace=True))
0345 |                     for row in gpu_rows:
0346 |                         used_mib, total_mib = float(row[2]), float(row[3])
0347 |                         vram_pct = (100 * used_mib / total_mib) if total_mib else None
0348 |                         writer.writerow([utc, time.monotonic() - self.started_monotonic, sample_number, phase, row[0], row[1], row[2], row[3],
0349 |                                          used_mib / 1024, total_mib / 1024, vram_pct, *row[4:]])
0350 |                     if sample_number == 1 or sample_number % 10 == 0:
0351 |                         compact = "; ".join(f"GPU{row[0]} VRAM={float(row[2]) / 1024:.2f}/{float(row[3]) / 1024:.2f} GiB "
0352 |                                              f"ocupada={100 * float(row[2]) / float(row[3]):.1f}% "
0353 |                                              f"atividade_gpu={row[4]}% atividade_leitura_escrita_memoria={row[5]}%" for row in gpu_rows)
0354 |                         elapsed = time.monotonic() - self.started_monotonic
0355 |                         utc_log = datetime.now(timezone.utc).isoformat()
0356 |                         self.log(f"[telemetria] utc={utc_log} decorrido={elapsed:.3f}s fase={phase} {compact or 'GPU sem amostra'}")
0357 |                 handle.flush()
0358 |                 if psutil:
0359 |                     vm = psutil.virtual_memory()
0360 |                     du = psutil.disk_usage(str(self.output.anchor or "/"))
0361 |                     io = psutil.disk_io_counters()
0362 |                     system_writer.writerow([utc, time.monotonic() - self.started_monotonic, sample_number, phase, psutil.cpu_percent(interval=None), vm.used / 1048576,
0363 |                                              vm.available / 1048576, vm.total / 1048576, os.getloadavg()[0],
0364 |                                              du.used / 1048576, du.free / 1048576,
0365 |                                              getattr(io, "read_bytes", None), getattr(io, "write_bytes", None)])
0366 |                     system_handle.flush()
0367 |                 self.stop_event.wait(1)
0368 |
0369 |     def stop(self):
0370 |         self.stop_event.set()
0371 |         if self.thread:
0372 |             self.thread.join(timeout=17)
0373 |         if self.kv_thread:
0374 |             self.kv_thread.join(timeout=3)
0375 |         self.set_phase("stopped", "monitor_stopped")
0376 |         if hasattr(self, "events_handle"):
0377 |             self.events_handle.close()
0378 |         if hasattr(self, "telemetry_handle"):
0379 |             self.telemetry_handle.close()
0380 |         self.write_telemetry_summary()
0381 |
0382 |     def write_telemetry_summary(self):
0383 |         """Agrega telemetria por fase para relacionar picos com eventos do benchmark."""
0384 |         def read_rows(name):
0385 |             path = self.output / name
0386 |             if not path.exists():
0387 |                 return []
0388 |             with path.open() as handle:
0389 |                 return list(csv.DictReader(handle))
0390 |         sources = {"gpu": read_rows("gpu.csv"), "system": read_rows("system.csv"), "kv": read_rows("kv-cache.csv")}
0391 |         # Arquivo de entrada simples para gráficos: uma observação por linha,
0392 |         # com UTC, tempo desde o início do monitor e fase experimental.
0393 |         if sources["gpu"]:
0394 |             import shutil
0395 |             shutil.copyfile(self.output / "gpu.csv", self.output / "telemetry-timeseries.csv")
0396 |         phases = sorted({r.get("phase") for rows in sources.values() for r in rows if r.get("phase")})
0397 |         output = {"definition": "Amostras observadas por fase; não são bytes nem tempos de transferência PCIe.", "phases": {}}
0398 |         for phase in phases:
0399 |             entry = {"gpu_samples": 0, "system_samples": 0, "kv_samples": 0}
0400 |             grows = [r for r in sources["gpu"] if r.get("phase") == phase]
0401 |             for key in ("used_mib", "total_mib", "used_gib", "total_gib", "vram_used_pct", "gpu_util_pct", "memory_util_pct", "temperature_c", "power_w"):
0402 |                 vals = []
0403 |                 for r in grows:
0404 |                     try: vals.append(float(r[key]))
0405 |                     except (ValueError, TypeError, KeyError): pass
0406 |                 entry[f"gpu_{key}_mean"] = sum(vals) / len(vals) if vals else None
0407 |                 entry[f"gpu_{key}_max"] = max(vals) if vals else None
0408 |                 if key in {"used_mib", "used_gib", "vram_used_pct"}:
0409 |                     entry[f"gpu_{key}_min"] = min(vals) if vals else None
0410 |             entry["gpu_samples"] = len(grows)
0411 |             srows = [r for r in sources["system"] if r.get("phase") == phase]
0412 |             for key in ("cpu_util_pct", "ram_used_mib", "ram_available_mib", "root_disk_used_mib", "root_disk_free_mib"):
0413 |                 vals = []
0414 |                 for r in srows:
0415 |                     try: vals.append(float(r[key]))
0416 |                     except (ValueError, TypeError, KeyError): pass
0417 |                 entry[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
0418 |                 entry[f"{key}_max"] = max(vals) if vals else None
0419 |             entry["system_samples"] = len(srows)
0420 |             krows = [r for r in sources["kv"] if r.get("phase") == phase]
0421 |             vals = []
0422 |             for r in krows:
0423 |                 try: vals.append(float(r["fraction"]) * 100)
0424 |                 except (ValueError, TypeError, KeyError): pass
0425 |             entry["kv_occupancy_pct_mean"] = sum(vals) / len(vals) if vals else None
0426 |             entry["kv_occupancy_pct_max"] = max(vals) if vals else None
0427 |             entry["kv_samples"] = len(vals)
0428 |             output["phases"][phase] = entry
0429 |         write_json(self.output / "telemetry-summary.json", output)
0430 |
0431 |
0432 | def percentile(values, q):
0433 |     values = sorted(v for v in values if v is not None and math.isfinite(v))
0434 |     if not values:
0435 |         return None
0436 |     pos = (len(values) - 1) * q
0437 |     lo, hi = math.floor(pos), math.ceil(pos)
0438 |     return values[lo] + (values[hi] - values[lo]) * (pos - lo)
0439 |
0440 |
0441 | def summarize(report):
0442 |     """Métricas por requisição, sem misturar warmup ou erros com sucessos."""
0443 |     benchmarks = report["benchmarks"]
0444 |     if len(benchmarks) != 1:
0445 |         raise ValueError("Esperado exatamente um benchmark sequencial.")
0446 |     requests = benchmarks[0]["requests"]
0447 |     from reporting import derived
0448 |     good = [{**r, **derived(r)} for r in requests["successful"]]
0449 |     result = {
0450 |         "successful_request_count": len(good),
0451 |         "errored_request_count": len(requests["errored"]),
0452 |         "incomplete_request_count": len(requests["incomplete"]),
0453 |     }
0454 |     # Os nomes são deliberadamente longos: summary.json é um artefato de
0455 |     # análise, e não uma API em que economizar alguns bytes melhora algo.
0456 |     metrics = {
0457 |         "request_first_token_latency_milliseconds": "time_to_first_token_ms",
0458 |         "request_latency_seconds": "request_latency",
0459 |         "within_response_next_token_latency_milliseconds": "inter_token_latency_ms",
0460 |         "output_completion_token_count": "output_tokens",
0461 |         "input_prompt_token_count": "prompt_tokens",
0462 |         "decode_generation_tokens_per_second": "decode_tokens_s",
0463 |         "effective_output_tokens_per_second": "effective_tokens_s",
0464 |     }
0465 |     for label, key in metrics.items():
0466 |         vals = [r.get(key) for r in good]
0467 |         result[label + "_sample_count"] = sum(v is not None for v in vals)
0468 |         result[label + "_p50"] = percentile(vals, .5)
0469 |         result[label + "_p95"] = percentile(vals, .95)
0470 |         result[label + "_p99"] = percentile(vals, .99)
0471 |     # Nomes canônicos para a comparação entre runtimes. Os campos históricos
0472 |     # acima permanecem para compatibilidade com relatórios já gerados.
0473 |     ttft = [r.get("time_to_first_token_ms") for r in good]
0474 |     tokens_s = [r.get("decode_tokens_s") for r in good]
0475 |     result["time_to_first_token_milliseconds_sample_count"] = sum(v is not None for v in ttft)
0476 |     result["tokens_per_second_sample_count"] = sum(v is not None for v in tokens_s)
0477 |     for suffix, q in (("p50", .50), ("p95", .95), ("p99", .99)):
0478 |         result[f"time_to_first_token_milliseconds_{suffix}"] = percentile(ttft, q)
0479 |         result[f"tokens_per_second_{suffix}"] = percentile(tokens_s, q)
0480 |     for label, key in {
0481 |         "time_to_first_token_seconds": "time_to_first_token_seconds",
0482 |         "generation_time_seconds": "generation_time_seconds",
0483 |         "end_to_end_latency_seconds": "end_to_end_latency_seconds",
0484 |         "decode_tokens_per_second": "decode_tokens_per_second",
0485 |         "end_to_end_tokens_per_second": "end_to_end_tokens_per_second",
0486 |     }.items():
0487 |         values = [r.get(key) for r in good]
0488 |         result[label + "_sample_count"] = sum(v is not None for v in values)
0489 |         result[label + "_p50"] = percentile(values, .50)
0490 |         result[label + "_p95"] = percentile(values, .95)
0491 |         result[label + "_p99"] = percentile(values, .99)
0492 |     result["metric_definitions"] = {
0493 |         "Time To First Token": "milissegundos entre o envio da requisição e o primeiro token/conteúdo observado; há um valor por requisição.",
0494 |         "Tokens/s": "tokens de saída por segundo durante o decode, calculado por requisição a partir do intervalo entre tokens; não inclui TTFT.",
0495 |     }
0496 |     # O hash exclui aliases do modelo e chaves: apenas carga de entrada e limite de saída.
0497 |     bodies = []
0498 |     for row in good:
0499 |         args = json.loads(row["request_args"])
0500 |         body = args.get("body", {})
0501 |         bodies.append({k: body.get(k) for k in ("messages", "max_tokens")})
0502 |     result["requests_sha256"] = hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
0503 |     result["percentiles_are_exploratory"] = len(good) < 100
0504 |     result["percentile_definition"] = "Empirical linear interpolation over successful requests in this phase/scenario/repetition block."
0505 |     return result
0506 |
0507 |
0508 | def write_requests_csv(path, report):
0509 |     """Amostras individuais para análise no R/Python, incluindo status de erro."""
0510 |     from reporting import derived
0511 |     fields = ["status", "error", "request_id", "request_start_time", "first_token_time", "request_end_time",
0512 |               "time_to_first_token_seconds", "generation_time_seconds", "end_to_end_latency_seconds",
0513 |               "completion_tokens", "prompt_tokens", "total_tokens", "decode_tokens_per_second",
0514 |               "end_to_end_tokens_per_second", "inter_token_latency_seconds", "request_latency",
0515 |               "time_to_first_token_ms", "inter_token_latency_ms", "output_tokens", "decode_tokens_s", "effective_tokens_s",
0516 |               "context_start_tokens", "context_end_tokens", "context_band"]
0517 |     with Path(path).open("w", newline="", encoding="utf-8") as handle:
0518 |         writer = csv.DictWriter(handle, fieldnames=fields)
0519 |         writer.writeheader()
0520 |         for status in ("successful", "errored", "incomplete"):
0521 |             for row in report["benchmarks"][0]["requests"][status]:
0522 |                 if status == "successful":
0523 |                     row = {**row, **derived(row)}
0524 |                 writer.writerow({"status": status, **{key: row.get(key) for key in fields[1:]}})
0525 |
0526 |
0527 | def write_summary(output, rows):
0528 |     from reporting import render
0529 |     write_json(output / "summary.json", rows)
0530 |     if not rows:
0531 |         render(output, rows)
0532 |         return
0533 |     with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
0534 |         writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
0535 |         writer.writeheader()
0536 |         writer.writerows(rows)
0537 |     render(output, rows)
0538 |
0539 |
0540 | def run(args):
0541 |     import fcntl
0542 |     from lifecycle import DEFAULT_PROMPT, Launch, lifecycle_report, timed_request, wait_models
0543 |     if importlib.metadata.version("guidellm") != GUIDELLM_VERSION:
0544 |         raise ValueError(f"Este projeto exige guidellm=={GUIDELLM_VERSION}; reinstale requirements.txt.")
0545 |     cfg = load_config(args.config)
0546 |     validate_run(cfg, args.scenarios, args.smoke)
0547 |     availability = local_model_check(args.local_model_path)
0548 |     availability["kv_bytes_per_token"] = args.kv_bytes_per_token
0549 |     availability["launch_hf_offline"] = bool(args.launch)
0550 |     digest = tokenizer_digest(cfg["tokenizer"])
0551 |     from transformers import AutoTokenizer
0552 |     measurement_tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True, trust_remote_code=False)
0553 |     count, repetitions = (3, 1) if args.smoke else (args.requests, args.repetitions)
0554 |     secret = os.environ.get("BENCH_API_KEY", "")
0555 |     prompt = Path(args.first_prompt_file).read_text(encoding="utf-8") if args.first_prompt_file else DEFAULT_PROMPT
0556 |     if not prompt.strip():
0557 |         raise ValueError("O prompt inicial não pode estar vazio.")
0558 |     base = Path(args.results)
0559 |     base.mkdir(parents=True, exist_ok=True)
0560 |     with (base / ".benchmark.lock").open("a") as lock:
0561 |         try:
0562 |             fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
0563 |         except BlockingIOError:
0564 |             raise ValueError("Já existe um benchmark usando esta pasta results. Não execute dois ao mesmo tempo.") from None
0565 |         stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
0566 |         output = base / stamp
0567 |         output.mkdir()
0568 |         manifest = {"project_version": VERSION, "guidellm_version": GUIDELLM_VERSION, "started_utc": stamp,
0569 |                     "config": cfg, "tokenizer": digest, "smoke": args.smoke, "requests": count,
0570 |                     "repetitions": repetitions, "warmup_requests_per_case": args.warmup,
0571 |                     "scenarios": args.scenarios, "seed": args.seed, "profile": "synchronous",
0572 |                     "model_availability": availability,
0573 |                     "guidellm_compat": "0.7.4 bounded drain of real late completion updates (5s); no request retry",
0574 |                     "telemetry": {"gpu_source": "local nvidia-smi", "system_source": "psutil host CPU/RAM/disk counters",
0575 |                                   "event_source": "events.csv phase markers", "kv_metrics_requested": args.collect_kv_metrics,
0576 |                                   "sampling": "approximately 1 Hz; not per-token; PCIe copy time is not directly measured"},
0577 |                     "python": sys.version, "platform": platform.platform(),
0578 |                     "git": capture(["git", "rev-parse", "HEAD"]), "status": "running"}
0579 |         write_json(output / "manifest.json", redact(manifest, secret))
0580 |         write_json(output / "client-packages.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
0581 |         write_json(output / "gpu-before.json", capture(["nvidia-smi"]))
0582 |         monitor, rows = Monitor(output, cfg, secret, args.collect_kv_metrics), []
0583 |         launch, origin, cleanup_error = None, None, None
0584 |         lifecycle = {"mode": "new-process" if args.launch else "existing-server-state-unknown",
0585 |                      "status": "running", "initial_state_note": args.initial_state,
0586 |                      "startup_timeout_s": args.startup_timeout}
0587 |         print(f"Resultados: {output.resolve()}", flush=True)
0588 |         try:
0589 |             monitor.start()
0590 |             if args.launch:
0591 |                 launch = Launch(cfg, args.launch, output, args.launch_extra_args, args.launch_executable)
0592 |                 lifecycle["argv"] = redact(launch.argv, secret)
0593 |                 monitor.set_phase("process_startup")
0594 |                 origin = launch.start()
0595 |                 lifecycle["pid"] = launch.process.pid
0596 |             lifecycle_report(output, redact(lifecycle, secret))
0597 |             monitor.set_phase("server_readiness")
0598 |             lifecycle["readiness"] = wait_models(cfg, secret, args.startup_timeout, launch)
0599 |             lifecycle_report(output, redact(lifecycle, secret))
0600 |             monitor.set_phase("first_request")
0601 |             print("Primeiro POST: medição da primeira resposta (nenhuma geração prévia enviada pelo cliente).", flush=True)
0602 |             lifecycle["first_request"] = timed_request(cfg, secret, args.timeout, prompt,
0603 |                                                        output / "first-request.json", origin)
0604 |             lifecycle_report(output, redact(lifecycle, secret))
0605 |             for rep in range(repetitions):
0606 |                 # Rotação balanceia parcialmente a posição dos cenários entre repetições.
0607 |                 names = args.scenarios[rep % len(args.scenarios):] + args.scenarios[:rep % len(args.scenarios)]
0608 |                 for name in names:
0609 |                     for phase, n in (("warmup", args.warmup), ("measure", count)):
0610 |                         if not n:
0611 |                             continue
0612 |                         seed = args.seed + rep * 100 + list(WORKLOADS).index(name)
0613 |                         if phase == "warmup":
0614 |                             seed += 1_000_000
0615 |                         prefix = f"r{rep+1}-{name}-{phase}"
0616 |                         monitor.set_phase(prefix)
0617 |                         config = scenario_config(cfg, name, n, seed, secret, args.timeout)
0618 |                         write_json(output / f"{prefix}-config.json", redact(config, secret))
0619 |                         print(f"{prefix}: {n} requisições, uma por vez", flush=True)
0620 |                         report = run_stream_batch(cfg, measurement_tokenizer, name, n, args.timeout, secret)
0621 |                         raw = redact(report, secret)
0622 |                         write_json(output / f"{prefix}.json", raw)
0623 |                         write_requests_csv(output / f"{prefix}-requests.csv", raw)
0624 |                         summary = summarize(raw)
0625 |                         summary["expected"] = n
0626 |                         summary["missing_request_count"] = max(0, n - sum(summary[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count")))
0627 |                         rows.append({"runtime": cfg["runtime"], "model": cfg["model"],
0628 |                                      "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
0629 |                                      "phase": phase, "scenario": name, "repetition": rep+1, **summary})
0630 |                         write_summary(output, rows)
0631 |                         if summary["successful_request_count"] != n or summary["errored_request_count"] or summary["incomplete_request_count"]:
0632 |                             raise RuntimeError(f"{prefix}: requisições falharam ou execução incompleta. Veja o JSON; não compare como sucesso.")
0633 |             monitor.set_phase("warm_reference")
0634 |             lifecycle["warm_reference"] = timed_request(cfg, secret, args.timeout, prompt,
0635 |                                                          output / "warm-reference.json")
0636 |             manifest["status"] = lifecycle["status"] = "complete"
0637 |         except BaseException as exc:
0638 |             manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0639 |             manifest["error"] = redact(str(exc), secret)
0640 |             lifecycle["status"] = manifest["status"]
0641 |             lifecycle["error"] = manifest["error"]
0642 |             raise
0643 |         finally:
0644 |             for key, filename in (("first_request", "first-request.json"), ("warm_reference", "warm-reference.json")):
0645 |                 if (output / filename).exists():
0646 |                     lifecycle[key] = json.loads((output / filename).read_text(encoding="utf-8"))
0647 |             if launch is not None:
0648 |                 monitor.set_phase("server_shutdown")
0649 |                 try:
0650 |                     launch.close()
0651 |                     lifecycle["server_cleanup"] = "stopped owned process group only"
0652 |                 except Exception as exc:
0653 |                     cleanup_error = redact(str(exc), secret)
0654 |                     lifecycle["server_cleanup_error"] = cleanup_error
0655 |                     lifecycle["status"] = manifest["status"] = "failed"
0656 |             lifecycle_report(output, redact(lifecycle, secret))
0657 |             monitor.stop()
0658 |             manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
0659 |             write_json(output / "manifest.json", redact(manifest, secret))
0660 |             write_json(output / "gpu-after.json", capture(["nvidia-smi"]))
0661 |             write_summary(output, rows)
0662 |         if cleanup_error:
0663 |             raise RuntimeError(f"Falha ao encerrar processo criado: {cleanup_error}. Confira o PID no lifecycle.json.")
0664 |         print(f"Concluído. Abra {output / 'lifecycle.html'} e {output / 'summary.html'}")
0665 |
0666 |
0667 | def positive(value):
0668 |     number = int(value)
0669 |     if number <= 0:
0670 |         raise argparse.ArgumentTypeError("Use um inteiro positivo.")
0671 |     return number
0672 |
0673 |
0674 | def nonnegative(value):
0675 |     number = int(value)
0676 |     if number < 0:
0677 |         raise argparse.ArgumentTypeError("Use zero ou um inteiro positivo.")
0678 |     return number
0679 |
0680 |
0681 | def rebuild_report(args):
0682 |     """Atualiza apenas derivados, preservando relatórios brutos e manifesto original."""
0683 |     output = Path(args.output)
0684 |     rows = json.loads((output / "summary.json").read_text())
0685 |     manifest = json.loads((output / "manifest.json").read_text())
0686 |     for row in rows:
0687 |         prefix = f"r{row['repetition']}-{row['scenario']}-{row['phase']}"
0688 |         raw = json.loads((output / f"{prefix}.json").read_text())
0689 |         row.update(summarize(raw))
0690 |         expected = manifest.get("warmup_requests_per_case") if row["phase"] == "warmup" else manifest.get("requests")
0691 |         row["expected"] = expected
0692 |     parser = argparse.ArgumentParser(description=__doc__)
0693 |     commands = parser.add_subparsers(dest="command", required=True)
0694 |     report = commands.add_parser("report", help="Regenera derivados de uma execução existente, sem nova inferência.")
0695 |     report.add_argument("--output", required=True)
0696 |     report.set_defaults(func=rebuild_report)
0697 |     prep = commands.add_parser("prepare-tokenizer", help="Baixa apenas tokenizer; fixa revisão e guarda origem.")
0698 |     prep.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
0699 |     prep.add_argument("--revision", default="main")
0700 |     prep.add_argument("--output", default="tokenizer")
0701 |     prep.set_defaults(func=prepare_tokenizer)
0702 |     cmd = commands.add_parser("run", help="Mede primeiro acesso, aquecimento e GuideLLM; lançamento do servidor é opcional.")
0703 |     cmd.add_argument("--config", required=True)
0704 |     cmd.add_argument("--local-model-path", required=True, help="Pesos já no SSD: pasta HF, arquivo GGUF ou blob local do Ollama. Não baixa arquivos.")
0705 |     cmd.add_argument("--collect-kv-metrics", action="store_true", help="Amostra /metrics do vLLM (~1 Hz); ocupação do pool KV, não bytes.")
0706 |     cmd.add_argument("--kv-bytes-per-token", type=positive, help="Opcional: bytes de KV lógico por token, calculados para arquitetura/dtype reais. Estimativa, não VRAM medida.")
0707 |     cmd.add_argument("--input-tokens", nargs="+", type=positive, help="Substitui --scenarios por uma grade de comprimentos sintéticos, ex.: 256 512 1024 2048 3072.")
0708 |     cmd.add_argument("--scenarios", nargs="+", choices=list(WORKLOADS), default=["short", "medium", "long"])
0709 |     cmd.add_argument("--requests", type=positive, default=50, help="Requisições de medição por cenário e repetição; 50×3 dá 150 sucessos para percentis.")
0710 |     cmd.add_argument("--repetitions", type=positive, default=3)
0711 |     cmd.add_argument("--warmup", type=nonnegative, default=3)
0712 |     cmd.add_argument("--seed", type=positive, default=42)
0713 |     cmd.add_argument("--timeout", type=positive, default=300)
0714 |     cmd.add_argument("--results", default="results")
0715 |     cmd.add_argument("--smoke", action="store_true", help="3 medições e 1 repetição; não vale como resultado final.")
0716 |     cmd.add_argument("--launch", help="Arquivo JSON com argv para iniciar um runtime LOCAL; encerra só esse processo ao final.")
0717 |     cmd.add_argument("--launch-extra-args", nargs="*", default=[],
0718 |                      help="Argumentos experimentais acrescentados ao argv do launch, sem editar o JSON; registrados no lifecycle.")
0719 |     cmd.add_argument("--launch-executable", help="Substitui argv[0] do launch pelo executável resolvido no ambiente do runtime.")
0720 |     cmd.add_argument("--launch-extra-args-json", default="[]", help="Array JSON de argumentos do runtime, preservando flags e espaços.")
0721 |     cmd.add_argument("--startup-timeout", type=positive, default=1800, help="Limite da espera pela API com --launch, em segundos.")
0722 |     cmd.add_argument("--first-prompt-file", help="Texto UTF-8 para a primeira requisição e referência final; default: pergunta sobre RAM/VRAM.")
0723 |     cmd.add_argument("--initial-state", default="weights local; OS/compilation caches not controlled", help="Descreva SSD e caches existentes; apenas registra, não limpa.")
0724 |     cmd.set_defaults(func=run)
0725 |     args = parser.parse_args()
0726 |     if hasattr(args, "launch_extra_args_json"):
0727 |         extra = json.loads(args.launch_extra_args_json)
0728 |         if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
0729 |             parser.error("--launch-extra-args-json deve ser array de strings")
0730 |         args.launch_extra_args.extend(extra)
0731 |     if getattr(args, "input_tokens", None):
0732 |         if len(set(args.input_tokens)) != len(args.input_tokens):
0733 |             parser.error("Não repita comprimentos em --input-tokens.")
0734 |         args.scenarios = []
0735 |         for size in args.input_tokens:
0736 |             name = f"ctx{size}"
0737 |             WORKLOADS[name] = size
0738 |             args.scenarios.append(name)
0739 |     if hasattr(args, "scenarios") and len(set(args.scenarios)) != len(args.scenarios):
0740 |         parser.error("Não repita cenários na lista.")
0741 |     try:
0742 |         args.func(args)
0743 |     except KeyboardInterrupt:
0744 |         print("Interrompido; resultados já concluídos foram preservados.", file=sys.stderr)
0745 |         return 130
0746 |     except Exception as exc:
0747 |         print(f"ERRO: {redact(str(exc), os.environ.get('BENCH_API_KEY', ''))}", file=sys.stderr)
0748 |         return 1
0749 |     return 0
0750 |
0751 |
0752 | if __name__ == "__main__":
0753 |     raise SystemExit(main())
```

## lifecycle.py

SHA-256: `d943f188a5ddf9c201c83888d975af0855a043f7b17e37e83d118b02c5f4f683`.

| Função/classe | Linhas |
|---|---|
| `Launch` | 23–81 |
| `wait_models` | 84–125 |
| `timed_request` | 128–222 |
| `lifecycle_report` | 225–240 |
| `__init__` | 24–41 |
| `start` | 43–61 |
| `close` | 63–81 |

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
0024 |     def __init__(self, cfg, command_file, output, extra_args=None, executable=None):
0025 |         url = urlsplit(cfg["base_url"])
0026 |         if url.hostname not in {"127.0.0.1", "localhost", "::1"}:
0027 |             raise ValueError("--launch só aceita servidor local (localhost).")
0028 |         self.host, self.port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
0029 |         self.argv = json.loads(Path(command_file).read_text(encoding="utf-8"))
0030 |         if not isinstance(self.argv, list) or not self.argv or any(not isinstance(x, str) or not x for x in self.argv):
0031 |             raise ValueError("O arquivo --launch deve conter um array JSON não vazio de strings (argv).")
0032 |         if executable:
0033 |             self.argv[0] = executable
0034 |         if extra_args:
0035 |             if any(not isinstance(x, str) or not x for x in extra_args):
0036 |                 raise ValueError("--launch-extra-args aceita somente strings não vazias.")
0037 |             self.argv.extend(extra_args)
0038 |         self.output = Path(output)
0039 |         self.process = None
0040 |         self.log = None
0041 |         self.started = None
0042 |
0043 |     def start(self):
0044 |         # Não interrompe servidores existentes nem tenta tomar uma porta ocupada.
0045 |         try:
0046 |             connection = socket.create_connection((self.host, self.port), timeout=1)
0047 |         except OSError:
0048 |             pass
0049 |         else:
0050 |             connection.close()
0051 |             raise ValueError("A porta já está em uso. Pare o seu servidor manualmente antes de usar --launch.")
0052 |         self.log = (self.output / "server.log").open("w", encoding="utf-8")
0053 |         self.started = time.perf_counter()
0054 |         try:
0055 |             self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
0056 |                                             start_new_session=True, shell=False,
0057 |                                             env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
0058 |         except BaseException:
0059 |             self.log.close()
0060 |             raise
0061 |         return self.started
0062 |
0063 |     def close(self):
0064 |         """Encerra exclusivamente o grupo criado por este objeto, inclusive filhos."""
0065 |         if self.process is not None:
0066 |             try:
0067 |                 os.killpg(self.process.pid, signal.SIGTERM)
0068 |             except ProcessLookupError:
0069 |                 pass
0070 |             try:
0071 |                 self.process.wait(timeout=20)
0072 |             except subprocess.TimeoutExpired:
0073 |                 os.killpg(self.process.pid, signal.SIGKILL)
0074 |                 self.process.wait(timeout=5)
0075 |             # Alguns workers podem sobreviver ao encerramento do processo líder.
0076 |             try:
0077 |                 os.killpg(self.process.pid, signal.SIGKILL)
0078 |             except ProcessLookupError:
0079 |                 pass
0080 |         if self.log is not None:
0081 |             self.log.close()
0082 |
0083 |
0084 | def wait_models(cfg, secret, timeout, launch=None):
0085 |     """Apenas GET. Modelo listado não comprova que seus pesos estão na GPU."""
0086 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0087 |     started = time.perf_counter()
0088 |     probes, last, next_update = 0, None, started + 15
0089 |     with httpx.Client(timeout=2, headers=headers, follow_redirects=False) as client:
0090 |         while True:
0091 |             probes += 1
0092 |             if launch is not None and launch.process.poll() is not None:
0093 |                 raise RuntimeError(f"Servidor encerrou antes de ficar disponível (exit={launch.process.returncode}). Veja server.log.")
0094 |             try:
0095 |                 response = client.get(cfg["base_url"] + "/v1/models")
0096 |                 if response.status_code in {401, 403}:
0097 |                     raise ValueError("API recusou autenticação; confira BENCH_API_KEY.")
0098 |                 response.raise_for_status()
0099 |                 models = [m["id"] for m in response.json().get("data", [])]
0100 |                 expected = cfg["model"]
0101 |                 accepted = {expected}
0102 |                 if cfg.get("runtime") == "ollama" and ":" not in expected:
0103 |                     accepted.add(expected + ":latest")
0104 |                 if accepted.intersection(models):
0105 |                     observed = time.perf_counter()
0106 |                     return {"models": models, "get_probes": probes,
0107 |                             "wait_wall_s": observed - started,
0108 |                             "process_to_api_observed_s": observed - launch.started if launch else None,
0109 |                             "criterion": "GET /v1/models retornou o ID; não é prova de pesos residentes"}
0110 |                 last = f"API respondeu, mas modelo esperado={expected!r}; disponíveis={models!r}"
0111 |                 if models:
0112 |                     raise RuntimeError(last + ". Confira --alias/--served-model-name e config.model.")
0113 |             except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
0114 |                 last = str(exc)
0115 |             elapsed = time.perf_counter() - started
0116 |             if not launch or elapsed >= timeout:
0117 |                 raise RuntimeError(f"API/modelo não disponível após {elapsed:.1f}s: {last}")
0118 |             if time.perf_counter() >= next_update:
0119 |                 print(f"[inicialização] runtime={cfg.get('runtime', 'não informado')} "
0120 |                       f"PID={launch.process.pid if launch else 'externo'} decorrido={elapsed:.1f}s "
0121 |                       f"limite={timeout}s; verificando GET {cfg['base_url']}/v1/models "
0122 |                       f"para modelo={cfg['model']!r}; última observação: {last}. "
0123 |                       f"Log do servidor: {launch.output / 'server.log' if launch else 'externo'}", flush=True)
0124 |                 next_update = time.perf_counter() + 15
0125 |             time.sleep(min(.5, max(0, timeout - elapsed)))
0126 |
0127 |
0128 | def timed_request(cfg, secret, timeout, prompt, output, process_origin=None):
0129 |     """Primeiro POST é simultaneamente medição e validação, sem pré-aquecimento oculto.
0130 |
0131 |     Grava resultado parcial inclusive em timeout, stream inválido ou usage ausente.
0132 |     TTFT aqui é primeiro conteúdo não vazio recebido (não mero cabeçalho/role).
0133 |     """
0134 |     path = Path(output)
0135 |     body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
0136 |             "temperature": 0, "top_p": 1, "max_tokens": 128, "stream": True,
0137 |             "stream_options": {"include_usage": True}}
0138 |     result = {"status": "running", "body": body, "stream_usage": None,
0139 |               "content_event_offsets_s": [], "output": "", "done": False,
0140 |               "ttft_ms": None, "e2e_s": None, "mean_itl_ms": None,
0141 |               "usage_observed": False, "request_start_time": None,
0142 |               "first_token_time": None, "request_end_time": None,
0143 |               "time_to_first_token_seconds": None, "generation_time_seconds": None,
0144 |               "end_to_end_latency_seconds": None, "completion_tokens": None,
0145 |               "prompt_tokens": None, "total_tokens": None,
0146 |               "decode_tokens_per_second": None, "end_to_end_tokens_per_second": None,
0147 |               "process_to_first_content_s": None, "process_to_response_end_s": None}
0148 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0149 |     started = None
0150 |     try:
0151 |         with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
0152 |             started = time.perf_counter()
0153 |             result["request_start_time"] = started
0154 |             with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as stream:
0155 |                 result["headers_ms"] = (time.perf_counter() - started) * 1000
0156 |                 stream.raise_for_status()
0157 |                 if "text/event-stream" not in stream.headers.get("content-type", ""):
0158 |                     raise ValueError("A API não respondeu com SSE.")
0159 |                 for line in stream.iter_lines():
0160 |                     if not line.startswith("data:"):
0161 |                         continue
0162 |                     value = line[5:].strip()
0163 |                     if value == "[DONE]":
0164 |                         result["done"] = True
0165 |                         break
0166 |                     event = json.loads(value)
0167 |                     if "error" in event:
0168 |                         raise ValueError(f"Erro no stream: {event['error']}")
0169 |                     result["stream_usage"] = event.get("usage") or result["stream_usage"]
0170 |                     text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
0171 |                     if text:
0172 |                         now = time.perf_counter()
0173 |                         result["content_event_offsets_s"].append(now - started)
0174 |                         result["output"] += text
0175 |                         result["first_token_time"] = result["first_token_time"] or now
0176 |                         result["last_token_time"] = now
0177 |                         if result["ttft_ms"] is None:
0178 |                             result["ttft_ms"] = (now - started) * 1000
0179 |                             if process_origin is not None:
0180 |                                 result["process_to_first_content_s"] = now - process_origin
0181 |             ended = time.perf_counter()
0182 |             result["request_end_time"] = ended
0183 |             result["e2e_s"] = ended - started
0184 |             result["end_to_end_latency_seconds"] = ended - started
0185 |             if process_origin is not None:
0186 |                 result["process_to_response_end_s"] = ended - process_origin
0187 |             if not result["done"] or not result["output"]:
0188 |                 raise ValueError("Stream incompleto ou sem conteúdo.")
0189 |             usage = result["stream_usage"]
0190 |             valid_usage = isinstance(usage, dict) and all(
0191 |                 type(usage.get(k)) is int and usage[k] >= 0
0192 |                 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
0193 |             result["usage_observed"] = valid_usage
0194 |             if valid_usage:
0195 |                 result["completion_tokens"] = usage["completion_tokens"]
0196 |                 result["prompt_tokens"] = usage["prompt_tokens"]
0197 |                 result["total_tokens"] = usage["total_tokens"]
0198 |                 result["time_to_first_token_seconds"] = ((result["first_token_time"] - started)
0199 |                                                            if result["first_token_time"] is not None else None)
0200 |                 if result["completion_tokens"] > 1 and result.get("last_token_time") is not None:
0201 |                     result["generation_time_seconds"] = result["last_token_time"] - result["first_token_time"]
0202 |                     result["mean_itl_ms"] = 1000 * result["generation_time_seconds"] / (result["completion_tokens"] - 1)
0203 |                 if result["generation_time_seconds"] and result["generation_time_seconds"] > 0:
0204 |                     result["decode_tokens_per_second"] = result["completion_tokens"] / result["generation_time_seconds"]
0205 |                 if result["end_to_end_latency_seconds"] > 0:
0206 |                     result["end_to_end_tokens_per_second"] = result["completion_tokens"] / result["end_to_end_latency_seconds"]
0207 |                 from reporting import derived
0208 |                 result.update(derived({"output_tokens": usage["completion_tokens"], "prompt_tokens": usage["prompt_tokens"],
0209 |                                        "inter_token_latency_ms": result["mean_itl_ms"], "request_latency": result["e2e_s"]}))
0210 |             result["status"] = "complete"
0211 |     except BaseException as exc:
0212 |         result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0213 |         result["error"] = str(exc)
0214 |         if started is not None:
0215 |             result["elapsed_until_exit_s"] = time.perf_counter() - started
0216 |         raise
0217 |     finally:
0218 |         serialized = json.dumps(result, ensure_ascii=False, indent=2)
0219 |         if secret:
0220 |             serialized = serialized.replace(secret, "[REDACTED]")
0221 |         path.write_text(serialized + "\n", encoding="utf-8")
0222 |     return result
0223 |
0224 |
0225 | def lifecycle_report(output, lifecycle):
0226 |     import html
0227 |     output = Path(output)
0228 |     (output / "lifecycle.json").write_text(json.dumps(lifecycle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0229 |     rows = []
0230 |     readiness = lifecycle.get("readiness", {})
0231 |     rows.append(("Processo → API observada (s)", readiness.get("process_to_api_observed_s")))
0232 |     for phase in ("first_request", "warm_reference"):
0233 |         req = lifecycle.get(phase, {})
0234 |         for metric in ("ttft_ms", "decode_tokens_s", "effective_tokens_s", "e2e_s", "mean_itl_ms", "process_to_first_content_s", "process_to_response_end_s"):
0235 |             if phase == "warm_reference" and metric.startswith("process_"):
0236 |                 continue
0237 |             rows.append((phase + " · " + metric, req.get(metric)))
0238 |     table = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value)) if value is not None else 'Não medido'}</td></tr>" for label, value in rows)
0239 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ciclo de vida</title><style>body{{font:17px/1.7 system-ui;background:#f6f3ec;color:#193835;margin:25px}}td,th{{padding:12px;border-bottom:1px solid #ccd6cc;text-align:left}}table{{width:100%;overflow-wrap:anywhere}}a{{color:#136d58}}</style><h1>Inicialização e primeira resposta</h1><p>Modo: {html.escape(lifecycle['mode'])}. Status: {html.escape(lifecycle['status'])}.</p><p>API disponível não implica modelo na GPU. O primeiro POST é cronometrado, sem teste de geração anterior. Processo novo não implica caches de disco/CUDA frios. A referência final repete o prompt e pode aproveitar prefix caching.</p><table>{table}</table><p><a href="summary.html">Aquecimento e blocos GuideLLM</a> · <a href="lifecycle.json">Dados do ciclo de vida</a></p></html>'''
0240 |     (output / "lifecycle.html").write_text(page, encoding="utf-8")
```

## reporting.py

SHA-256: `bef2911869a34a557c48e95bc82c8a3ea618332815b5b36e7fb973e0466e118e`.

| Função/classe | Linhas |
|---|---|
| `ratio` | 10–13 |
| `percentile` | 16–23 |
| `derived` | 26–36 |
| `context_band` | 39–46 |
| `table` | 49–56 |
| `context_summary` | 59–96 |
| `gpu_summary` | 99–121 |
| `render` | 124–189 |
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
0147 |         ("generation_time_seconds_p50", "Tempo de geração p50 (s)"),
0148 |         ("decode_tokens_per_second_p50", "Tokens/s de decodificação p50"),
0149 |         ("end_to_end_tokens_per_second_p50", "Tokens/s ponta a ponta p50"),
0150 |         ("end_to_end_latency_seconds_p50", "Latência ponta a ponta p50 (s)"),
0151 |         ("input_prompt_token_count_p50", "Tokens de entrada p50"),
0152 |         ("output_completion_token_count_p50", "Tokens de saída p50")])
0153 |     by_context = table(context, [("phase", "Fase"), ("repetition", "Repetição"), ("scenario", "Cenário"), ("context_band", "Faixa de entrada (tokens)"),
0154 |         ("successful_request_count", "Requisições bem-sucedidas"),
0155 |         ("initial_context_input_token_count_p50", "Tokens de entrada inicial p50"),
0156 |         ("final_logical_context_token_count_p50", "Tokens de contexto lógico final p50"),
0157 |         ("estimated_start_logical_kv_cache_mebibytes", "KV lógico inicial estimado (MiB)"),
0158 |         ("estimated_end_logical_kv_cache_mebibytes", "KV lógico final estimado (MiB)"),
0159 |         ("request_first_token_latency_milliseconds_p50", "Latência da requisição até primeiro token p50 (ms)"),
0160 |         ("decode_generation_tokens_per_second_p50", "Velocidade de geração p50 (tokens/s)"),
0161 |         ("effective_output_tokens_per_second_p50", "Velocidade efetiva p50 (tokens/s)")])
0162 |     hardware = table(gpu, [("phase", "Fase"), ("gpu", "GPU"), ("name", "Nome"), ("samples", "Amostras"),
0163 |         ("used_mib_max", "Memória máx. (MiB)"), ("total_mib_max", "Memória total (MiB)"),
0164 |         ("gpu_util_pct_mean", "Utilização média (%)"), ("gpu_util_pct_max", "Utilização máx. (%)"),
0165 |         ("temperature_c_max", "Temperatura máx. (°C)"), ("power_w_mean", "Potência média (W)"), ("power_w_max", "Potência máx. (W)")]) if gpu else "<p>Não disponível: nenhuma amostra NVIDIA válida. Em Apple/Metal este coletor não mede GPU; isso não significa utilização zero.</p>"
0166 |     kvfile = output / "kv-cache.csv"
0167 |     kvrows = []
0168 |     if kvfile.exists():
0169 |         with kvfile.open() as handle:
0170 |             groups = defaultdict(list)
0171 |             for r in csv.DictReader(handle):
0172 |                 groups[(r["phase"], r["series"])].append(float(r["fraction"]) * 100)
0173 |             kvrows = [{"phase": p, "series": s, "n": len(v), "mean": sum(v)/len(v), "max": max(v)} for (p, s), v in groups.items()]
0174 |     kv = table(kvrows, [("phase", "Fase"), ("series", "Série do servidor"), ("n", "Amostras"), ("mean", "Ocupação média (%)"), ("max", "Ocupação máx. (%)")]) if kvrows else "<p>Ocupação real de KV não disponível nesta execução. Não foi estimada a partir da VRAM.</p>"
0175 |     first = table(initial, [("phase", "Fase"), ("ttft", "TTFT (ms)"), ("decode_tokens_s", "Geração (tokens/s)"), ("effective_tokens_s", "Efetiva (tokens/s)"), ("e2e", "Total (s)")])
0176 |     policy = manifest.get("model_availability", {}).get("policy", "Execução anterior: veja o estado inicial; ausência de download não verificada por esta versão.")
0177 |     failure = f'<p class="note">Execução não concluída: {html.escape(str(manifest["error"]))}. Dados parciais não constituem uma bateria válida.</p>' if manifest.get("error") else ""
0178 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark · latência, geração e GPU</title>
0179 | <style>body{{font:16px/1.7 system-ui;margin:32px;background:#f6f3ec;color:#193835}}main{{max-width:1400px;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border:1px solid #ccd6cc;text-align:left}}th{{background:#e0e9df}}.scroll{{overflow:auto}}h2{{margin-top:38px}}a{{color:#136d58}}.note{{padding:16px;background:#fff0de;border-left:4px solid #b54e27}}</style><main>
0180 | <h1>Um usuário · latência, geração e GPU</h1><p>Status: <strong>{html.escape(manifest.get('status', 'desconhecido'))}</strong>. {html.escape(policy)}</p>{failure}
0181 | <p class="note">TTFT = espera pelo primeiro token/conteúdo observado. Geração = (tokens de saída − 1)/(tempo entre primeiro e último token). Efetiva = tokens de saída/tempo total da requisição, incluindo TTFT. São taxas por requisição, não throughput agregado de usuários.</p>
0182 | <h2>1. Inicialização e primeira resposta</h2><p>Processo → API disponível: {html.escape(str(startup)) if startup is not None else 'não medido'} s. <a href="lifecycle.html">Ver ciclo de vida completo</a>.</p>{first}
0183 | <h2>2. Aquecimento e operação posterior</h2><p>Percentis entre requisições bem-sucedidas. Warmup e measure separados; p95 com menos de 100 sucessos é exploratório. Geração indisponível com menos de dois tokens ou intervalo não positivo.</p>{metrics}
0184 | <h2>3. Tokens/s por faixa de contexto — proxy da carga de KV</h2><p>Faixa definida pela entrada real, incluindo template, antes do decode. O contexto cresce durante a saída; mostramos também seu comprimento lógico final. Esta é uma comparação de velocidades médias de respostas iniciadas em cada faixa, não uma medição token a token dentro de faixas de ocupação física do cache.</p>{by_context}
0185 | <p>Para atenção completa, mantendo modelo, dtype de KV e uma sequência: KV lógico ≈ 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento × tokens. Pesos 4/8 bits não determinam o dtype do KV. Blocos, reserva, prefix caching e sliding window impedem tratar essa fórmula como medição de VRAM. MiB estimados só aparecem com --kv-bytes-per-token informado e verificado pelo operador; caso contrário, ficam indisponíveis.</p>
0186 | <h2>4. GPU, CPU, RAM e SSD por fase</h2><p>O monitor amostra aproximadamente 1 vez/s e relaciona cada amostra a eventos/fases. GPU vem do nvidia-smi; CPU, RAM e I/O de disco vêm do host. Máximos amostrados podem perder picos e não há atribuição por processo. N/A é ausência de dado, não zero. <a href="telemetry-summary.json">Resumo de telemetria</a> · <a href="events.csv">Eventos</a> · <a href="system.csv">CPU/RAM/SSD</a></p>{hardware}
0187 | <h2>5. Ocupação real do pool KV — vLLM</h2><p>Coleta opcional de /metrics via --collect-kv-metrics. Percentual de blocos ocupados do pool, não percentual de VRAM nem bytes. Séries/engines separados. Amostragem e atualização do servidor podem perder transientes; não sincronizada por token.</p>{kv}
0188 | <p><a href="summary.json">Resumo JSON</a> · <a href="context-summary.json">Faixas JSON</a> · <a href="gpu-summary.json">GPU JSON</a> · <a href="manifest.json">Manifesto</a></p></main></html>'''
0189 |     (output / "summary.html").write_text(page, encoding="utf-8")
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
