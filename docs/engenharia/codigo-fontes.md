# Código completo: referência linha a linha

Leia junto com [a explicação detalhada](codigo-explicado.md). Gerado dos arquivos reais; não é um segundo programa. Os números não pertencem ao Python.

## bench.py

SHA-256: `0d896f026c31034e89415090a27e5af0494063e35aada5b4a112db7822dd203f`.

| Função/classe | Linhas |
|---|---|
| `_synthetic_prompt` | 33–40 |
| `estimate_messages_tokens` | 43–47 |
| `request_sha256` | 50–52 |
| `stream_messages_request` | 55–105 |
| `stream_request` | 108–109 |
| `stream_metrics` | 112–131 |
| `run_stream_batch` | 134–146 |
| `load_conversation_fixture` | 149–162 |
| `validate_conversation_fixture_for_mode` | 165–170 |
| `fixture_messages_for_turn` | 173–187 |
| `run_conversational_batch` | 190–217 |
| `local_model_check` | 220–240 |
| `write_json` | 243–244 |
| `redact` | 247–255 |
| `capture` | 258–263 |
| `load_config` | 266–282 |
| `validate_run` | 285–298 |
| `tokenizer_digest` | 301–310 |
| `prepare_tokenizer` | 313–325 |
| `scenario_config` | 328–347 |
| `Monitor` | 350–525 |
| `percentile` | 528–534 |
| `summarize` | 537–601 |
| `requests_digest` | 604–611 |
| `turn_manifest` | 614–624 |
| `conversation_artifact` | 627–649 |
| `write_requests_csv` | 652–671 |
| `write_summary` | 674–684 |
| `run` | 687–853 |
| `positive` | 856–860 |
| `nonnegative` | 863–867 |
| `rebuild_report` | 870–884 |
| `main` | 887–952 |
| `__init__` | 352–359 |
| `set_phase` | 361–369 |
| `start` | 371–381 |
| `log` | 383–387 |
| `kv_loop` | 389–413 |
| `loop` | 415–463 |
| `stop` | 465–476 |
| `write_telemetry_summary` | 478–525 |
| `read_rows` | 480–485 |

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
0023 | from results_layout import artifact, execution_label, href, locate, prepare, runtime_root, slug
0024 |
0025 | ROOT = Path(__file__).resolve().parent
0026 | VERSION = "0.4.0"
0027 | GUIDELLM_VERSION = "0.7.4"
0028 | WORKLOADS = {"short": 256, "medium": 2048, "long": 8192}
0029 | OUTPUT_TOKENS = 128
0030 | DEFAULT_CONVERSATION_FIXTURE = ROOT / "workloads" / "conversations" / "qwen_chat_v1.json"
0031 |
0032 |
0033 | def _synthetic_prompt(tokenizer, target_tokens: int) -> str:
0034 |     """Cria texto localmente; nenhuma chamada de servidor participa do relógio."""
0035 |     seed = "Explique de forma objetiva este conceito para um estudante de estatística. "
0036 |     text = seed
0037 |     while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
0038 |         text += seed
0039 |     ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
0040 |     return tokenizer.decode(ids, skip_special_tokens=False)
0041 |
0042 |
0043 | def estimate_messages_tokens(tokenizer, messages) -> int:
0044 |     try:
0045 |         return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
0046 |     except Exception:
0047 |         return sum(len(tokenizer.encode(m.get("content", ""), add_special_tokens=False)) for m in messages)
0048 |
0049 |
0050 | def request_sha256(body) -> str:
0051 |     payload = {k: body.get(k) for k in ("messages", "max_tokens")}
0052 |     return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
0053 |
0054 |
0055 | def stream_messages_request(cfg, messages, timeout, secret="", max_tokens=OUTPUT_TOKENS, metadata=None):
0056 |     """Mede uma requisição SSE com ``time.perf_counter`` por requisição.
0057 |
0058 |     ``first_token_time``/``last_token_time`` são os instantes do primeiro/último
0059 |     evento SSE com conteúdo. A API OpenAI não promete um evento por token; por
0060 |     isso a resolução é a do evento de conteúdo, enquanto ``completion_tokens``
0061 |     vem exclusivamente de ``usage`` quando o servidor o fornece.
0062 |     """
0063 |     import httpx
0064 |     body = {"model": cfg["model"], "messages": messages,
0065 |             "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
0066 |             "stream": True, "stream_options": {"include_usage": True}}
0067 |     started = time.perf_counter()
0068 |     first = last = None
0069 |     chunks = 0
0070 |     content = ""
0071 |     usage = None
0072 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0073 |     with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
0074 |         with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as response:
0075 |             response.raise_for_status()
0076 |             if "text/event-stream" not in response.headers.get("content-type", ""):
0077 |                 raise ValueError("A API não respondeu com SSE.")
0078 |             for line in response.iter_lines():
0079 |                 if not line.startswith("data:"):
0080 |                     continue
0081 |                 value = line[5:].strip()
0082 |                 if value == "[DONE]":
0083 |                     break
0084 |                 event = json.loads(value)
0085 |                 if "error" in event:
0086 |                     raise ValueError(f"Erro no stream: {event['error']}")
0087 |                 if event.get("usage"):
0088 |                     usage = event["usage"]
0089 |                 text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
0090 |                 if text:
0091 |                     now = time.perf_counter()
0092 |                     first = first or now
0093 |                     last = now
0094 |                     chunks += 1
0095 |                     content += text
0096 |     ended = time.perf_counter()
0097 |     e2e = ended - started
0098 |     metrics = stream_metrics(started, first, last, ended, usage)
0099 |     metadata = dict(metadata or {})
0100 |     if "history_tokens_estimate" in metadata:
0101 |         metadata["history_tokens"] = metrics["prompt_tokens"] if metrics["prompt_tokens"] is not None else metadata["history_tokens_estimate"]
0102 |     return {**metrics,
0103 |             "stream_content_event_count": chunks, "output": content, "usage_observed": usage is not None,
0104 |             "request_sha256": request_sha256(body),
0105 |             "request_args": json.dumps({"body": body}, ensure_ascii=False), **metadata}
0106 |
0107 |
0108 | def stream_request(cfg, prompt, timeout, secret=""):
0109 |     return stream_messages_request(cfg, [{"role": "user", "content": prompt}], timeout, secret)
0110 |
0111 |
0112 | def stream_metrics(request_start_time, first_token_time, last_token_time, request_end_time, usage):
0113 |     """Calcula métricas exclusivamente de timestamps monotônicos do cliente."""
0114 |     usage = usage if isinstance(usage, dict) else {}
0115 |     completion = usage.get("completion_tokens") if type(usage.get("completion_tokens")) is int else None
0116 |     prompt_tokens = usage.get("prompt_tokens") if type(usage.get("prompt_tokens")) is int else None
0117 |     total = usage.get("total_tokens") if type(usage.get("total_tokens")) is int else None
0118 |     ttft = first_token_time - request_start_time if first_token_time is not None else None
0119 |     e2e = request_end_time - request_start_time
0120 |     generation = (last_token_time - first_token_time) if first_token_time is not None and last_token_time is not None and completion is not None and completion > 1 else None
0121 |     inter = generation / (completion - 1) if generation is not None else None
0122 |     decode = completion / generation if completion is not None and generation and generation > 0 else None
0123 |     effective = completion / e2e if completion is not None and e2e > 0 else None
0124 |     return {"request_start_time": request_start_time, "first_token_time": first_token_time,
0125 |             "request_end_time": request_end_time, "time_to_first_token_seconds": ttft,
0126 |             "generation_time_seconds": generation, "end_to_end_latency_seconds": e2e,
0127 |             "completion_tokens": completion, "prompt_tokens": prompt_tokens, "total_tokens": total,
0128 |             "decode_tokens_per_second": decode, "end_to_end_tokens_per_second": effective,
0129 |             "inter_token_latency_seconds": inter, "time_to_first_token_ms": ttft * 1000 if ttft is not None else None,
0130 |             "request_latency": e2e, "inter_token_latency_ms": inter * 1000 if inter is not None else None,
0131 |             "output_tokens": completion, "decode_tokens_s": decode, "effective_tokens_s": effective}
0132 |
0133 |
0134 | def run_stream_batch(cfg, tokenizer, scenario, count, timeout, secret=""):
0135 |     prompt = _synthetic_prompt(tokenizer, WORKLOADS[scenario])
0136 |     successful, errored = [], []
0137 |     for index in range(count):
0138 |         try:
0139 |             row = stream_request(cfg, prompt, timeout, secret)
0140 |             row.update({"mode": "independent", "request_id": f"{scenario}-independent-{index+1}"})
0141 |             row["status"] = "successful"
0142 |             successful.append(row)
0143 |         except Exception as exc:
0144 |             errored.append({"status": "errored", "mode": "independent",
0145 |                             "request_id": f"{scenario}-independent-{index+1}", "error": str(exc)})
0146 |     return {"benchmarks": [{"requests": {"successful": successful, "errored": errored, "incomplete": []}}]}
0147 |
0148 |
0149 | def load_conversation_fixture(path):
0150 |     path = Path(path)
0151 |     data = json.loads(path.read_text(encoding="utf-8"))
0152 |     if not isinstance(data.get("system"), str) or not data["system"].strip():
0153 |         raise ValueError("Fixture conversacional precisa de campo system textual nao vazio.")
0154 |     if not isinstance(data.get("turns"), list) or not data["turns"]:
0155 |         raise ValueError("Fixture conversacional precisa de lista nao vazia em turns.")
0156 |     for index, turn in enumerate(data["turns"], 1):
0157 |         if not isinstance(turn, dict) or not isinstance(turn.get("user"), str) or not turn["user"].strip():
0158 |             raise ValueError(f"Turno {index} do fixture precisa de user textual.")
0159 |         if "assistant" in turn and (not isinstance(turn["assistant"], str) or not turn["assistant"].strip()):
0160 |             raise ValueError(f"Turno {index} do fixture tem assistant vazio/invalido.")
0161 |     digest = hashlib.sha256(path.read_bytes()).hexdigest()
0162 |     return {"path": str(path), "sha256": digest, "data": data}
0163 |
0164 |
0165 | def validate_conversation_fixture_for_mode(fixture, loop_mode, turns):
0166 |     if loop_mode == "replay":
0167 |         missing = [index for index, turn in enumerate(fixture["data"]["turns"][:turns], 1)
0168 |                    if "assistant" not in turn]
0169 |         if missing:
0170 |             raise ValueError(f"--mode replay exige assistant fixo em cada turno usado do fixture; ausente em: {missing}.")
0171 |
0172 |
0173 | def fixture_messages_for_turn(fixture, turn_index, prior_assistant_outputs, loop_mode):
0174 |     data = fixture["data"]
0175 |     messages = []
0176 |     if data.get("system"):
0177 |         messages.append({"role": "system", "content": data["system"]})
0178 |     for index, turn in enumerate(data["turns"][:turn_index], 1):
0179 |         messages.append({"role": "user", "content": turn["user"]})
0180 |         if index < turn_index:
0181 |             if loop_mode == "closed-loop":
0182 |                 messages.append({"role": "assistant", "content": prior_assistant_outputs[index - 1]})
0183 |             else:
0184 |                 if "assistant" not in turn:
0185 |                     raise ValueError("--mode replay exige assistant fixo nos turnos anteriores do fixture.")
0186 |                 messages.append({"role": "assistant", "content": turn["assistant"]})
0187 |     return messages
0188 |
0189 |
0190 | def run_conversational_batch(cfg, tokenizer, scenario, conversations, turns, timeout, secret="", fixture=None, loop_mode="closed-loop"):
0191 |     if fixture is None:
0192 |         fixture = load_conversation_fixture(DEFAULT_CONVERSATION_FIXTURE)
0193 |     if turns > len(fixture["data"]["turns"]):
0194 |         raise ValueError(f"--conversation-turns={turns} excede turnos disponiveis no fixture ({len(fixture['data']['turns'])}).")
0195 |     validate_conversation_fixture_for_mode(fixture, loop_mode, turns)
0196 |     target = WORKLOADS[scenario]
0197 |     successful, errored = [], []
0198 |     for conversation_index in range(1, conversations + 1):
0199 |         assistant_outputs = []
0200 |         for turn_index in range(1, turns + 1):
0201 |             messages = fixture_messages_for_turn(fixture, turn_index, assistant_outputs, loop_mode)
0202 |             estimate = estimate_messages_tokens(tokenizer, messages)
0203 |             metadata = {"mode": loop_mode, "conversation_index": conversation_index,
0204 |                         "fixture_sha256": fixture["sha256"],
0205 |                         "fixture_name": fixture["data"].get("name"),
0206 |                         "turn_index": turn_index, "history_tokens_estimate": estimate,
0207 |                         "target_history_tokens": target, "messages_count": len(messages),
0208 |                         "request_id": f"{scenario}-c{conversation_index}-t{turn_index}"}
0209 |             try:
0210 |                 row = stream_messages_request(cfg, messages, timeout, secret, metadata=metadata)
0211 |                 row["status"] = "successful"
0212 |                 successful.append(row)
0213 |                 assistant_outputs.append(row.get("output") or "")
0214 |             except Exception as exc:
0215 |                 errored.append({"status": "errored", **metadata, "history_tokens": estimate, "error": str(exc)})
0216 |                 break
0217 |     return {"benchmarks": [{"requests": {"successful": successful, "errored": errored, "incomplete": []}}]}
0218 |
0219 |
0220 | def local_model_check(value):
0221 |     """Checa presença, não lê pesos nem aquece o page cache deliberadamente."""
0222 |     path = Path(value).expanduser().resolve()
0223 |     if path.is_file():
0224 |         if path.suffix not in {".safetensors", ".gguf", ".bin", ".pt", ".pth"} and not path.name.startswith("sha256-"):
0225 |             raise ValueError("Informe um arquivo de pesos (não configuração/texto) ou blob sha256- do Ollama.")
0226 |         files = [path]
0227 |     elif path.is_dir():
0228 |         files = [f for f in path.rglob("*") if f.is_file() and f.suffix in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}]
0229 |         for index in path.glob("*.index.json"):
0230 |             data = json.loads(index.read_text())
0231 |             missing = [name for name in set(data.get("weight_map", {}).values()) if not (path / name).is_file()]
0232 |             if missing:
0233 |                 raise ValueError(f"Shards ausentes: {missing}")
0234 |     else:
0235 |         files = []
0236 |     if not files or any(f.stat().st_size == 0 for f in files):
0237 |         raise ValueError("Pesos locais ausentes/vazios. Prepare o modelo antes do benchmark; downloads não são permitidos na medição.")
0238 |     return {"policy": "Pesos locais obrigatórios; download excluído do protocolo.", "path": str(path),
0239 |             "files": len(files), "bytes": sum(f.stat().st_size for f in files),
0240 |             "validation": "presença/tamanho e shards declarados; não verifica conteúdo, SSD físico ou vínculo com API"}
0241 |
0242 |
0243 | def write_json(path, data):
0244 |     Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0245 |
0246 |
0247 | def redact(value, secret):
0248 |     if isinstance(value, dict):
0249 |         return {k: ("[REDACTED]" if k.lower() in {"api_key", "authorization"} else redact(v, secret))
0250 |                 for k, v in value.items()}
0251 |     if isinstance(value, list):
0252 |         return [redact(v, secret) for v in value]
0253 |     if isinstance(value, str) and secret:
0254 |         return value.replace(secret, "[REDACTED]")
0255 |     return value
0256 |
0257 |
0258 | def capture(command):
0259 |     try:
0260 |         proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
0261 |         return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
0262 |     except (OSError, subprocess.TimeoutExpired) as exc:
0263 |         return {"unavailable": str(exc)}
0264 |
0265 |
0266 | def load_config(path):
0267 |     cfg = json.loads(Path(path).read_text(encoding="utf-8"))
0268 |     required = {"runtime", "base_url", "model", "tokenizer", "context_window", "cache_policy",
0269 |                 "runtime_version", "model_artifact", "server_command", "notes"}
0270 |     if set(cfg) != required:
0271 |         raise ValueError(f"Campos da configuração devem ser exatamente: {sorted(required)}")
0272 |     if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in required - {"context_window"}):
0273 |         raise ValueError("Os campos textuais da configuração devem estar preenchidos.")
0274 |     if type(cfg["context_window"]) is not int or cfg["context_window"] < 512:
0275 |         raise ValueError("context_window deve ser um inteiro >= 512.")
0276 |     url = urlsplit(cfg["base_url"])
0277 |     if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
0278 |         raise ValueError("URL inválida: use http(s), sem credenciais, query ou fragmento.")
0279 |     if url.path not in {"", "/", "/v1", "/v1/"}:
0280 |         raise ValueError("Use a raiz do servidor (ex.: http://127.0.0.1:8000), sem endpoint.")
0281 |     cfg["base_url"] = f"{url.scheme}://{url.netloc}"
0282 |     return cfg
0283 |
0284 |
0285 | def validate_run(cfg, scenarios, smoke):
0286 |     for scenario in scenarios:
0287 |         # Margem para o template; o servidor continua sendo a autoridade final.
0288 |         needed = WORKLOADS[scenario] + OUTPUT_TOKENS + 256
0289 |         if cfg["context_window"] < needed:
0290 |             raise ValueError(f"{scenario} precisa de contexto declarado >= {needed}; "
0291 |                              "altere o servidor e depois o JSON, ou retire esse cenário.")
0292 |     if not smoke:
0293 |         if any("PREENCHER" in cfg[k] or "SUBSTITUA" in cfg[k]
0294 |                for k in ("model", "runtime_version", "model_artifact", "server_command")):
0295 |             raise ValueError("Preencha modelo, versão, artefato e comando antes da medição formal; --smoke permite rascunhos.")
0296 |         if cfg["cache_policy"] == "runtime-default-unverified":
0297 |             raise ValueError("Registre cache_policy após verificar o servidor. Ex.: disabled-confirmed ou enabled-recorded. "
0298 |                              "O script NÃO altera nem comprova a política de cache.")
0299 |
0300 |
0301 | def tokenizer_digest(path):
0302 |     path = Path(path)
0303 |     if not path.is_dir() or not (path / "tokenizer_config.json").exists():
0304 |         raise ValueError("Tokenizer local ausente. Execute: python bench.py prepare-tokenizer")
0305 |     hashes = {}
0306 |     for file in sorted(path.rglob("*")):
0307 |         if file.is_file() and not file.name.startswith("."):
0308 |             hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
0309 |     digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
0310 |     return {"sha256": digest, "files": hashes}
0311 |
0312 |
0313 | def prepare_tokenizer(args):
0314 |     from huggingface_hub import HfApi
0315 |     from transformers import AutoTokenizer
0316 |     path = Path(args.output)
0317 |     if path.exists():
0318 |         raise ValueError(f"{path} já existe; não será sobrescrito. Use outro --output.")
0319 |     revision = HfApi().model_info(args.model, revision=args.revision).sha
0320 |     tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
0321 |     path.mkdir(parents=True)
0322 |     tokenizer.save_pretrained(path)
0323 |     write_json(path / "source.json", {"model": args.model, "revision": revision})
0324 |     print(f"Tokenizer salvo em {path}; revisão {revision}. Não foram baixados pesos.")
0325 |     print("Compartilhe esta pasta com o grupo para usar os mesmos arquivos.")
0326 |
0327 |
0328 | def scenario_config(cfg, name, count, seed, secret, timeout):
0329 |     return {
0330 |         "spec": {
0331 |             "backend": {"kind": "openai_http", "target": cfg["base_url"], "model": cfg["model"],
0332 |                         "request_format": "/v1/chat/completions", "stream": True,
0333 |                         "validate_backend": False, "verify": True, "follow_redirects": False,
0334 |                         "timeout": timeout, "api_key": secret or None,
0335 |                         "extras": {"body": {"temperature": 0, "top_p": 1}}},
0336 |             "profile": {"kind": "synchronous", "warmup": 0, "cooldown": 0},
0337 |             "constraints": [{"kind": "max_requests", "count": count}, {"kind": "max_errors", "count": 1}],
0338 |             "tokenizer": {"kind": "huggingface_auto", "model": str(Path(cfg["tokenizer"]).resolve()),
0339 |                           "load_kwargs": {"local_files_only": True, "trust_remote_code": False}},
0340 |             "data": [{"kind": "synthetic_text", "prompt_tokens": WORKLOADS[name],
0341 |                       "output_tokens": OUTPUT_TOKENS}],
0342 |             "data_loader": {"kind": "pytorch", "samples": count, "num_workers": 0, "shuffle": False},
0343 |             "seed": {"kind": "static", "value": seed},
0344 |             "metrics": {"kind": "generative", "sample_size": None, "prefer_response_metrics": True},
0345 |             "outputs": [],
0346 |         }
0347 |     }
0348 |
0349 |
0350 | class Monitor:
0351 |     """Amostragem contínua e marcação de fases; não é um profiler PCIe."""
0352 |     def __init__(self, output, cfg=None, secret="", collect_kv=False):
0353 |         self.output = Path(output)
0354 |         self.stop_event = threading.Event()
0355 |         self.thread = None
0356 |         self.phase = "setup"
0357 |         self.cfg, self.secret, self.collect_kv = cfg, secret, collect_kv
0358 |         self.kv_thread = None
0359 |         self.started_monotonic = None
0360 |
0361 |     def set_phase(self, phase, event=None):
0362 |         self.phase = phase
0363 |         utc = datetime.now(timezone.utc).isoformat()
0364 |         elapsed = (time.monotonic() - self.started_monotonic) if self.started_monotonic else 0.0
0365 |         self.log(f"[telemetria] utc={utc} decorrido={elapsed:.3f}s fase={phase} evento={event or 'phase_change'}")
0366 |         if hasattr(self, "events_handle"):
0367 |             writer = csv.writer(self.events_handle)
0368 |             writer.writerow([utc, time.monotonic(), elapsed, phase, event or "phase_change"])
0369 |             self.events_handle.flush()
0370 |
0371 |     def start(self):
0372 |         self.started_monotonic = time.monotonic()
0373 |         self.telemetry_handle = artifact(self.output, "telemetry.log").open("a", encoding="utf-8")
0374 |         self.events_handle = artifact(self.output, "events.csv").open("w", newline="", encoding="utf-8")
0375 |         csv.writer(self.events_handle).writerow(["utc", "monotonic_s", "elapsed_s", "phase", "event"])
0376 |         self.set_phase(self.phase, "monitor_started")
0377 |         self.thread = threading.Thread(target=self.loop, daemon=True)
0378 |         self.thread.start()
0379 |         if self.collect_kv:
0380 |             self.kv_thread = threading.Thread(target=self.kv_loop, daemon=True)
0381 |             self.kv_thread.start()
0382 |
0383 |     def log(self, message):
0384 |         print(message, flush=True)
0385 |         if hasattr(self, "telemetry_handle"):
0386 |             self.telemetry_handle.write(message + "\n")
0387 |             self.telemetry_handle.flush()
0388 |
0389 |     def kv_loop(self):
0390 |         import httpx
0391 |         headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
0392 |         pattern = re.compile(r'^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc))(\{[^}]*\})?\s+([0-9.eE+\-]+)(?:\s|$)')
0393 |         with artifact(self.output, "kv-cache.csv").open("w", newline="") as handle, httpx.Client(timeout=1, headers=headers, follow_redirects=False) as client:
0394 |             writer = csv.writer(handle)
0395 |             writer.writerow(["utc", "elapsed_s", "phase", "series", "fraction"])
0396 |             while not self.stop_event.is_set():
0397 |                 phase = self.phase
0398 |                 try:
0399 |                     response = client.get(self.cfg["base_url"] + "/metrics")
0400 |                     response.raise_for_status()
0401 |                     matches = [m for line in response.text.splitlines() if (m := pattern.match(line))]
0402 |                     modern = any(m[1] == "vllm:kv_cache_usage_perc" for m in matches)
0403 |                     for m in matches:
0404 |                         if modern and m[1] != "vllm:kv_cache_usage_perc":
0405 |                             continue
0406 |                         value = float(m[3])
0407 |                         if math.isfinite(value) and 0 <= value <= 1:
0408 |                             writer.writerow([datetime.now(timezone.utc).isoformat(), time.monotonic() - self.started_monotonic,
0409 |                                              phase, redact(m[1] + (m[2] or ""), self.secret), value])
0410 |                     handle.flush()
0411 |                 except (httpx.HTTPError, ValueError):
0412 |                     pass  # Serveur inicializando/endpoint ausente: nunca inventar zeros.
0413 |                 self.stop_event.wait(1)
0414 |
0415 |     def loop(self):
0416 |         try:
0417 |             import psutil
0418 |         except ImportError:
0419 |             psutil = None
0420 |         with artifact(self.output, "gpu.csv").open("w", newline="", encoding="utf-8") as handle, \
0421 |              artifact(self.output, "system.csv").open("w", newline="", encoding="utf-8") as system_handle:
0422 |             writer = csv.writer(handle)
0423 |             writer.writerow(["utc", "elapsed_s", "sample_index", "phase", "index", "name", "used_mib", "total_mib", "used_gib", "total_gib",
0424 |                              "vram_used_pct", "gpu_util_pct", "memory_util_pct", "temperature_c", "power_w"])
0425 |             system_writer = csv.writer(system_handle)
0426 |             system_writer.writerow(["utc", "elapsed_s", "sample_index", "phase", "cpu_util_pct", "ram_used_mib", "ram_available_mib", "ram_total_mib",
0427 |                                     "load1", "root_disk_used_mib", "root_disk_free_mib", "disk_read_bytes", "disk_write_bytes"])
0428 |             if psutil:
0429 |                 psutil.cpu_percent(interval=None)
0430 |             sample_number = 0
0431 |             while not self.stop_event.is_set():
0432 |                 sample_number += 1
0433 |                 phase = self.phase
0434 |                 utc = datetime.now(timezone.utc).isoformat()
0435 |                 result = capture(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
0436 |                                   "--format=csv,noheader,nounits"])
0437 |                 if result.get("returncode") != 0:
0438 |                     write_json(artifact(self.output, "gpu-unavailable.json"), result)
0439 |                 else:
0440 |                     gpu_rows = list(csv.reader(result["stdout"].splitlines(), skipinitialspace=True))
0441 |                     for row in gpu_rows:
0442 |                         used_mib, total_mib = float(row[2]), float(row[3])
0443 |                         vram_pct = (100 * used_mib / total_mib) if total_mib else None
0444 |                         writer.writerow([utc, time.monotonic() - self.started_monotonic, sample_number, phase, row[0], row[1], row[2], row[3],
0445 |                                          used_mib / 1024, total_mib / 1024, vram_pct, *row[4:]])
0446 |                     if sample_number == 1 or sample_number % 10 == 0:
0447 |                         compact = "; ".join(f"GPU{row[0]} VRAM={float(row[2]) / 1024:.2f}/{float(row[3]) / 1024:.2f} GiB "
0448 |                                              f"ocupada={100 * float(row[2]) / float(row[3]):.1f}% "
0449 |                                              f"atividade_gpu={row[4]}% atividade_leitura_escrita_memoria={row[5]}%" for row in gpu_rows)
0450 |                         elapsed = time.monotonic() - self.started_monotonic
0451 |                         utc_log = datetime.now(timezone.utc).isoformat()
0452 |                         self.log(f"[telemetria] utc={utc_log} decorrido={elapsed:.3f}s fase={phase} {compact or 'GPU sem amostra'}")
0453 |                 handle.flush()
0454 |                 if psutil:
0455 |                     vm = psutil.virtual_memory()
0456 |                     du = psutil.disk_usage(str(self.output.anchor or "/"))
0457 |                     io = psutil.disk_io_counters()
0458 |                     system_writer.writerow([utc, time.monotonic() - self.started_monotonic, sample_number, phase, psutil.cpu_percent(interval=None), vm.used / 1048576,
0459 |                                              vm.available / 1048576, vm.total / 1048576, os.getloadavg()[0],
0460 |                                              du.used / 1048576, du.free / 1048576,
0461 |                                              getattr(io, "read_bytes", None), getattr(io, "write_bytes", None)])
0462 |                     system_handle.flush()
0463 |                 self.stop_event.wait(1)
0464 |
0465 |     def stop(self):
0466 |         self.stop_event.set()
0467 |         if self.thread:
0468 |             self.thread.join(timeout=17)
0469 |         if self.kv_thread:
0470 |             self.kv_thread.join(timeout=3)
0471 |         self.set_phase("stopped", "monitor_stopped")
0472 |         if hasattr(self, "events_handle"):
0473 |             self.events_handle.close()
0474 |         if hasattr(self, "telemetry_handle"):
0475 |             self.telemetry_handle.close()
0476 |         self.write_telemetry_summary()
0477 |
0478 |     def write_telemetry_summary(self):
0479 |         """Agrega telemetria por fase para relacionar picos com eventos do benchmark."""
0480 |         def read_rows(name):
0481 |             path = locate(self.output, name)
0482 |             if not path.exists():
0483 |                 return []
0484 |             with path.open() as handle:
0485 |                 return list(csv.DictReader(handle))
0486 |         sources = {"gpu": read_rows("gpu.csv"), "system": read_rows("system.csv"), "kv": read_rows("kv-cache.csv")}
0487 |         # Arquivo de entrada simples para gráficos: uma observação por linha,
0488 |         # com UTC, tempo desde o início do monitor e fase experimental.
0489 |         if sources["gpu"]:
0490 |             import shutil
0491 |             shutil.copyfile(artifact(self.output, "gpu.csv"), artifact(self.output, "telemetry-timeseries.csv"))
0492 |         phases = sorted({r.get("phase") for rows in sources.values() for r in rows if r.get("phase")})
0493 |         output = {"definition": "Amostras observadas por fase; não são bytes nem tempos de transferência PCIe.", "phases": {}}
0494 |         for phase in phases:
0495 |             entry = {"gpu_samples": 0, "system_samples": 0, "kv_samples": 0}
0496 |             grows = [r for r in sources["gpu"] if r.get("phase") == phase]
0497 |             for key in ("used_mib", "total_mib", "used_gib", "total_gib", "vram_used_pct", "gpu_util_pct", "memory_util_pct", "temperature_c", "power_w"):
0498 |                 vals = []
0499 |                 for r in grows:
0500 |                     try: vals.append(float(r[key]))
0501 |                     except (ValueError, TypeError, KeyError): pass
0502 |                 entry[f"gpu_{key}_mean"] = sum(vals) / len(vals) if vals else None
0503 |                 entry[f"gpu_{key}_max"] = max(vals) if vals else None
0504 |                 if key in {"used_mib", "used_gib", "vram_used_pct"}:
0505 |                     entry[f"gpu_{key}_min"] = min(vals) if vals else None
0506 |             entry["gpu_samples"] = len(grows)
0507 |             srows = [r for r in sources["system"] if r.get("phase") == phase]
0508 |             for key in ("cpu_util_pct", "ram_used_mib", "ram_available_mib", "root_disk_used_mib", "root_disk_free_mib"):
0509 |                 vals = []
0510 |                 for r in srows:
0511 |                     try: vals.append(float(r[key]))
0512 |                     except (ValueError, TypeError, KeyError): pass
0513 |                 entry[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
0514 |                 entry[f"{key}_max"] = max(vals) if vals else None
0515 |             entry["system_samples"] = len(srows)
0516 |             krows = [r for r in sources["kv"] if r.get("phase") == phase]
0517 |             vals = []
0518 |             for r in krows:
0519 |                 try: vals.append(float(r["fraction"]) * 100)
0520 |                 except (ValueError, TypeError, KeyError): pass
0521 |             entry["kv_occupancy_pct_mean"] = sum(vals) / len(vals) if vals else None
0522 |             entry["kv_occupancy_pct_max"] = max(vals) if vals else None
0523 |             entry["kv_samples"] = len(vals)
0524 |             output["phases"][phase] = entry
0525 |         write_json(artifact(self.output, "telemetry-summary.json"), output)
0526 |
0527 |
0528 | def percentile(values, q):
0529 |     values = sorted(v for v in values if v is not None and math.isfinite(v))
0530 |     if not values:
0531 |         return None
0532 |     pos = (len(values) - 1) * q
0533 |     lo, hi = math.floor(pos), math.ceil(pos)
0534 |     return values[lo] + (values[hi] - values[lo]) * (pos - lo)
0535 |
0536 |
0537 | def summarize(report):
0538 |     """Métricas por requisição, sem misturar warmup ou erros com sucessos."""
0539 |     benchmarks = report["benchmarks"]
0540 |     if len(benchmarks) != 1:
0541 |         raise ValueError("Esperado exatamente um benchmark sequencial.")
0542 |     requests = benchmarks[0]["requests"]
0543 |     from reporting import derived
0544 |     good = [{**r, **derived(r)} for r in requests["successful"]]
0545 |     result = {
0546 |         "successful_request_count": len(good),
0547 |         "errored_request_count": len(requests["errored"]),
0548 |         "incomplete_request_count": len(requests["incomplete"]),
0549 |     }
0550 |     # Os nomes são deliberadamente longos: summary.json é um artefato de
0551 |     # análise, e não uma API em que economizar alguns bytes melhora algo.
0552 |     metrics = {
0553 |         "request_first_token_latency_milliseconds": "time_to_first_token_ms",
0554 |         "request_latency_seconds": "request_latency",
0555 |         "within_response_next_token_latency_milliseconds": "inter_token_latency_ms",
0556 |         "output_completion_token_count": "output_tokens",
0557 |         "input_prompt_token_count": "prompt_tokens",
0558 |         "decode_generation_tokens_per_second": "decode_tokens_s",
0559 |         "effective_output_tokens_per_second": "effective_tokens_s",
0560 |     }
0561 |     for label, key in metrics.items():
0562 |         vals = [r.get(key) for r in good]
0563 |         result[label + "_sample_count"] = sum(v is not None for v in vals)
0564 |         result[label + "_p50"] = percentile(vals, .5)
0565 |         result[label + "_p95"] = percentile(vals, .95)
0566 |         result[label + "_p99"] = percentile(vals, .99)
0567 |     # Nomes canônicos para a comparação entre runtimes. Os campos históricos
0568 |     # acima permanecem para compatibilidade com relatórios já gerados.
0569 |     ttft = [r.get("time_to_first_token_ms") for r in good]
0570 |     tokens_s = [r.get("decode_tokens_s") for r in good]
0571 |     result["time_to_first_token_milliseconds_sample_count"] = sum(v is not None for v in ttft)
0572 |     result["tokens_per_second_sample_count"] = sum(v is not None for v in tokens_s)
0573 |     for suffix, q in (("p50", .50), ("p95", .95), ("p99", .99)):
0574 |         result[f"time_to_first_token_milliseconds_{suffix}"] = percentile(ttft, q)
0575 |         result[f"tokens_per_second_{suffix}"] = percentile(tokens_s, q)
0576 |     for label, key in {
0577 |         "time_to_first_token_seconds": "time_to_first_token_seconds",
0578 |         "generation_time_seconds": "generation_time_seconds",
0579 |         "end_to_end_latency_seconds": "end_to_end_latency_seconds",
0580 |         "decode_tokens_per_second": "decode_tokens_per_second",
0581 |         "end_to_end_tokens_per_second": "end_to_end_tokens_per_second",
0582 |     }.items():
0583 |         values = [r.get(key) for r in good]
0584 |         result[label + "_sample_count"] = sum(v is not None for v in values)
0585 |         result[label + "_p50"] = percentile(values, .50)
0586 |         result[label + "_p95"] = percentile(values, .95)
0587 |         result[label + "_p99"] = percentile(values, .99)
0588 |     result["metric_definitions"] = {
0589 |         "Time To First Token": "milissegundos entre o envio da requisição e o primeiro token/conteúdo observado; há um valor por requisição.",
0590 |         "Tokens/s": "tokens de saída por segundo durante o decode, calculado por requisição a partir do intervalo entre tokens; não inclui TTFT.",
0591 |     }
0592 |     result["requests_sha256"] = requests_digest(report)
0593 |     turns = sorted({r.get("turn_index") for r in good if r.get("turn_index") is not None})
0594 |     result["turn_indices"] = turns
0595 |     result["history_tokens_by_turn"] = [{"turn_index": turn,
0596 |                                          "history_tokens_p50": percentile([r.get("history_tokens") for r in good if r.get("turn_index") == turn], .50),
0597 |                                          "history_tokens_p95": percentile([r.get("history_tokens") for r in good if r.get("turn_index") == turn], .95)}
0598 |                                         for turn in turns]
0599 |     result["percentiles_are_exploratory"] = len(good) < 100
0600 |     result["percentile_definition"] = "Empirical linear interpolation over successful requests in this phase/scenario/repetition block."
0601 |     return result
0602 |
0603 |
0604 | def requests_digest(report):
0605 |     # O hash exclui aliases do modelo e chaves: apenas carga de entrada e limite de saída.
0606 |     bodies = []
0607 |     for row in report["benchmarks"][0]["requests"]["successful"]:
0608 |         args = json.loads(row["request_args"])
0609 |         body = args.get("body", {})
0610 |         bodies.append({k: body.get(k) for k in ("messages", "max_tokens")})
0611 |     return hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
0612 |
0613 |
0614 | def turn_manifest(report):
0615 |     rows = report["benchmarks"][0]["requests"]["successful"]
0616 |     turns = []
0617 |     for row in rows:
0618 |         turns.append({"request_id": row.get("request_id"), "conversation_index": row.get("conversation_index"),
0619 |                       "turn_index": row.get("turn_index"), "mode": row.get("mode"),
0620 |                       "fixture_sha256": row.get("fixture_sha256"), "history_tokens": row.get("history_tokens"),
0621 |                       "history_tokens_estimate": row.get("history_tokens_estimate"),
0622 |                       "prompt_tokens": row.get("prompt_tokens"), "total_tokens": row.get("total_tokens"),
0623 |                       "request_sha256": row.get("request_sha256")})
0624 |     return turns
0625 |
0626 |
0627 | def conversation_artifact(report, fixture=None):
0628 |     requests = report["benchmarks"][0]["requests"]
0629 |     rows = []
0630 |     for status in ("successful", "errored", "incomplete"):
0631 |         for row in requests[status]:
0632 |             body = {}
0633 |             if row.get("request_args"):
0634 |                 body = json.loads(row["request_args"]).get("body", {})
0635 |             rows.append({"status": status, "request_id": row.get("request_id"),
0636 |                          "conversation_index": row.get("conversation_index"),
0637 |                          "turn_index": row.get("turn_index"), "mode": row.get("mode"),
0638 |                          "messages": body.get("messages"),
0639 |                          "assistant_output": row.get("output") if status == "successful" else None,
0640 |                          "error": row.get("error") if status != "successful" else None,
0641 |                          "request_sha256": row.get("request_sha256"),
0642 |                          "history_tokens": row.get("history_tokens"),
0643 |                          "history_tokens_estimate": row.get("history_tokens_estimate")})
0644 |     artifact = {"mode": rows[0].get("mode") if rows else None, "turns": rows}
0645 |     if fixture:
0646 |         artifact["fixture"] = {"path": fixture["path"], "sha256": fixture["sha256"],
0647 |                                "name": fixture["data"].get("name"),
0648 |                                "version": fixture["data"].get("version")}
0649 |     return artifact
0650 |
0651 |
0652 | def write_requests_csv(path, report):
0653 |     """Amostras individuais para análise no R/Python, incluindo status de erro."""
0654 |     from reporting import derived
0655 |     fields = ["status", "error", "mode", "request_id", "conversation_index", "turn_index",
0656 |               "fixture_name", "fixture_sha256",
0657 |               "history_tokens", "history_tokens_estimate", "target_history_tokens", "messages_count",
0658 |               "request_sha256", "request_start_time", "first_token_time", "request_end_time",
0659 |               "time_to_first_token_seconds", "generation_time_seconds", "end_to_end_latency_seconds",
0660 |               "completion_tokens", "prompt_tokens", "total_tokens", "decode_tokens_per_second",
0661 |               "end_to_end_tokens_per_second", "inter_token_latency_seconds", "request_latency",
0662 |               "time_to_first_token_ms", "inter_token_latency_ms", "output_tokens", "decode_tokens_s", "effective_tokens_s",
0663 |               "context_start_tokens", "context_end_tokens", "context_band"]
0664 |     with Path(path).open("w", newline="", encoding="utf-8") as handle:
0665 |         writer = csv.DictWriter(handle, fieldnames=fields)
0666 |         writer.writeheader()
0667 |         for status in ("successful", "errored", "incomplete"):
0668 |             for row in report["benchmarks"][0]["requests"][status]:
0669 |                 if status == "successful":
0670 |                     row = {**row, **derived(row)}
0671 |                 writer.writerow({"status": status, **{key: row.get(key) for key in fields[1:]}})
0672 |
0673 |
0674 | def write_summary(output, rows):
0675 |     from reporting import render
0676 |     write_json(artifact(output, "summary.json"), rows)
0677 |     if not rows:
0678 |         render(output, rows)
0679 |         return
0680 |     with artifact(output, "summary.csv").open("w", newline="", encoding="utf-8") as handle:
0681 |         writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
0682 |         writer.writeheader()
0683 |         writer.writerows(rows)
0684 |     render(output, rows)
0685 |
0686 |
0687 | def run(args):
0688 |     import fcntl
0689 |     from lifecycle import DEFAULT_PROMPT, Launch, lifecycle_report, timed_request, wait_models
0690 |     if importlib.metadata.version("guidellm") != GUIDELLM_VERSION:
0691 |         raise ValueError(f"Este projeto exige guidellm=={GUIDELLM_VERSION}; reinstale requirements.txt.")
0692 |     cfg = load_config(args.config)
0693 |     validate_run(cfg, args.scenarios, args.smoke)
0694 |     availability = local_model_check(args.local_model_path)
0695 |     availability["kv_bytes_per_token"] = args.kv_bytes_per_token
0696 |     availability["launch_hf_offline"] = bool(args.launch)
0697 |     digest = tokenizer_digest(cfg["tokenizer"])
0698 |     from transformers import AutoTokenizer
0699 |     measurement_tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True, trust_remote_code=False)
0700 |     count, repetitions = (3, 1) if args.smoke else (args.requests, args.repetitions)
0701 |     secret = os.environ.get("BENCH_API_KEY", "")
0702 |     prompt = Path(args.first_prompt_file).read_text(encoding="utf-8") if args.first_prompt_file else DEFAULT_PROMPT
0703 |     if not prompt.strip():
0704 |         raise ValueError("O prompt inicial não pode estar vazio.")
0705 |     conversation_mode = args.mode in {"closed-loop", "replay"}
0706 |     if not conversation_mode and args.conversation_turns != 1:
0707 |         raise ValueError("--conversation-turns só pode ser maior que 1 com --mode closed-loop ou --mode replay.")
0708 |     conversation_fixture = load_conversation_fixture(args.conversation_fixture) if conversation_mode else None
0709 |     if conversation_fixture:
0710 |         validate_conversation_fixture_for_mode(conversation_fixture, args.mode, args.conversation_turns)
0711 |     base = Path(args.results)
0712 |     base.mkdir(parents=True, exist_ok=True)
0713 |     with (base / ".benchmark.lock").open("a") as lock:
0714 |         try:
0715 |             fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
0716 |         except BlockingIOError:
0717 |             raise ValueError("Já existe um benchmark usando esta pasta results. Não execute dois ao mesmo tempo.") from None
0718 |         stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
0719 |         label = execution_label(smoke=args.smoke, scenarios=args.scenarios,
0720 |                                 input_tokens=args.input_tokens, result_name=args.result_name)
0721 |         output = runtime_root(base, cfg["runtime"], stamp, label)
0722 |         prepare(output)
0723 |         manifest = {"project_version": VERSION, "guidellm_version": GUIDELLM_VERSION, "started_utc": stamp,
0724 |                     "results_layout": "runtime/timestamp/human-name/{html,json,csv,logs,text}",
0725 |                     "result_name": label, "runtime_directory": slug(cfg["runtime"]),
0726 |                     "config": cfg, "tokenizer": digest, "smoke": args.smoke, "requests": count,
0727 |                     "repetitions": repetitions, "warmup_requests_per_case": args.warmup,
0728 |                     "scenarios": args.scenarios, "seed": args.seed, "profile": "synchronous",
0729 |                     "mode": args.mode, "conversation_turns": args.conversation_turns,
0730 |                     "conversation_fixture": {"path": conversation_fixture["path"], "sha256": conversation_fixture["sha256"],
0731 |                                              "name": conversation_fixture["data"].get("name"),
0732 |                                              "version": conversation_fixture["data"].get("version")} if conversation_fixture else None,
0733 |                     "conversation_first_request": "first-request.json is a separate single-turn lifecycle probe; closed-loop/replay measured histories start empty per conversation and per block",
0734 |                     "blocks": [],
0735 |                     "model_availability": availability,
0736 |                     "guidellm_compat": "0.7.4 bounded drain of real late completion updates (5s); no request retry",
0737 |                     "telemetry": {"gpu_source": "local nvidia-smi", "system_source": "psutil host CPU/RAM/disk counters",
0738 |                                   "event_source": "events.csv phase markers", "kv_metrics_requested": args.collect_kv_metrics,
0739 |                                   "sampling": "approximately 1 Hz; not per-token; PCIe copy time is not directly measured"},
0740 |                     "python": sys.version, "platform": platform.platform(),
0741 |                     "git": capture(["git", "rev-parse", "HEAD"]), "status": "running"}
0742 |         write_json(artifact(output, "manifest.json"), redact(manifest, secret))
0743 |         write_json(artifact(output, "client-packages.json"), {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
0744 |         write_json(artifact(output, "gpu-before.json"), capture(["nvidia-smi"]))
0745 |         monitor, rows = Monitor(output, cfg, secret, args.collect_kv_metrics), []
0746 |         launch, origin, cleanup_error = None, None, None
0747 |         lifecycle = {"mode": "new-process" if args.launch else "existing-server-state-unknown",
0748 |                      "status": "running", "initial_state_note": args.initial_state,
0749 |                      "startup_timeout_s": args.startup_timeout}
0750 |         print(f"Resultados: {output.resolve()}", flush=True)
0751 |         try:
0752 |             monitor.start()
0753 |             if args.launch:
0754 |                 launch = Launch(cfg, args.launch, output, args.launch_extra_args, args.launch_executable)
0755 |                 lifecycle["argv"] = redact(launch.argv, secret)
0756 |                 monitor.set_phase("process_startup")
0757 |                 origin = launch.start()
0758 |                 lifecycle["pid"] = launch.process.pid
0759 |             lifecycle_report(output, redact(lifecycle, secret))
0760 |             monitor.set_phase("server_readiness")
0761 |             lifecycle["readiness"] = wait_models(cfg, secret, args.startup_timeout, launch)
0762 |             lifecycle_report(output, redact(lifecycle, secret))
0763 |             monitor.set_phase("first_request")
0764 |             print("Primeiro POST: medição da primeira resposta (nenhuma geração prévia enviada pelo cliente).", flush=True)
0765 |             lifecycle["first_request"] = timed_request(cfg, secret, args.timeout, prompt,
0766 |                                                        artifact(output, "first-request.json"), origin)
0767 |             lifecycle_report(output, redact(lifecycle, secret))
0768 |             for rep in range(repetitions):
0769 |                 # Rotação balanceia parcialmente a posição dos cenários entre repetições.
0770 |                 names = args.scenarios[rep % len(args.scenarios):] + args.scenarios[:rep % len(args.scenarios)]
0771 |                 for name in names:
0772 |                     for phase, n in (("warmup", args.warmup), ("measure", count)):
0773 |                         if not n:
0774 |                             continue
0775 |                         seed = args.seed + rep * 100 + list(WORKLOADS).index(name)
0776 |                         if phase == "warmup":
0777 |                             seed += 1_000_000
0778 |                         prefix = f"r{rep+1}-{name}-{phase}"
0779 |                         monitor.set_phase(prefix)
0780 |                         config = scenario_config(cfg, name, n, seed, secret, args.timeout)
0781 |                         config["mode"] = args.mode
0782 |                         if conversation_mode:
0783 |                             config["conversation_turns"] = args.conversation_turns
0784 |                             config["conversation_fixture_sha256"] = conversation_fixture["sha256"]
0785 |                             config["spec_note"] = "GuideLLM synthetic data spec is retained for independent baseline context only; measured conversational requests are generated from the fixed fixture with accumulated chat history."
0786 |                         write_json(artifact(output, f"{prefix}-config.json"), redact(config, secret))
0787 |                         expected = n * args.conversation_turns if conversation_mode else n
0788 |                         unit = "conversas" if conversation_mode else "requisições"
0789 |                         print(f"{prefix}: {n} {unit}, uma chamada por vez", flush=True)
0790 |                         if conversation_mode:
0791 |                             report = run_conversational_batch(cfg, measurement_tokenizer, name, n, args.conversation_turns,
0792 |                                                               args.timeout, secret, conversation_fixture, args.mode)
0793 |                         else:
0794 |                             report = run_stream_batch(cfg, measurement_tokenizer, name, n, args.timeout, secret)
0795 |                         raw = redact(report, secret)
0796 |                         write_json(artifact(output, f"{prefix}.json"), raw)
0797 |                         turns = turn_manifest(raw) if conversation_mode else []
0798 |                         conversation_file = None
0799 |                         if conversation_mode:
0800 |                             write_json(artifact(output, f"{prefix}-turns.json"), turns)
0801 |                             conversation_file = f"{prefix}-conversation.json"
0802 |                             write_json(artifact(output, conversation_file), conversation_artifact(raw, conversation_fixture))
0803 |                         write_requests_csv(artifact(output, f"{prefix}-requests.csv"), raw)
0804 |                         summary = summarize(raw)
0805 |                         summary["expected"] = expected
0806 |                         summary["missing_request_count"] = max(0, expected - sum(summary[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count")))
0807 |                         manifest["blocks"].append({"prefix": prefix, "mode": args.mode, "phase": phase,
0808 |                                                    "scenario": name, "repetition": rep+1,
0809 |                                                    "conversations": n if conversation_mode else None,
0810 |                                                    "conversation_turns": args.conversation_turns if conversation_mode else None,
0811 |                                                    "expected_requests": expected,
0812 |                                                    "requests_sha256": summary["requests_sha256"],
0813 |                                                    "conversation_artifact": conversation_file,
0814 |                                                    "turns": turns if conversation_mode else None})
0815 |                         write_json(artifact(output, "manifest.json"), redact(manifest, secret))
0816 |                         rows.append({"runtime": cfg["runtime"], "model": cfg["model"],
0817 |                                      "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
0818 |                                      "mode": args.mode, "phase": phase, "scenario": name, "repetition": rep+1, **summary})
0819 |                         write_summary(output, rows)
0820 |                         if summary["successful_request_count"] != expected or summary["errored_request_count"] or summary["incomplete_request_count"]:
0821 |                             raise RuntimeError(f"{prefix}: requisições falharam ou execução incompleta. Veja o JSON; não compare como sucesso.")
0822 |             monitor.set_phase("warm_reference")
0823 |             lifecycle["warm_reference"] = timed_request(cfg, secret, args.timeout, prompt,
0824 |                                                          artifact(output, "warm-reference.json"))
0825 |             manifest["status"] = lifecycle["status"] = "complete"
0826 |         except BaseException as exc:
0827 |             manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0828 |             manifest["error"] = redact(str(exc), secret)
0829 |             lifecycle["status"] = manifest["status"]
0830 |             lifecycle["error"] = manifest["error"]
0831 |             raise
0832 |         finally:
0833 |             for key, filename in (("first_request", "first-request.json"), ("warm_reference", "warm-reference.json")):
0834 |                 if artifact(output, filename).exists():
0835 |                     lifecycle[key] = json.loads(artifact(output, filename).read_text(encoding="utf-8"))
0836 |             if launch is not None:
0837 |                 monitor.set_phase("server_shutdown")
0838 |                 try:
0839 |                     launch.close()
0840 |                     lifecycle["server_cleanup"] = "stopped owned process group only"
0841 |                 except Exception as exc:
0842 |                     cleanup_error = redact(str(exc), secret)
0843 |                     lifecycle["server_cleanup_error"] = cleanup_error
0844 |                     lifecycle["status"] = manifest["status"] = "failed"
0845 |             lifecycle_report(output, redact(lifecycle, secret))
0846 |             monitor.stop()
0847 |             manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
0848 |             write_json(artifact(output, "manifest.json"), redact(manifest, secret))
0849 |             write_json(artifact(output, "gpu-after.json"), capture(["nvidia-smi"]))
0850 |             write_summary(output, rows)
0851 |         if cleanup_error:
0852 |             raise RuntimeError(f"Falha ao encerrar processo criado: {cleanup_error}. Confira o PID no lifecycle.json.")
0853 |         print(f"Concluído. Abra {artifact(output, 'lifecycle.html')} e {artifact(output, 'summary.html')}")
0854 |
0855 |
0856 | def positive(value):
0857 |     number = int(value)
0858 |     if number <= 0:
0859 |         raise argparse.ArgumentTypeError("Use um inteiro positivo.")
0860 |     return number
0861 |
0862 |
0863 | def nonnegative(value):
0864 |     number = int(value)
0865 |     if number < 0:
0866 |         raise argparse.ArgumentTypeError("Use zero ou um inteiro positivo.")
0867 |     return number
0868 |
0869 |
0870 | def rebuild_report(args):
0871 |     """Atualiza apenas derivados, preservando relatórios brutos e manifesto original."""
0872 |     output = Path(args.output)
0873 |     rows = json.loads(locate(output, "summary.json").read_text())
0874 |     manifest = json.loads(locate(output, "manifest.json").read_text())
0875 |     for row in rows:
0876 |         prefix = f"r{row['repetition']}-{row['scenario']}-{row['phase']}"
0877 |         raw = json.loads(locate(output, f"{prefix}.json").read_text())
0878 |         row.update(summarize(raw))
0879 |         expected = manifest.get("warmup_requests_per_case") if row["phase"] == "warmup" else manifest.get("requests")
0880 |         row["expected"] = expected
0881 |         row["missing_request_count"] = max(0, expected - sum(row[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count"))) if expected is not None else None
0882 |         write_requests_csv(artifact(output, f"{prefix}-requests.csv"), raw)
0883 |     write_summary(output, rows)
0884 |     print(f"Relatório atualizado: {artifact(output, 'summary.html')}; dados brutos preservados.")
0885 |
0886 |
0887 | def main():
0888 |     parser = argparse.ArgumentParser(description=__doc__)
0889 |     commands = parser.add_subparsers(dest="command", required=True)
0890 |     report = commands.add_parser("report", help="Regenera derivados de uma execução existente, sem nova inferência.")
0891 |     report.add_argument("--output", required=True)
0892 |     report.set_defaults(func=rebuild_report)
0893 |     prep = commands.add_parser("prepare-tokenizer", help="Baixa apenas tokenizer; fixa revisão e guarda origem.")
0894 |     prep.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
0895 |     prep.add_argument("--revision", default="main")
0896 |     prep.add_argument("--output", default="tokenizer")
0897 |     prep.set_defaults(func=prepare_tokenizer)
0898 |     cmd = commands.add_parser("run", help="Mede primeiro acesso, aquecimento e GuideLLM; lançamento do servidor é opcional.")
0899 |     cmd.add_argument("--config", required=True)
0900 |     cmd.add_argument("--local-model-path", required=True, help="Pesos já no SSD: pasta HF, arquivo GGUF ou blob local do Ollama. Não baixa arquivos.")
0901 |     cmd.add_argument("--collect-kv-metrics", action="store_true", help="Amostra /metrics do vLLM (~1 Hz); ocupação do pool KV, não bytes.")
0902 |     cmd.add_argument("--kv-bytes-per-token", type=positive, help="Opcional: bytes de KV lógico por token, calculados para arquitetura/dtype reais. Estimativa, não VRAM medida.")
0903 |     cmd.add_argument("--input-tokens", nargs="+", type=positive, help="Substitui --scenarios por uma grade de comprimentos sintéticos, ex.: 256 512 1024 2048 3072.")
0904 |     cmd.add_argument("--scenarios", nargs="+", choices=list(WORKLOADS), default=["short", "medium", "long"])
0905 |     cmd.add_argument("--requests", type=positive, default=50, help="Requisições de medição por cenário e repetição; 50×3 dá 150 sucessos para percentis.")
0906 |     cmd.add_argument("--repetitions", type=positive, default=3)
0907 |     cmd.add_argument("--warmup", type=nonnegative, default=3)
0908 |     cmd.add_argument("--mode", choices=["independent", "closed-loop", "replay"], default="independent",
0909 |                      help="independent preserva uma requisição sem histórico; closed-loop acumula respostas reais; replay usa assistant fixo do fixture.")
0910 |     cmd.add_argument("--conversation-turns", type=positive, default=1,
0911 |                      help="Turnos por conversa em --mode closed-loop ou replay; cada turno envia o histórico completo.")
0912 |     cmd.add_argument("--conversation-fixture", default=str(DEFAULT_CONVERSATION_FIXTURE),
0913 |                      help="Fixture JSON versionado com system e lista fixa de user turns; replay tambem exige assistant nos turnos anteriores.")
0914 |     cmd.add_argument("--seed", type=positive, default=42)
0915 |     cmd.add_argument("--timeout", type=positive, default=300)
0916 |     cmd.add_argument("--results", default="results")
0917 |     cmd.add_argument("--result-name", help="Nome humano da execução na pasta timestamp; sem isso é derivado do modo/cenário.")
0918 |     cmd.add_argument("--smoke", action="store_true", help="3 medições e 1 repetição; não vale como resultado final.")
0919 |     cmd.add_argument("--launch", help="Arquivo JSON com argv para iniciar um runtime LOCAL; encerra só esse processo ao final.")
0920 |     cmd.add_argument("--launch-extra-args", nargs="*", default=[],
0921 |                      help="Argumentos experimentais acrescentados ao argv do launch, sem editar o JSON; registrados no lifecycle.")
0922 |     cmd.add_argument("--launch-executable", help="Substitui argv[0] do launch pelo executável resolvido no ambiente do runtime.")
0923 |     cmd.add_argument("--launch-extra-args-json", default="[]", help="Array JSON de argumentos do runtime, preservando flags e espaços.")
0924 |     cmd.add_argument("--startup-timeout", type=positive, default=1800, help="Limite da espera pela API com --launch, em segundos.")
0925 |     cmd.add_argument("--first-prompt-file", help="Texto UTF-8 para a primeira requisição e referência final; default: pergunta sobre RAM/VRAM.")
0926 |     cmd.add_argument("--initial-state", default="weights local; OS/compilation caches not controlled", help="Descreva SSD e caches existentes; apenas registra, não limpa.")
0927 |     cmd.set_defaults(func=run)
0928 |     args = parser.parse_args()
0929 |     if hasattr(args, "launch_extra_args_json"):
0930 |         extra = json.loads(args.launch_extra_args_json)
0931 |         if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
0932 |             parser.error("--launch-extra-args-json deve ser array de strings")
0933 |         args.launch_extra_args.extend(extra)
0934 |     if getattr(args, "input_tokens", None):
0935 |         if len(set(args.input_tokens)) != len(args.input_tokens):
0936 |             parser.error("Não repita comprimentos em --input-tokens.")
0937 |         args.scenarios = []
0938 |         for size in args.input_tokens:
0939 |             name = f"ctx{size}"
0940 |             WORKLOADS[name] = size
0941 |             args.scenarios.append(name)
0942 |     if hasattr(args, "scenarios") and len(set(args.scenarios)) != len(args.scenarios):
0943 |         parser.error("Não repita cenários na lista.")
0944 |     try:
0945 |         args.func(args)
0946 |     except KeyboardInterrupt:
0947 |         print("Interrompido; resultados já concluídos foram preservados.", file=sys.stderr)
0948 |         return 130
0949 |     except Exception as exc:
0950 |         print(f"ERRO: {redact(str(exc), os.environ.get('BENCH_API_KEY', ''))}", file=sys.stderr)
0951 |         return 1
0952 |     return 0
0953 |
0954 |
0955 | if __name__ == "__main__":
0956 |     raise SystemExit(main())
```

## lifecycle.py

SHA-256: `07492c79450e9aca442b814081443c5a51fd69e8c66706fb8d188c1234310269`.

| Função/classe | Linhas |
|---|---|
| `Launch` | 25–83 |
| `wait_models` | 86–127 |
| `timed_request` | 130–224 |
| `lifecycle_report` | 227–242 |
| `__init__` | 26–43 |
| `start` | 45–63 |
| `close` | 65–83 |

```text
0001 | """Cronometria de inicialização/primeiro stream, fora das fases GuideLLM.
0002 |
0003 | Nenhum POST de inferência é enviado antes da primeira requisição medida.
0004 | O lançamento é opt-in e usa argv sem shell. Só o processo criado é encerrado.
0005 | """
0006 | import json
0007 | import os
0008 | from pathlib import Path
0009 |
0010 | from results_layout import artifact, href
0011 | import signal
0012 | import socket
0013 | import subprocess
0014 | import time
0015 | from urllib.parse import urlsplit
0016 |
0017 | import httpx
0018 |
0019 | DEFAULT_PROMPT = (
0020 |     "Explique para um estudante de estatística a diferença entre memória RAM e VRAM. "
0021 |     "Inclua um exemplo de uso de um chatbot e conclua com um resumo em três itens."
0022 | )
0023 |
0024 |
0025 | class Launch:
0026 |     def __init__(self, cfg, command_file, output, extra_args=None, executable=None):
0027 |         url = urlsplit(cfg["base_url"])
0028 |         if url.hostname not in {"127.0.0.1", "localhost", "::1"}:
0029 |             raise ValueError("--launch só aceita servidor local (localhost).")
0030 |         self.host, self.port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
0031 |         self.argv = json.loads(Path(command_file).read_text(encoding="utf-8"))
0032 |         if not isinstance(self.argv, list) or not self.argv or any(not isinstance(x, str) or not x for x in self.argv):
0033 |             raise ValueError("O arquivo --launch deve conter um array JSON não vazio de strings (argv).")
0034 |         if executable:
0035 |             self.argv[0] = executable
0036 |         if extra_args:
0037 |             if any(not isinstance(x, str) or not x for x in extra_args):
0038 |                 raise ValueError("--launch-extra-args aceita somente strings não vazias.")
0039 |             self.argv.extend(extra_args)
0040 |         self.output = Path(output)
0041 |         self.process = None
0042 |         self.log = None
0043 |         self.started = None
0044 |
0045 |     def start(self):
0046 |         # Não interrompe servidores existentes nem tenta tomar uma porta ocupada.
0047 |         try:
0048 |             connection = socket.create_connection((self.host, self.port), timeout=1)
0049 |         except OSError:
0050 |             pass
0051 |         else:
0052 |             connection.close()
0053 |             raise ValueError("A porta já está em uso. Pare o seu servidor manualmente antes de usar --launch.")
0054 |         self.log = artifact(self.output, "server.log").open("w", encoding="utf-8")
0055 |         self.started = time.perf_counter()
0056 |         try:
0057 |             self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
0058 |                                             start_new_session=True, shell=False,
0059 |                                             env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
0060 |         except BaseException:
0061 |             self.log.close()
0062 |             raise
0063 |         return self.started
0064 |
0065 |     def close(self):
0066 |         """Encerra exclusivamente o grupo criado por este objeto, inclusive filhos."""
0067 |         if self.process is not None:
0068 |             try:
0069 |                 os.killpg(self.process.pid, signal.SIGTERM)
0070 |             except ProcessLookupError:
0071 |                 pass
0072 |             try:
0073 |                 self.process.wait(timeout=20)
0074 |             except subprocess.TimeoutExpired:
0075 |                 os.killpg(self.process.pid, signal.SIGKILL)
0076 |                 self.process.wait(timeout=5)
0077 |             # Alguns workers podem sobreviver ao encerramento do processo líder.
0078 |             try:
0079 |                 os.killpg(self.process.pid, signal.SIGKILL)
0080 |             except ProcessLookupError:
0081 |                 pass
0082 |         if self.log is not None:
0083 |             self.log.close()
0084 |
0085 |
0086 | def wait_models(cfg, secret, timeout, launch=None):
0087 |     """Apenas GET. Modelo listado não comprova que seus pesos estão na GPU."""
0088 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0089 |     started = time.perf_counter()
0090 |     probes, last, next_update = 0, None, started + 15
0091 |     with httpx.Client(timeout=2, headers=headers, follow_redirects=False) as client:
0092 |         while True:
0093 |             probes += 1
0094 |             if launch is not None and launch.process.poll() is not None:
0095 |                 raise RuntimeError(f"Servidor encerrou antes de ficar disponível (exit={launch.process.returncode}). Veja server.log.")
0096 |             try:
0097 |                 response = client.get(cfg["base_url"] + "/v1/models")
0098 |                 if response.status_code in {401, 403}:
0099 |                     raise ValueError("API recusou autenticação; confira BENCH_API_KEY.")
0100 |                 response.raise_for_status()
0101 |                 models = [m["id"] for m in response.json().get("data", [])]
0102 |                 expected = cfg["model"]
0103 |                 accepted = {expected}
0104 |                 if cfg.get("runtime") == "ollama" and ":" not in expected:
0105 |                     accepted.add(expected + ":latest")
0106 |                 if accepted.intersection(models):
0107 |                     observed = time.perf_counter()
0108 |                     return {"models": models, "get_probes": probes,
0109 |                             "wait_wall_s": observed - started,
0110 |                             "process_to_api_observed_s": observed - launch.started if launch else None,
0111 |                             "criterion": "GET /v1/models retornou o ID; não é prova de pesos residentes"}
0112 |                 last = f"API respondeu, mas modelo esperado={expected!r}; disponíveis={models!r}"
0113 |                 if models:
0114 |                     raise RuntimeError(last + ". Confira --alias/--served-model-name e config.model.")
0115 |             except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
0116 |                 last = str(exc)
0117 |             elapsed = time.perf_counter() - started
0118 |             if not launch or elapsed >= timeout:
0119 |                 raise RuntimeError(f"API/modelo não disponível após {elapsed:.1f}s: {last}")
0120 |             if time.perf_counter() >= next_update:
0121 |                 print(f"[inicialização] runtime={cfg.get('runtime', 'não informado')} "
0122 |                       f"PID={launch.process.pid if launch else 'externo'} decorrido={elapsed:.1f}s "
0123 |                       f"limite={timeout}s; verificando GET {cfg['base_url']}/v1/models "
0124 |                       f"para modelo={cfg['model']!r}; última observação: {last}. "
0125 |                       f"Log do servidor: {artifact(launch.output, 'server.log') if launch else 'externo'}", flush=True)
0126 |                 next_update = time.perf_counter() + 15
0127 |             time.sleep(min(.5, max(0, timeout - elapsed)))
0128 |
0129 |
0130 | def timed_request(cfg, secret, timeout, prompt, output, process_origin=None):
0131 |     """Primeiro POST é simultaneamente medição e validação, sem pré-aquecimento oculto.
0132 |
0133 |     Grava resultado parcial inclusive em timeout, stream inválido ou usage ausente.
0134 |     TTFT aqui é primeiro conteúdo não vazio recebido (não mero cabeçalho/role).
0135 |     """
0136 |     path = Path(output)
0137 |     body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
0138 |             "temperature": 0, "top_p": 1, "max_tokens": 128, "stream": True,
0139 |             "stream_options": {"include_usage": True}}
0140 |     result = {"status": "running", "body": body, "stream_usage": None,
0141 |               "content_event_offsets_s": [], "output": "", "done": False,
0142 |               "ttft_ms": None, "e2e_s": None, "mean_itl_ms": None,
0143 |               "usage_observed": False, "request_start_time": None,
0144 |               "first_token_time": None, "request_end_time": None,
0145 |               "time_to_first_token_seconds": None, "generation_time_seconds": None,
0146 |               "end_to_end_latency_seconds": None, "completion_tokens": None,
0147 |               "prompt_tokens": None, "total_tokens": None,
0148 |               "decode_tokens_per_second": None, "end_to_end_tokens_per_second": None,
0149 |               "process_to_first_content_s": None, "process_to_response_end_s": None}
0150 |     headers = {"Authorization": f"Bearer {secret}"} if secret else {}
0151 |     started = None
0152 |     try:
0153 |         with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
0154 |             started = time.perf_counter()
0155 |             result["request_start_time"] = started
0156 |             with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as stream:
0157 |                 result["headers_ms"] = (time.perf_counter() - started) * 1000
0158 |                 stream.raise_for_status()
0159 |                 if "text/event-stream" not in stream.headers.get("content-type", ""):
0160 |                     raise ValueError("A API não respondeu com SSE.")
0161 |                 for line in stream.iter_lines():
0162 |                     if not line.startswith("data:"):
0163 |                         continue
0164 |                     value = line[5:].strip()
0165 |                     if value == "[DONE]":
0166 |                         result["done"] = True
0167 |                         break
0168 |                     event = json.loads(value)
0169 |                     if "error" in event:
0170 |                         raise ValueError(f"Erro no stream: {event['error']}")
0171 |                     result["stream_usage"] = event.get("usage") or result["stream_usage"]
0172 |                     text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
0173 |                     if text:
0174 |                         now = time.perf_counter()
0175 |                         result["content_event_offsets_s"].append(now - started)
0176 |                         result["output"] += text
0177 |                         result["first_token_time"] = result["first_token_time"] or now
0178 |                         result["last_token_time"] = now
0179 |                         if result["ttft_ms"] is None:
0180 |                             result["ttft_ms"] = (now - started) * 1000
0181 |                             if process_origin is not None:
0182 |                                 result["process_to_first_content_s"] = now - process_origin
0183 |             ended = time.perf_counter()
0184 |             result["request_end_time"] = ended
0185 |             result["e2e_s"] = ended - started
0186 |             result["end_to_end_latency_seconds"] = ended - started
0187 |             if process_origin is not None:
0188 |                 result["process_to_response_end_s"] = ended - process_origin
0189 |             if not result["done"] or not result["output"]:
0190 |                 raise ValueError("Stream incompleto ou sem conteúdo.")
0191 |             usage = result["stream_usage"]
0192 |             valid_usage = isinstance(usage, dict) and all(
0193 |                 type(usage.get(k)) is int and usage[k] >= 0
0194 |                 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
0195 |             result["usage_observed"] = valid_usage
0196 |             if valid_usage:
0197 |                 result["completion_tokens"] = usage["completion_tokens"]
0198 |                 result["prompt_tokens"] = usage["prompt_tokens"]
0199 |                 result["total_tokens"] = usage["total_tokens"]
0200 |                 result["time_to_first_token_seconds"] = ((result["first_token_time"] - started)
0201 |                                                            if result["first_token_time"] is not None else None)
0202 |                 if result["completion_tokens"] > 1 and result.get("last_token_time") is not None:
0203 |                     result["generation_time_seconds"] = result["last_token_time"] - result["first_token_time"]
0204 |                     result["mean_itl_ms"] = 1000 * result["generation_time_seconds"] / (result["completion_tokens"] - 1)
0205 |                 if result["generation_time_seconds"] and result["generation_time_seconds"] > 0:
0206 |                     result["decode_tokens_per_second"] = result["completion_tokens"] / result["generation_time_seconds"]
0207 |                 if result["end_to_end_latency_seconds"] > 0:
0208 |                     result["end_to_end_tokens_per_second"] = result["completion_tokens"] / result["end_to_end_latency_seconds"]
0209 |                 from reporting import derived
0210 |                 result.update(derived({"output_tokens": usage["completion_tokens"], "prompt_tokens": usage["prompt_tokens"],
0211 |                                        "inter_token_latency_ms": result["mean_itl_ms"], "request_latency": result["e2e_s"]}))
0212 |             result["status"] = "complete"
0213 |     except BaseException as exc:
0214 |         result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
0215 |         result["error"] = str(exc)
0216 |         if started is not None:
0217 |             result["elapsed_until_exit_s"] = time.perf_counter() - started
0218 |         raise
0219 |     finally:
0220 |         serialized = json.dumps(result, ensure_ascii=False, indent=2)
0221 |         if secret:
0222 |             serialized = serialized.replace(secret, "[REDACTED]")
0223 |         path.write_text(serialized + "\n", encoding="utf-8")
0224 |     return result
0225 |
0226 |
0227 | def lifecycle_report(output, lifecycle):
0228 |     import html
0229 |     output = Path(output)
0230 |     artifact(output, "lifecycle.json").write_text(json.dumps(lifecycle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
0231 |     rows = []
0232 |     readiness = lifecycle.get("readiness", {})
0233 |     rows.append(("Processo → API observada (s)", readiness.get("process_to_api_observed_s")))
0234 |     for phase in ("first_request", "warm_reference"):
0235 |         req = lifecycle.get(phase, {})
0236 |         for metric in ("ttft_ms", "decode_tokens_s", "effective_tokens_s", "e2e_s", "mean_itl_ms", "process_to_first_content_s", "process_to_response_end_s"):
0237 |             if phase == "warm_reference" and metric.startswith("process_"):
0238 |                 continue
0239 |             rows.append((phase + " · " + metric, req.get(metric)))
0240 |     table = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value)) if value is not None else 'Não medido'}</td></tr>" for label, value in rows)
0241 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ciclo de vida</title><style>body{{font:17px/1.7 system-ui;background:#f6f3ec;color:#193835;margin:25px}}td,th{{padding:12px;border-bottom:1px solid #ccd6cc;text-align:left}}table{{width:100%;overflow-wrap:anywhere}}a{{color:#136d58}}</style><h1>Inicialização e primeira resposta</h1><p>Modo: {html.escape(lifecycle['mode'])}. Status: {html.escape(lifecycle['status'])}.</p><p>API disponível não implica modelo na GPU. O primeiro POST é cronometrado, sem teste de geração anterior. Processo novo não implica caches de disco/CUDA frios. A referência final repete o prompt e pode aproveitar prefix caching.</p><table>{table}</table><p><a href="{href('summary.html')}">Aquecimento e blocos GuideLLM</a> · <a href="{href('lifecycle.json')}">Dados do ciclo de vida</a></p></html>'''
0242 |     artifact(output, "lifecycle.html").write_text(page, encoding="utf-8")
```

## reporting.py

SHA-256: `42f9c070134b30a7cbb25e758cc25a6e11f675904096848c34b9457b611de02d`.

| Função/classe | Linhas |
|---|---|
| `ratio` | 12–15 |
| `percentile` | 18–25 |
| `derived` | 28–38 |
| `context_band` | 41–48 |
| `table` | 51–58 |
| `context_summary` | 61–100 |
| `gpu_summary` | 103–125 |
| `_timeseries` | 128–143 |
| `_chart` | 146–168 |
| `render` | 171–242 |
| `fmt` | 52–55 |

```text
0001 | """Métricas derivadas e relatório offline; não confunde contexto com VRAM."""
0002 | import csv
0003 | import html
0004 | import json
0005 | import math
0006 | from collections import defaultdict
0007 | from pathlib import Path
0008 |
0009 | from results_layout import artifact, href, locate
0010 |
0011 |
0012 | def ratio(a, b):
0013 |     if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
0014 |         return None
0015 |     return a / b if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0 else None
0016 |
0017 |
0018 | def percentile(values, q):
0019 |     """Percentil empírico com interpolação linear; ausências não viram zero."""
0020 |     values = sorted(v for v in values if v is not None and math.isfinite(v))
0021 |     if not values:
0022 |         return None
0023 |     position = (len(values) - 1) * q
0024 |     lower, upper = math.floor(position), math.ceil(position)
0025 |     return values[lower] + (values[upper] - values[lower]) * (position - lower)
0026 |
0027 |
0028 | def derived(row):
0029 |     n, p = row.get("output_tokens"), row.get("prompt_tokens")
0030 |     itl = row.get("inter_token_latency_ms")
0031 |     return {
0032 |         "decode_tokens_s": ratio(1000, itl) if n is not None and n > 1 else None,
0033 |         "effective_tokens_s": ratio(n, row.get("request_latency")),
0034 |         "context_start_tokens": p,
0035 |         # Comprimento lógico final; não é o número exato de posições materializadas.
0036 |         "context_end_tokens": p + n if p is not None and n is not None else None,
0037 |         "context_band": context_band(p),
0038 |     }
0039 |
0040 |
0041 | def context_band(p):
0042 |     if p is None:
0043 |         return "não informado"
0044 |     for limit in (512, 1024, 2048, 4096, 8192, 16384):
0045 |         if p < limit:
0046 |             lower = 0 if limit == 512 else limit // 2
0047 |             return f"[{lower}, {limit})"
0048 |     return "[16384, +∞)"
0049 |
0050 |
0051 | def table(rows, columns):
0052 |     def fmt(v):
0053 |         if v is None:
0054 |             return "Não disponível"
0055 |         return f"{v:.3f}" if isinstance(v, float) else str(v)
0056 |     head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
0057 |     body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(r.get(k)))}</td>" for k, _ in columns) + "</tr>" for r in rows)
0058 |     return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
0059 |
0060 |
0061 | def context_summary(output, kv_bytes_per_token=None):
0062 |     groups = defaultdict(list)
0063 |     csv_dir = Path(output) / "csv"
0064 |     files = csv_dir.glob("r*-*-requests.csv") if csv_dir.exists() else Path(output).glob("r*-*-requests.csv")
0065 |     for file in sorted(files):
0066 |         rep, scenario, phase, _ = file.stem.split("-", 3)
0067 |         with file.open() as handle:
0068 |             for row in csv.DictReader(handle):
0069 |                 if row["status"] != "successful":
0070 |                     continue
0071 |                 for key in ("prompt_tokens", "output_tokens", "inter_token_latency_ms", "request_latency", "time_to_first_token_ms"):
0072 |                     row[key] = float(row[key]) if row.get(key) else None
0073 |                 row.update(derived(row))
0074 |                 # Preservamos o cenário mesmo quando dois cenários caem na
0075 |                 # mesma faixa de contexto. Isso evita que short/medium/long
0076 |                 # apareçam como uma única população no JSON derivado.
0077 |                 groups[(phase, rep, scenario, row["context_band"])].append(row)
0078 |     result = []
0079 |     for (phase, rep, scenario, band), rows in groups.items():
0080 |         entry = {"phase": phase, "repetition": rep, "scenario": scenario,
0081 |                  "context_band": band, "successful_request_count": len(rows)}
0082 |         context_metrics = {
0083 |             "decode_generation_tokens_per_second": "decode_tokens_s",
0084 |             "effective_output_tokens_per_second": "effective_tokens_s",
0085 |             "request_first_token_latency_milliseconds": "time_to_first_token_ms",
0086 |             "initial_context_input_token_count": "context_start_tokens",
0087 |             "final_logical_context_token_count": "context_end_tokens",
0088 |         }
0089 |         for label, key in context_metrics.items():
0090 |             values = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
0091 |             entry[label + "_sample_count"] = len(values)
0092 |             entry[label + "_p50"] = percentile(values, .50)
0093 |             entry[label + "_p95"] = percentile(values, .95)
0094 |             entry[label + "_p99"] = percentile(values, .99)
0095 |         result.append(entry)
0096 |         for edge in ("start", "end"):
0097 |             tokens_key = "initial_context_input_token_count_p50" if edge == "start" else "final_logical_context_token_count_p50"
0098 |             tokens = entry[tokens_key]
0099 |             entry[f"estimated_{edge}_logical_kv_cache_mebibytes"] = tokens * kv_bytes_per_token / 1048576 if tokens is not None and kv_bytes_per_token else None
0100 |     return result
0101 |
0102 |
0103 | def gpu_summary(output):
0104 |     groups = defaultdict(list)
0105 |     path = locate(Path(output), "gpu.csv")
0106 |     if path.exists():
0107 |         with path.open() as handle:
0108 |             for row in csv.DictReader(handle):
0109 |                 groups[(row["phase"], row["index"], row["name"])].append(row)
0110 |     result = []
0111 |     for (phase, index, name), rows in groups.items():
0112 |         entry = {"phase": phase, "gpu": index, "name": name, "samples": len(rows)}
0113 |         for key in ("used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"):
0114 |             values = []
0115 |             for row in rows:
0116 |                 try:
0117 |                     value = float(row[key])
0118 |                     if math.isfinite(value):
0119 |                         values.append(value)
0120 |                 except (ValueError, TypeError, KeyError):
0121 |                     pass
0122 |             entry[key + "_mean"] = sum(values) / len(values) if values else None
0123 |             entry[key + "_max"] = max(values) if values else None
0124 |         result.append(entry)
0125 |     return result
0126 |
0127 |
0128 | def _timeseries(output):
0129 |     path = locate(Path(output), "gpu.csv")
0130 |     if not path.exists():
0131 |         return []
0132 |     with path.open() as handle:
0133 |         rows = []
0134 |         for row in csv.DictReader(handle):
0135 |             try:
0136 |                 elapsed = float(row["elapsed_s"])
0137 |                 used = float(row["used_mib"])
0138 |                 util = float(row["gpu_util_pct"])
0139 |                 if all(math.isfinite(value) for value in (elapsed, used, util)):
0140 |                     rows.append({"elapsed_s": elapsed, "used_mib": used, "gpu_util_pct": util})
0141 |             except (KeyError, TypeError, ValueError):
0142 |                 continue
0143 |         return rows
0144 |
0145 |
0146 | def _chart(rows, key, title, ylabel, color, maximum=None):
0147 |     """Gera SVG autônomo para abrir no navegador sem dependências externas."""
0148 |     if not rows:
0149 |         return f'<p class="chart-empty">Sem amostras válidas para {html.escape(title.lower())}.</p>'
0150 |     width, height, pad = 920, 260, 42
0151 |     xmax = max(row["elapsed_s"] for row in rows) or 1.0
0152 |     ymax = maximum or max(row[key] for row in rows) or 1.0
0153 |     points = []
0154 |     for row in rows:
0155 |         x = pad + (width - 2 * pad) * row["elapsed_s"] / xmax
0156 |         y = height - pad - (height - 2 * pad) * row[key] / ymax
0157 |         points.append(f"{x:.1f},{y:.1f}")
0158 |     polyline = " ".join(points)
0159 |     svg = (f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
0160 |            f'<rect width="100%" height="100%" fill="#fbfaf5"/><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#789"/>'
0161 |            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#789"/>'
0162 |            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
0163 |            f'<text x="{pad}" y="20" fill="#193835">{html.escape(title)}</text>'
0164 |            f'<text x="{pad}" y="{height-8}" fill="#526">0 s</text>'
0165 |            f'<text x="{width-pad-55}" y="{height-8}" fill="#526">{xmax:.1f} s</text>'
0166 |            f'<text x="6" y="{pad+5}" fill="#526">{ymax:.0f}</text>'
0167 |            f'<text x="6" y="{height-pad+5}" fill="#526">0</text></svg>')
0168 |     return svg
0169 |
0170 |
0171 | def render(output, rows):
0172 |     output = Path(output)
0173 |     lifecycle_path, manifest_path = locate(output, "lifecycle.json"), locate(output, "manifest.json")
0174 |     lifecycle = json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else {}
0175 |     manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
0176 |     kv_bytes = manifest.get("model_availability", {}).get("kv_bytes_per_token")
0177 |     context, gpu = context_summary(output, kv_bytes), gpu_summary(output)
0178 |     artifact(output, "context-summary.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
0179 |     artifact(output, "gpu-summary.json").write_text(json.dumps(gpu, ensure_ascii=False, indent=2) + "\n")
0180 |     startup = lifecycle.get("readiness", {}).get("process_to_api_observed_s")
0181 |     initial = []
0182 |     for key, label in (("first_request", "Primeira resposta"), ("warm_reference", "Referência final")):
0183 |         req = lifecycle.get(key, {})
0184 |         usage = req.get("stream_usage") or {}
0185 |         d = derived({"output_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
0186 |                      "inter_token_latency_ms": req.get("mean_itl_ms"), "request_latency": req.get("e2e_s")})
0187 |         initial.append({"phase": label, "ttft": req.get("ttft_ms"), "e2e": req.get("e2e_s"), **d})
0188 |     metrics = table(rows, [("phase", "Fase"), ("scenario", "Cenário"), ("repetition", "Repetição"),
0189 |         ("expected", "Requisições previstas"), ("successful_request_count", "Requisições bem-sucedidas"),
0190 |         ("errored_request_count", "Requisições com erro"), ("incomplete_request_count", "Requisições incompletas"),
0191 |         ("missing_request_count", "Requisições ausentes do relatório bruto"),
0192 |         ("time_to_first_token_milliseconds_p50", "Time To First Token p50 (ms)"),
0193 |         ("time_to_first_token_milliseconds_p95", "Time To First Token p95 (ms)"),
0194 |         ("time_to_first_token_milliseconds_p99", "Time To First Token p99 (ms)"),
0195 |         ("generation_time_seconds_p50", "Tempo de geração p50 (s)"),
0196 |         ("decode_tokens_per_second_p50", "Tokens/s de decodificação p50"),
0197 |         ("end_to_end_tokens_per_second_p50", "Tokens/s ponta a ponta p50"),
0198 |         ("end_to_end_latency_seconds_p50", "Latência ponta a ponta p50 (s)"),
0199 |         ("input_prompt_token_count_p50", "Tokens de entrada p50"),
0200 |         ("output_completion_token_count_p50", "Tokens de saída p50")])
0201 |     by_context = table(context, [("phase", "Fase"), ("repetition", "Repetição"), ("scenario", "Cenário"), ("context_band", "Faixa de entrada (tokens)"),
0202 |         ("successful_request_count", "Requisições bem-sucedidas"),
0203 |         ("initial_context_input_token_count_p50", "Tokens de entrada inicial p50"),
0204 |         ("final_logical_context_token_count_p50", "Tokens de contexto lógico final p50"),
0205 |         ("estimated_start_logical_kv_cache_mebibytes", "KV lógico inicial estimado (MiB)"),
0206 |         ("estimated_end_logical_kv_cache_mebibytes", "KV lógico final estimado (MiB)"),
0207 |         ("request_first_token_latency_milliseconds_p50", "Latência da requisição até primeiro token p50 (ms)"),
0208 |         ("decode_generation_tokens_per_second_p50", "Velocidade de geração p50 (tokens/s)"),
0209 |         ("effective_output_tokens_per_second_p50", "Velocidade efetiva p50 (tokens/s)")])
0210 |     hardware = table(gpu, [("phase", "Fase"), ("gpu", "GPU"), ("name", "Nome"), ("samples", "Amostras"),
0211 |         ("used_mib_max", "Memória máx. (MiB)"), ("total_mib_max", "Memória total (MiB)"),
0212 |         ("gpu_util_pct_mean", "Utilização média (%)"), ("gpu_util_pct_max", "Utilização máx. (%)"),
0213 |         ("temperature_c_max", "Temperatura máx. (°C)"), ("power_w_mean", "Potência média (W)"), ("power_w_max", "Potência máx. (W)")]) if gpu else "<p>Não disponível: nenhuma amostra NVIDIA válida. Em Apple/Metal este coletor não mede GPU; isso não significa utilização zero.</p>"
0214 |     kvfile = locate(output, "kv-cache.csv")
0215 |     kvrows = []
0216 |     if kvfile.exists():
0217 |         with kvfile.open() as handle:
0218 |             groups = defaultdict(list)
0219 |             for r in csv.DictReader(handle):
0220 |                 groups[(r["phase"], r["series"])].append(float(r["fraction"]) * 100)
0221 |             kvrows = [{"phase": p, "series": s, "n": len(v), "mean": sum(v)/len(v), "max": max(v)} for (p, s), v in groups.items()]
0222 |     kv = table(kvrows, [("phase", "Fase"), ("series", "Série do servidor"), ("n", "Amostras"), ("mean", "Ocupação média (%)"), ("max", "Ocupação máx. (%)")]) if kvrows else "<p>Ocupação real de KV não disponível nesta execução. Não foi estimada a partir da VRAM.</p>"
0223 |     first = table(initial, [("phase", "Fase"), ("ttft", "TTFT (ms)"), ("decode_tokens_s", "Geração (tokens/s)"), ("effective_tokens_s", "Efetiva (tokens/s)"), ("e2e", "Total (s)")])
0224 |     policy = manifest.get("model_availability", {}).get("policy", "Execução anterior: veja o estado inicial; ausência de download não verificada por esta versão.")
0225 |     failure = f'<p class="note">Execução não concluída: {html.escape(str(manifest["error"]))}. Dados parciais não constituem uma bateria válida.</p>' if manifest.get("error") else ""
0226 |     series = _timeseries(output)
0227 |     memory_chart = _chart(series, "used_mib", "Memória da GPU ocupada (MiB)", "MiB", "#b54e27")
0228 |     util_chart = _chart(series, "gpu_util_pct", "Utilização computacional da GPU (%)", "%", "#136d58", 100)
0229 |     artifact(output, "telemetry-memory.svg").write_text(memory_chart, encoding="utf-8") if series else None
0230 |     artifact(output, "telemetry-gpu-util.svg").write_text(util_chart, encoding="utf-8") if series else None
0231 |     page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark · latência, geração e GPU</title>
0232 | <style>body{{font:16px/1.7 system-ui;margin:32px;background:#f6f3ec;color:#193835}}main{{max-width:1400px;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border:1px solid #ccd6cc;text-align:left}}th{{background:#e0e9df}}.scroll{{overflow:auto}}h2{{margin-top:38px}}a{{color:#136d58}}.note{{padding:16px;background:#fff0de;border-left:4px solid #b54e27}}.chart{{width:100%;max-height:280px;border:1px solid #ccd6cc;margin:10px 0 22px}}.chart-empty{{padding:16px;background:#fff0de}}</style><main>
0233 | <h1>Um usuário · latência, geração e GPU</h1><p>Status: <strong>{html.escape(manifest.get('status', 'desconhecido'))}</strong>. {html.escape(policy)}</p>{failure}
0234 | <p class="note">TTFT = espera pelo primeiro token/conteúdo observado. Geração = (tokens de saída − 1)/(tempo entre primeiro e último token). Efetiva = tokens de saída/tempo total da requisição, incluindo TTFT. São taxas por requisição, não throughput agregado de usuários.</p>
0235 | <h2>1. Inicialização e primeira resposta</h2><p>Processo → API disponível: {html.escape(str(startup)) if startup is not None else 'não medido'} s. <a href="lifecycle.html">Ver ciclo de vida completo</a>.</p>{first}
0236 | <h2>2. Aquecimento e operação posterior</h2><p>Percentis entre requisições bem-sucedidas. Warmup e measure separados; p95 com menos de 100 sucessos é exploratório. Geração indisponível com menos de dois tokens ou intervalo não positivo.</p>{metrics}
0237 | <h2>3. Tokens/s por faixa de contexto — proxy da carga de KV</h2><p>Faixa definida pela entrada real, incluindo template, antes do decode. O contexto cresce durante a saída; mostramos também seu comprimento lógico final. Esta é uma comparação de velocidades médias de respostas iniciadas em cada faixa, não uma medição token a token dentro de faixas de ocupação física do cache.</p>{by_context}
0238 | <p>Para atenção completa, mantendo modelo, dtype de KV e uma sequência: KV lógico ≈ 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento × tokens. Pesos 4/8 bits não determinam o dtype do KV. Blocos, reserva, prefix caching e sliding window impedem tratar essa fórmula como medição de VRAM. MiB estimados só aparecem com --kv-bytes-per-token informado e verificado pelo operador; caso contrário, ficam indisponíveis.</p>
0239 | <h2>4. GPU, CPU, RAM e SSD por fase</h2><p>O monitor amostra aproximadamente 1 vez/s e relaciona cada amostra a eventos/fases. GPU vem do nvidia-smi; CPU, RAM e I/O de disco vêm do host. Máximos amostrados podem perder picos e não há atribuição por processo. N/A é ausência de dado, não zero.</p><h3>Memória e utilização ao longo da execução</h3>{memory_chart}{util_chart}<p><a href="{href('telemetry-memory.svg')}">SVG da memória</a> · <a href="{href('telemetry-gpu-util.svg')}">SVG da GPU-util</a> · <a href="{href('telemetry-summary.json')}">Resumo de telemetria</a> · <a href="{href('events.csv')}">Eventos</a> · <a href="{href('system.csv')}">CPU/RAM/SSD</a></p>{hardware}
0240 | <h2>5. Ocupação real do pool KV — vLLM</h2><p>Coleta opcional de /metrics via --collect-kv-metrics. Percentual de blocos ocupados do pool, não percentual de VRAM nem bytes. Séries/engines separados. Amostragem e atualização do servidor podem perder transientes; não sincronizada por token.</p>{kv}
0241 | <p><a href="{href('summary.json')}">Resumo JSON</a> · <a href="{href('context-summary.json')}">Faixas JSON</a> · <a href="{href('gpu-summary.json')}">GPU JSON</a> · <a href="{href('manifest.json')}">Manifesto</a></p></main></html>'''
0242 |     artifact(output, "summary.html").write_text(page, encoding="utf-8")
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
