#!/usr/bin/env python3
"""Um usuário: inicialização, primeira resposta, aquecimento e GuideLLM sequencial."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
VERSION = "0.3.0"
GUIDELLM_VERSION = "0.7.4"
WORKLOADS = {"short": 256, "medium": 2048, "long": 8192}
OUTPUT_TOKENS = 128


def local_model_check(value):
    """Checa presença, não lê pesos nem aquece o page cache deliberadamente."""
    path = Path(value).expanduser().resolve()
    if path.is_file():
        if path.suffix not in {".safetensors", ".gguf", ".bin", ".pt", ".pth"} and not path.name.startswith("sha256-"):
            raise ValueError("Informe um arquivo de pesos (não configuração/texto) ou blob sha256- do Ollama.")
        files = [path]
    elif path.is_dir():
        files = [f for f in path.rglob("*") if f.is_file() and f.suffix in {".safetensors", ".gguf", ".bin", ".pt", ".pth"}]
        for index in path.glob("*.index.json"):
            data = json.loads(index.read_text())
            missing = [name for name in set(data.get("weight_map", {}).values()) if not (path / name).is_file()]
            if missing:
                raise ValueError(f"Shards ausentes: {missing}")
    else:
        files = []
    if not files or any(f.stat().st_size == 0 for f in files):
        raise ValueError("Pesos locais ausentes/vazios. Prepare o modelo antes do benchmark; downloads não são permitidos na medição.")
    return {"policy": "Pesos locais obrigatórios; download excluído do protocolo.", "path": str(path),
            "files": len(files), "bytes": sum(f.stat().st_size for f in files),
            "validation": "presença/tamanho e shards declarados; não verifica conteúdo, SSD físico ou vínculo com API"}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def redact(value, secret):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in {"api_key", "authorization"} else redact(v, secret))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[REDACTED]")
    return value


def capture(command):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"runtime", "base_url", "model", "tokenizer", "context_window", "cache_policy",
                "runtime_version", "model_artifact", "server_command", "notes"}
    if set(cfg) != required:
        raise ValueError(f"Campos da configuração devem ser exatamente: {sorted(required)}")
    if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in required - {"context_window"}):
        raise ValueError("Os campos textuais da configuração devem estar preenchidos.")
    if type(cfg["context_window"]) is not int or cfg["context_window"] < 512:
        raise ValueError("context_window deve ser um inteiro >= 512.")
    url = urlsplit(cfg["base_url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("URL inválida: use http(s), sem credenciais, query ou fragmento.")
    if url.path not in {"", "/", "/v1", "/v1/"}:
        raise ValueError("Use a raiz do servidor (ex.: http://127.0.0.1:8000), sem endpoint.")
    cfg["base_url"] = f"{url.scheme}://{url.netloc}"
    return cfg


def validate_run(cfg, scenarios, smoke):
    for scenario in scenarios:
        # Margem para o template; o servidor continua sendo a autoridade final.
        needed = WORKLOADS[scenario] + OUTPUT_TOKENS + 256
        if cfg["context_window"] < needed:
            raise ValueError(f"{scenario} precisa de contexto declarado >= {needed}; "
                             "altere o servidor e depois o JSON, ou retire esse cenário.")
    if not smoke:
        if any("PREENCHER" in cfg[k] or "SUBSTITUA" in cfg[k]
               for k in ("model", "runtime_version", "model_artifact", "server_command")):
            raise ValueError("Preencha modelo, versão, artefato e comando antes da medição formal; --smoke permite rascunhos.")
        if cfg["cache_policy"] == "runtime-default-unverified":
            raise ValueError("Registre cache_policy após verificar o servidor. Ex.: disabled-confirmed ou enabled-recorded. "
                             "O script NÃO altera nem comprova a política de cache.")


def tokenizer_digest(path):
    path = Path(path)
    if not path.is_dir() or not (path / "tokenizer_config.json").exists():
        raise ValueError("Tokenizer local ausente. Execute: python bench.py prepare-tokenizer")
    hashes = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and not file.name.startswith("."):
            hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def prepare_tokenizer(args):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    path = Path(args.output)
    if path.exists():
        raise ValueError(f"{path} já existe; não será sobrescrito. Use outro --output.")
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    path.mkdir(parents=True)
    tokenizer.save_pretrained(path)
    write_json(path / "source.json", {"model": args.model, "revision": revision})
    print(f"Tokenizer salvo em {path}; revisão {revision}. Não foram baixados pesos.")
    print("Compartilhe esta pasta com o grupo para usar os mesmos arquivos.")


def scenario_config(cfg, name, count, seed, secret, timeout):
    return {
        "spec": {
            "backend": {"kind": "openai_http", "target": cfg["base_url"], "model": cfg["model"],
                        "request_format": "/v1/chat/completions", "stream": True,
                        "validate_backend": False, "verify": True, "follow_redirects": False,
                        "timeout": timeout, "api_key": secret or None,
                        "extras": {"body": {"temperature": 0, "top_p": 1}}},
            "profile": {"kind": "synchronous", "warmup": 0, "cooldown": 0},
            "constraints": [{"kind": "max_requests", "count": count}, {"kind": "max_errors", "count": 1}],
            "tokenizer": {"kind": "huggingface_auto", "model": str(Path(cfg["tokenizer"]).resolve()),
                          "load_kwargs": {"local_files_only": True, "trust_remote_code": False}},
            "data": [{"kind": "synthetic_text", "prompt_tokens": WORKLOADS[name],
                      "output_tokens": OUTPUT_TOKENS}],
            "data_loader": {"kind": "pytorch", "samples": count, "num_workers": 0, "shuffle": False},
            "seed": {"kind": "static", "value": seed},
            "metrics": {"kind": "generative", "sample_size": None, "prefer_response_metrics": True},
            "outputs": [],
        }
    }


class Monitor:
    """Amostragem best effort; CSV local, não um profiler de largura de banda."""
    def __init__(self, output, cfg=None, secret="", collect_kv=False):
        self.output = Path(output)
        self.stop_event = threading.Event()
        self.thread = None
        self.phase = "setup"
        self.cfg, self.secret, self.collect_kv = cfg, secret, collect_kv
        self.kv_thread = None

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        if self.collect_kv:
            self.kv_thread = threading.Thread(target=self.kv_loop, daemon=True)
            self.kv_thread.start()

    def kv_loop(self):
        import httpx
        headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
        pattern = re.compile(r'^(vllm:(?:kv_cache_usage_perc|gpu_cache_usage_perc))(\{[^}]*\})?\s+([0-9.eE+\-]+)(?:\s|$)')
        with (self.output / "kv-cache.csv").open("w", newline="") as handle, httpx.Client(timeout=1, headers=headers, follow_redirects=False) as client:
            writer = csv.writer(handle)
            writer.writerow(["utc", "phase", "series", "fraction"])
            while not self.stop_event.is_set():
                phase = self.phase
                try:
                    response = client.get(self.cfg["base_url"] + "/metrics")
                    response.raise_for_status()
                    matches = [m for line in response.text.splitlines() if (m := pattern.match(line))]
                    modern = any(m[1] == "vllm:kv_cache_usage_perc" for m in matches)
                    for m in matches:
                        if modern and m[1] != "vllm:kv_cache_usage_perc":
                            continue
                        value = float(m[3])
                        if math.isfinite(value) and 0 <= value <= 1:
                            writer.writerow([datetime.now(timezone.utc).isoformat(), phase, redact(m[1] + (m[2] or ""), self.secret), value])
                    handle.flush()
                except (httpx.HTTPError, ValueError):
                    pass  # Serveur inicializando/endpoint ausente: nunca inventar zeros.
                self.stop_event.wait(1)

    def loop(self):
        with (self.output / "gpu.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["utc", "phase", "index", "name", "used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"])
            while not self.stop_event.is_set():
                phase = self.phase
                result = capture(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                                  "--format=csv,noheader,nounits"])
                if result.get("returncode") != 0:
                    write_json(self.output / "gpu-unavailable.json", result)
                    return
                for row in csv.reader(result["stdout"].splitlines(), skipinitialspace=True):
                    writer.writerow([datetime.now(timezone.utc).isoformat(), phase, *row])
                handle.flush()
                self.stop_event.wait(1)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=17)
        if self.kv_thread:
            self.kv_thread.join(timeout=3)


def percentile(values, q):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(report):
    """Métricas por requisição, sem misturar warmup ou erros com sucessos."""
    benchmarks = report["benchmarks"]
    if len(benchmarks) != 1:
        raise ValueError("Esperado exatamente um benchmark sequencial.")
    requests = benchmarks[0]["requests"]
    from reporting import derived
    good = [{**r, **derived(r)} for r in requests["successful"]]
    result = {
        "successful_request_count": len(good),
        "errored_request_count": len(requests["errored"]),
        "incomplete_request_count": len(requests["incomplete"]),
    }
    # Os nomes são deliberadamente longos: summary.json é um artefato de
    # análise, e não uma API em que economizar alguns bytes melhora algo.
    metrics = {
        "request_first_token_latency_milliseconds": "time_to_first_token_ms",
        "request_latency_seconds": "request_latency",
        "within_response_next_token_latency_milliseconds": "inter_token_latency_ms",
        "output_completion_token_count": "output_tokens",
        "input_prompt_token_count": "prompt_tokens",
        "decode_generation_tokens_per_second": "decode_tokens_s",
        "effective_output_tokens_per_second": "effective_tokens_s",
    }
    for label, key in metrics.items():
        vals = [r.get(key) for r in good]
        result[label + "_sample_count"] = sum(v is not None for v in vals)
        result[label + "_p50"] = percentile(vals, .5)
        result[label + "_p95"] = percentile(vals, .95)
        result[label + "_p99"] = percentile(vals, .99)
    # O hash exclui aliases do modelo e chaves: apenas carga de entrada e limite de saída.
    bodies = []
    for row in good:
        args = json.loads(row["request_args"])
        body = args.get("body", {})
        bodies.append({k: body.get(k) for k in ("messages", "max_tokens")})
    result["requests_sha256"] = hashlib.sha256(json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    result["percentiles_are_exploratory"] = len(good) < 100
    result["percentile_definition"] = "Empirical linear interpolation over successful requests in this phase/scenario/repetition block."
    return result


def write_requests_csv(path, report):
    """Amostras individuais para análise no R/Python, incluindo status de erro."""
    from reporting import derived
    fields = ["status", "request_id", "request_start_time", "request_latency", "time_to_first_token_ms",
              "inter_token_latency_ms", "prompt_tokens", "output_tokens", "decode_tokens_s", "effective_tokens_s",
              "context_start_tokens", "context_end_tokens", "context_band"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for status in ("successful", "errored", "incomplete"):
            for row in report["benchmarks"][0]["requests"][status]:
                if status == "successful":
                    row = {**row, **derived(row)}
                writer.writerow({"status": status, **{key: row.get(key) for key in fields[1:]}})


def write_summary(output, rows):
    from reporting import render
    write_json(output / "summary.json", rows)
    if not rows:
        render(output, rows)
        return
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render(output, rows)


def run(args):
    import fcntl
    from lifecycle import DEFAULT_PROMPT, Launch, lifecycle_report, timed_request, wait_models
    if importlib.metadata.version("guidellm") != GUIDELLM_VERSION:
        raise ValueError(f"Este projeto exige guidellm=={GUIDELLM_VERSION}; reinstale requirements.txt.")
    cfg = load_config(args.config)
    validate_run(cfg, args.scenarios, args.smoke)
    availability = local_model_check(args.local_model_path)
    availability["kv_bytes_per_token"] = args.kv_bytes_per_token
    availability["launch_hf_offline"] = bool(args.launch)
    digest = tokenizer_digest(cfg["tokenizer"])
    count, repetitions = (3, 1) if args.smoke else (args.requests, args.repetitions)
    secret = os.environ.get("BENCH_API_KEY", "")
    prompt = Path(args.first_prompt_file).read_text(encoding="utf-8") if args.first_prompt_file else DEFAULT_PROMPT
    if not prompt.strip():
        raise ValueError("O prompt inicial não pode estar vazio.")
    base = Path(args.results)
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Já existe um benchmark usando esta pasta results. Não execute dois ao mesmo tempo.") from None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = base / stamp
        output.mkdir()
        manifest = {"project_version": VERSION, "guidellm_version": GUIDELLM_VERSION, "started_utc": stamp,
                    "config": cfg, "tokenizer": digest, "smoke": args.smoke, "requests": count,
                    "repetitions": repetitions, "warmup_requests_per_case": args.warmup,
                    "scenarios": args.scenarios, "seed": args.seed, "profile": "synchronous",
                    "model_availability": availability,
                    "guidellm_compat": "0.7.4 bounded drain of real late completion updates (5s); no request retry",
                    "telemetry": {"gpu_source": "local nvidia-smi", "kv_metrics_requested": args.collect_kv_metrics,
                                  "sampling": "approximately 1 Hz; not per-token"},
                    "python": sys.version, "platform": platform.platform(),
                    "git": capture(["git", "rev-parse", "HEAD"]), "status": "running"}
        write_json(output / "manifest.json", redact(manifest, secret))
        write_json(output / "client-packages.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
        write_json(output / "gpu-before.json", capture(["nvidia-smi"]))
        monitor, rows = Monitor(output, cfg, secret, args.collect_kv_metrics), []
        launch, origin, cleanup_error = None, None, None
        lifecycle = {"mode": "new-process" if args.launch else "existing-server-state-unknown",
                     "status": "running", "initial_state_note": args.initial_state,
                     "startup_timeout_s": args.startup_timeout}
        print(f"Resultados: {output.resolve()}", flush=True)
        try:
            monitor.start()
            if args.launch:
                launch = Launch(cfg, args.launch, output)
                lifecycle["argv"] = redact(launch.argv, secret)
                monitor.phase = "process_startup"
                origin = launch.start()
                lifecycle["pid"] = launch.process.pid
            lifecycle_report(output, redact(lifecycle, secret))
            lifecycle["readiness"] = wait_models(cfg, secret, args.startup_timeout, launch)
            lifecycle_report(output, redact(lifecycle, secret))
            monitor.phase = "first_request"
            print("Primeiro POST: medição da primeira resposta (nenhuma geração prévia enviada pelo cliente).", flush=True)
            lifecycle["first_request"] = timed_request(cfg, secret, args.timeout, prompt,
                                                       output / "first-request.json", origin)
            lifecycle_report(output, redact(lifecycle, secret))
            from guidellm.benchmark import BenchmarkScenario, benchmark_generative_text
            from guidellm_compat import completion_drain
            for rep in range(repetitions):
                # Rotação balanceia parcialmente a posição dos cenários entre repetições.
                names = args.scenarios[rep % len(args.scenarios):] + args.scenarios[:rep % len(args.scenarios)]
                for name in names:
                    for phase, n in (("warmup", args.warmup), ("measure", count)):
                        if not n:
                            continue
                        seed = args.seed + rep * 100 + list(WORKLOADS).index(name)
                        if phase == "warmup":
                            seed += 1_000_000
                        prefix = f"r{rep+1}-{name}-{phase}"
                        monitor.phase = prefix
                        config = scenario_config(cfg, name, n, seed, secret, args.timeout)
                        write_json(output / f"{prefix}-config.json", redact(config, secret))
                        print(f"{prefix}: {n} requisições, uma por vez", flush=True)
                        with completion_drain():
                            report, _ = asyncio.run(benchmark_generative_text(BenchmarkScenario.model_validate(config)))
                        raw = redact(report.model_dump(mode="json"), secret)
                        write_json(output / f"{prefix}.json", raw)
                        write_requests_csv(output / f"{prefix}-requests.csv", raw)
                        summary = summarize(raw)
                        summary["expected"] = n
                        summary["missing_request_count"] = max(0, n - sum(summary[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count")))
                        rows.append({"runtime": cfg["runtime"], "model": cfg["model"],
                                     "cache_policy": cfg["cache_policy"], "tokenizer_sha256": digest["sha256"],
                                     "phase": phase, "scenario": name, "repetition": rep+1, **summary})
                        write_summary(output, rows)
                        if summary["successful_request_count"] != n or summary["errored_request_count"] or summary["incomplete_request_count"]:
                            raise RuntimeError(f"{prefix}: requisições falharam ou execução incompleta. Veja o JSON; não compare como sucesso.")
            monitor.phase = "warm_reference"
            lifecycle["warm_reference"] = timed_request(cfg, secret, args.timeout, prompt,
                                                         output / "warm-reference.json")
            manifest["status"] = lifecycle["status"] = "complete"
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error"] = redact(str(exc), secret)
            lifecycle["status"] = manifest["status"]
            lifecycle["error"] = manifest["error"]
            raise
        finally:
            for key, filename in (("first_request", "first-request.json"), ("warm_reference", "warm-reference.json")):
                if (output / filename).exists():
                    lifecycle[key] = json.loads((output / filename).read_text(encoding="utf-8"))
            if launch is not None:
                monitor.phase = "server_shutdown"
                try:
                    launch.close()
                    lifecycle["server_cleanup"] = "stopped owned process group only"
                except Exception as exc:
                    cleanup_error = redact(str(exc), secret)
                    lifecycle["server_cleanup_error"] = cleanup_error
                    lifecycle["status"] = manifest["status"] = "failed"
            lifecycle_report(output, redact(lifecycle, secret))
            monitor.stop()
            manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(output / "manifest.json", redact(manifest, secret))
            write_json(output / "gpu-after.json", capture(["nvidia-smi"]))
            write_summary(output, rows)
        if cleanup_error:
            raise RuntimeError(f"Falha ao encerrar processo criado: {cleanup_error}. Confira o PID no lifecycle.json.")
        print(f"Concluído. Abra {output / 'lifecycle.html'} e {output / 'summary.html'}")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Use um inteiro positivo.")
    return number


def rebuild_report(args):
    """Atualiza apenas derivados, preservando relatórios brutos e manifesto original."""
    output = Path(args.output)
    rows = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    for row in rows:
        prefix = f"r{row['repetition']}-{row['scenario']}-{row['phase']}"
        raw = json.loads((output / f"{prefix}.json").read_text())
        row.update(summarize(raw))
        expected = manifest.get("warmup_requests_per_case") if row["phase"] == "warmup" else manifest.get("requests")
        row["expected"] = expected
        row["missing_request_count"] = max(0, expected - sum(row[k] for k in ("successful_request_count", "errored_request_count", "incomplete_request_count"))) if expected is not None else None
        write_requests_csv(output / f"{prefix}-requests.csv", raw)
    write_summary(output, rows)
    print(f"Relatório atualizado: {output / 'summary.html'}; dados brutos preservados.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Regenera derivados de uma execução existente, sem nova inferência.")
    report.add_argument("--output", required=True)
    report.set_defaults(func=rebuild_report)
    prep = commands.add_parser("prepare-tokenizer", help="Baixa apenas tokenizer; fixa revisão e guarda origem.")
    prep.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    prep.add_argument("--revision", default="main")
    prep.add_argument("--output", default="tokenizer")
    prep.set_defaults(func=prepare_tokenizer)
    cmd = commands.add_parser("run", help="Mede primeiro acesso, aquecimento e GuideLLM; lançamento do servidor é opcional.")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--local-model-path", required=True, help="Pesos já no SSD: pasta HF, arquivo GGUF ou blob local do Ollama. Não baixa arquivos.")
    cmd.add_argument("--collect-kv-metrics", action="store_true", help="Amostra /metrics do vLLM (~1 Hz); ocupação do pool KV, não bytes.")
    cmd.add_argument("--kv-bytes-per-token", type=positive, help="Opcional: bytes de KV lógico por token, calculados para arquitetura/dtype reais. Estimativa, não VRAM medida.")
    cmd.add_argument("--input-tokens", nargs="+", type=positive, help="Substitui --scenarios por uma grade de comprimentos sintéticos, ex.: 256 512 1024 2048 3072.")
    cmd.add_argument("--scenarios", nargs="+", choices=list(WORKLOADS), default=["short", "medium"])
    cmd.add_argument("--requests", type=positive, default=30)
    cmd.add_argument("--repetitions", type=positive, default=3)
    cmd.add_argument("--warmup", type=positive, default=3)
    cmd.add_argument("--seed", type=positive, default=42)
    cmd.add_argument("--timeout", type=positive, default=300)
    cmd.add_argument("--results", default="results")
    cmd.add_argument("--smoke", action="store_true", help="3 medições e 1 repetição; não vale como resultado final.")
    cmd.add_argument("--launch", help="Arquivo JSON com argv para iniciar um runtime LOCAL; encerra só esse processo ao final.")
    cmd.add_argument("--startup-timeout", type=positive, default=1800, help="Limite da espera pela API com --launch, em segundos.")
    cmd.add_argument("--first-prompt-file", help="Texto UTF-8 para a primeira requisição e referência final; default: pergunta sobre RAM/VRAM.")
    cmd.add_argument("--initial-state", default="weights local; OS/compilation caches not controlled", help="Descreva SSD e caches existentes; apenas registra, não limpa.")
    cmd.set_defaults(func=run)
    args = parser.parse_args()
    if getattr(args, "input_tokens", None):
        if len(set(args.input_tokens)) != len(args.input_tokens):
            parser.error("Não repita comprimentos em --input-tokens.")
        args.scenarios = []
        for size in args.input_tokens:
            name = f"ctx{size}"
            WORKLOADS[name] = size
            args.scenarios.append(name)
    if hasattr(args, "scenarios") and len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Não repita cenários na lista.")
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrompido; resultados já concluídos foram preservados.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERRO: {redact(str(exc), os.environ.get('BENCH_API_KEY', ''))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
