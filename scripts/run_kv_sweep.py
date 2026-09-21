#!/usr/bin/env python3
"""Executa uma bateria de contexto/KV, reiniciando o servidor a cada ponto."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("use um inteiro positivo")
    return number


def launch_with_context(source: Path, target: Path, context: int) -> None:
    args = json.loads(source.read_text())
    if not isinstance(args, list):
        raise ValueError("launch JSON deve ser uma lista de argumentos")
    if "--max-model-len" in args:
        index = args.index("--max-model-len") + 1
        args[index] = str(context)
    else:
        args.extend(["--max-model-len", str(context)])
    target.write_text(json.dumps(args, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--start", type=positive, default=1024)
    parser.add_argument("--step", type=positive, default=1024)
    parser.add_argument("--max-context", type=positive, default=16384)
    parser.add_argument("--requests", type=positive, default=50)
    parser.add_argument("--repetitions", type=positive, default=3)
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--startup-timeout", type=positive, default=1800)
    parser.add_argument("--results", default="results/kv-sweep")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"step": args.step, "start": args.start, "max_context": args.max_context, "points": []}
    with tempfile.TemporaryDirectory(prefix="kv-sweep-") as temp:
        temp = Path(temp)
        for context in range(args.start, args.max_context + 1, args.step):
            # O contexto declarado inclui prompt, saída e margem do template.
            config_copy = dict(config)
            config_copy["context_window"] = context + 128 + 256
            config_path = temp / f"config-{context}.json"
            launch_path = temp / f"launch-{context}.json"
            config_path.write_text(json.dumps(config_copy, indent=2) + "\n")
            launch_with_context(Path(args.launch), launch_path, context + 128 + 256)
            command = [args.python, "bench.py", "run", "--config", str(config_path),
                       "--local-model-path", args.local_model_path, "--launch", str(launch_path),
                       "--input-tokens", str(context), "--requests", str(args.requests),
                       "--repetitions", str(args.repetitions), "--warmup", str(args.warmup),
                       "--collect-kv-metrics", "--startup-timeout", str(args.startup_timeout),
                       "--results", str(root)]
            print("\n=== KV/context", context, "tokens ===", flush=True)
            completed = subprocess.run(command)
            manifest["points"].append({"input_tokens": context, "returncode": completed.returncode})
            (root / "sweep-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if completed.returncode != 0:
                print("Falha neste ponto; encerrando a varredura para preservar o primeiro limite de memória.")
                return completed.returncode
    print("Varredura concluída; consulte cada diretório e sweep-manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
