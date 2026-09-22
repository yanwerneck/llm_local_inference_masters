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


def launch_with_context(source: Path, target: Path, context: int, context_flag: str, extra_args: list[str]) -> None:
    args = json.loads(source.read_text())
    if not isinstance(args, list):
        raise ValueError("launch JSON deve ser uma lista de argumentos")
    if context_flag != "none":
        if context_flag in args:
            index = args.index(context_flag) + 1
            args[index] = str(context)
        else:
            args.extend([context_flag, str(context)])
    args.extend(extra_args)
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
    parser.add_argument("--runtime-label", default="runtime")
    parser.add_argument("--context-flag", default="--max-model-len",
                        help="Flag do servidor que controla contexto; use none quando ele é configurado fora do argv.")
    parser.add_argument("--launch-extra-args", nargs="*", default=[],
                        help="Argumentos adicionais acrescentados ao launch em todos os pontos.")
    parser.add_argument("--launch-executable", help="Substitui argv[0] do launch em todos os pontos.")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"runtime": args.runtime_label, "step": args.step, "start": args.start,
                "max_context": args.max_context, "points": []}
    with tempfile.TemporaryDirectory(prefix="kv-sweep-") as temp:
        temp = Path(temp)
        for context in range(args.start, args.max_context + 1, args.step):
            # O contexto declarado inclui prompt, saída e margem do template.
            config_copy = dict(config)
            config_copy["context_window"] = context + 128 + 256
            config_path = temp / f"config-{context}.json"
            launch_path = temp / f"launch-{context}.json"
            config_path.write_text(json.dumps(config_copy, indent=2) + "\n")
            launch_with_context(Path(args.launch), launch_path, context + 128 + 256,
                                args.context_flag, args.launch_extra_args)
            command = [args.python, "bench.py", "run", "--config", str(config_path),
                       "--local-model-path", args.local_model_path, "--launch", str(launch_path),
                       "--input-tokens", str(context), "--requests", str(args.requests),
                       "--repetitions", str(args.repetitions), "--warmup", str(args.warmup),
                       "--collect-kv-metrics", "--startup-timeout", str(args.startup_timeout),
                       "--results", str(root)]
            if args.launch_executable:
                command.extend(["--launch-executable", args.launch_executable])
            print(f"\n[KV-SWEEP] runtime={args.runtime_label} input_tokens={context} "
                  f"server_max_model_len={context + 128 + 256} "
                  f"requests={args.requests} repetitions={args.repetitions}", flush=True)
            completed = subprocess.run(command)
            manifest["points"].append({"input_tokens": context, "returncode": completed.returncode})
            (root / "sweep-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if completed.returncode != 0:
                print(f"[KV-SWEEP] falha em input_tokens={context}; encerrando para preservar o primeiro limite de memória.")
                return completed.returncode
    print("Varredura concluída; consulte cada diretório e sweep-manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
