"""Métricas derivadas e relatório offline; não confunde contexto com VRAM."""
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def ratio(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return a / b if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0 else None


def derived(row):
    n, p = row.get("output_tokens"), row.get("prompt_tokens")
    itl = row.get("inter_token_latency_ms")
    return {
        "decode_tokens_s": ratio(1000, itl) if n is not None and n > 1 else None,
        "effective_tokens_s": ratio(n, row.get("request_latency")),
        "context_start_tokens": p,
        # Comprimento lógico final; não é o número exato de posições materializadas.
        "context_end_tokens": p + n if p is not None and n is not None else None,
        "context_band": context_band(p),
    }


def context_band(p):
    if p is None:
        return "não informado"
    for limit in (512, 1024, 2048, 4096, 8192, 16384):
        if p < limit:
            lower = 0 if limit == 512 else limit // 2
            return f"[{lower}, {limit})"
    return "[16384, +∞)"


def table(rows, columns):
    def fmt(v):
        if v is None:
            return "Não disponível"
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(r.get(k)))}</td>" for k, _ in columns) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def context_summary(output, kv_bytes_per_token=None):
    groups = defaultdict(list)
    for file in sorted(Path(output).glob("r*-*-requests.csv")):
        rep, scenario, phase, _ = file.stem.split("-", 3)
        with file.open() as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "successful":
                    continue
                for key in ("prompt_tokens", "output_tokens", "inter_token_latency_ms", "request_latency", "time_to_first_token_ms"):
                    row[key] = float(row[key]) if row.get(key) else None
                row.update(derived(row))
                groups[(phase, rep, row["context_band"])].append(row)
    result = []
    for (phase, rep, band), rows in groups.items():
        entry = {"phase": phase, "repetition": rep, "context_band": band, "n": len(rows)}
        for key in ("decode_tokens_s", "effective_tokens_s", "time_to_first_token_ms", "context_start_tokens", "context_end_tokens"):
            values = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
            entry[key + "_p50"] = median(values) if values else None
        result.append(entry)
        for edge in ("start", "end"):
            tokens = entry[f"context_{edge}_tokens_p50"]
            entry[f"kv_{edge}_mib_estimate"] = tokens * kv_bytes_per_token / 1048576 if tokens is not None and kv_bytes_per_token else None
    return result


def gpu_summary(output):
    groups = defaultdict(list)
    path = Path(output) / "gpu.csv"
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                groups[(row["phase"], row["index"], row["name"])].append(row)
    result = []
    for (phase, index, name), rows in groups.items():
        entry = {"phase": phase, "gpu": index, "name": name, "samples": len(rows)}
        for key in ("used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"):
            values = []
            for row in rows:
                try:
                    value = float(row[key])
                    if math.isfinite(value):
                        values.append(value)
                except (ValueError, TypeError, KeyError):
                    pass
            entry[key + "_mean"] = sum(values) / len(values) if values else None
            entry[key + "_max"] = max(values) if values else None
        result.append(entry)
    return result


def render(output, rows):
    output = Path(output)
    lifecycle = json.loads((output / "lifecycle.json").read_text()) if (output / "lifecycle.json").exists() else {}
    manifest = json.loads((output / "manifest.json").read_text()) if (output / "manifest.json").exists() else {}
    kv_bytes = manifest.get("model_availability", {}).get("kv_bytes_per_token")
    context, gpu = context_summary(output, kv_bytes), gpu_summary(output)
    (output / "context-summary.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    (output / "gpu-summary.json").write_text(json.dumps(gpu, ensure_ascii=False, indent=2) + "\n")
    startup = lifecycle.get("readiness", {}).get("process_to_api_observed_s")
    initial = []
    for key, label in (("first_request", "Primeira resposta"), ("warm_reference", "Referência final")):
        req = lifecycle.get(key, {})
        usage = req.get("stream_usage") or {}
        d = derived({"output_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
                     "inter_token_latency_ms": req.get("mean_itl_ms"), "request_latency": req.get("e2e_s")})
        initial.append({"phase": label, "ttft": req.get("ttft_ms"), "e2e": req.get("e2e_s"), **d})
    metrics = table(rows, [("phase", "Fase"), ("scenario", "Cenário"), ("repetition", "Repetição"),
        ("expected", "Previstas"), ("successful", "Sucessos"), ("errored", "Erros"), ("incomplete", "Incompletas"), ("missing", "Ausentes do relatório bruto"),
        ("ttft_ms_p50", "TTFT p50 (ms)"), ("ttft_ms_p95", "TTFT p95 (ms)"),
        ("decode_tokens_s_p50", "Geração p50 (tokens/s)"), ("effective_tokens_s_p50", "Efetiva p50 (tokens/s)"),
        ("e2e_s_p50", "Total p50 (s)"), ("prompt_tokens_p50", "Entrada p50 (tokens)"), ("output_tokens_p50", "Saída p50 (tokens)")])
    by_context = table(context, [("phase", "Fase"), ("repetition", "Repetição"), ("context_band", "Faixa de entrada (tokens)"),
        ("n", "n"), ("context_start_tokens_p50", "Contexto inicial p50"), ("context_end_tokens_p50", "Contexto final p50"),
        ("kv_start_mib_estimate", "KV inicial estimado (MiB)"), ("kv_end_mib_estimate", "KV final estimado (MiB)"),
        ("time_to_first_token_ms_p50", "TTFT p50 (ms)"), ("decode_tokens_s_p50", "Geração p50 (tokens/s)"),
        ("effective_tokens_s_p50", "Efetiva p50 (tokens/s)")])
    hardware = table(gpu, [("phase", "Fase"), ("gpu", "GPU"), ("name", "Nome"), ("samples", "Amostras"),
        ("used_mib_max", "Memória máx. (MiB)"), ("total_mib_max", "Memória total (MiB)"),
        ("gpu_util_pct_mean", "Utilização média (%)"), ("gpu_util_pct_max", "Utilização máx. (%)"),
        ("temperature_c_max", "Temperatura máx. (°C)"), ("power_w_mean", "Potência média (W)"), ("power_w_max", "Potência máx. (W)")]) if gpu else "<p>Não disponível: nenhuma amostra NVIDIA válida. Em Apple/Metal este coletor não mede GPU; isso não significa utilização zero.</p>"
    kvfile = output / "kv-cache.csv"
    kvrows = []
    if kvfile.exists():
        with kvfile.open() as handle:
            groups = defaultdict(list)
            for r in csv.DictReader(handle):
                groups[(r["phase"], r["series"])].append(float(r["fraction"]) * 100)
            kvrows = [{"phase": p, "series": s, "n": len(v), "mean": sum(v)/len(v), "max": max(v)} for (p, s), v in groups.items()]
    kv = table(kvrows, [("phase", "Fase"), ("series", "Série do servidor"), ("n", "Amostras"), ("mean", "Ocupação média (%)"), ("max", "Ocupação máx. (%)")]) if kvrows else "<p>Ocupação real de KV não disponível nesta execução. Não foi estimada a partir da VRAM.</p>"
    first = table(initial, [("phase", "Fase"), ("ttft", "TTFT (ms)"), ("decode_tokens_s", "Geração (tokens/s)"), ("effective_tokens_s", "Efetiva (tokens/s)"), ("e2e", "Total (s)")])
    policy = manifest.get("model_availability", {}).get("policy", "Execução anterior: veja o estado inicial; ausência de download não verificada por esta versão.")
    failure = f'<p class="note">Execução não concluída: {html.escape(str(manifest["error"]))}. Dados parciais não constituem uma bateria válida.</p>' if manifest.get("error") else ""
    page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark · latência, geração e GPU</title>
<style>body{{font:16px/1.7 system-ui;margin:32px;background:#f6f3ec;color:#193835}}main{{max-width:1400px;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border:1px solid #ccd6cc;text-align:left}}th{{background:#e0e9df}}.scroll{{overflow:auto}}h2{{margin-top:38px}}a{{color:#136d58}}.note{{padding:16px;background:#fff0de;border-left:4px solid #b54e27}}</style><main>
<h1>Um usuário · latência, geração e GPU</h1><p>Status: <strong>{html.escape(manifest.get('status', 'desconhecido'))}</strong>. {html.escape(policy)}</p>{failure}
<p class="note">TTFT = espera pelo primeiro token/conteúdo observado. Geração = (tokens de saída − 1)/(tempo entre primeiro e último token). Efetiva = tokens de saída/tempo total da requisição, incluindo TTFT. São taxas por requisição, não throughput agregado de usuários.</p>
<h2>1. Inicialização e primeira resposta</h2><p>Processo → API disponível: {html.escape(str(startup)) if startup is not None else 'não medido'} s. <a href="lifecycle.html">Ver ciclo de vida completo</a>.</p>{first}
<h2>2. Aquecimento e operação posterior</h2><p>Percentis entre requisições bem-sucedidas. Warmup e measure separados; p95 com menos de 100 sucessos é exploratório. Geração indisponível com menos de dois tokens ou intervalo não positivo.</p>{metrics}
<h2>3. Tokens/s por faixa de contexto — proxy da carga de KV</h2><p>Faixa definida pela entrada real, incluindo template, antes do decode. O contexto cresce durante a saída; mostramos também seu comprimento lógico final. Esta é uma comparação de velocidades médias de respostas iniciadas em cada faixa, não uma medição token a token dentro de faixas de ocupação física do cache.</p>{by_context}
<p>Para atenção completa, mantendo modelo, dtype de KV e uma sequência: KV lógico ≈ 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento × tokens. Pesos 4/8 bits não determinam o dtype do KV. Blocos, reserva, prefix caching e sliding window impedem tratar essa fórmula como medição de VRAM. MiB estimados só aparecem com --kv-bytes-per-token informado e verificado pelo operador; caso contrário, ficam indisponíveis.</p>
<h2>4. GPU por fase</h2><p>Host do cliente; execute no mesmo pod do servidor. Aproximadamente 1 amostra/s, todas as GPUs visíveis, sem atribuição por processo. Máximos amostrados podem perder picos. N/A é ausência de dado, não zero.</p>{hardware}
<h2>5. Ocupação real do pool KV — vLLM</h2><p>Coleta opcional de /metrics via --collect-kv-metrics. Percentual de blocos ocupados do pool, não percentual de VRAM nem bytes. Séries/engines separados. Amostragem e atualização do servidor podem perder transientes; não sincronizada por token.</p>{kv}
<p><a href="summary.json">Resumo JSON</a> · <a href="context-summary.json">Faixas JSON</a> · <a href="gpu-summary.json">GPU JSON</a> · <a href="manifest.json">Manifesto</a></p></main></html>'''
    (output / "summary.html").write_text(page, encoding="utf-8")
