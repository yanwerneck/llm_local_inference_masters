# Chatbot Runtime Bench

Benchmark acadêmico de inferência local para comparar **vLLM, llama.cpp e Ollama** usando o mesmo modelo GGUF, uma requisição por vez e uma carga de chatbot reproduzível.

O fluxo oficial é baseado no `Makefile`. O benchmark não baixa modelos, não troca artefatos automaticamente e não faz fallback quando um runtime falha. Download, criação de diretórios e instalação são preparação; não entram nos tempos medidos.

Documentação detalhada: [fluxo completo do Make](docs/make-fluxo.md) · [metodologia](docs/metodologia.html) · [código explicado](docs/codigo-explicado.html).

## Preparar um pod novo

Em uma imagem Ubuntu/Debian do RunPod:

```bash
apt-get update
apt-get install -y git curl ca-certificates build-essential cmake pkg-config python3 python3-venv python3-pip
nvidia-smi
git --version
python3 --version
mkdir -p /workspace
cd /workspace
git clone git@github.com:yanwerneck/llm_local_inference_masters.git
cd /workspace/llm_local_inference_masters/chatbot-runtime-bench
```

Se SSH não estiver configurado, use `git clone https://github.com/yanwerneck/llm_local_inference_masters.git`. O `hf` é instalado dentro do venv pelo `make install-benchmark`; não é necessário instalar um `hf` separado no sistema. Depois da instalação, verifique com `/workspace/chatbot-runtime-bench/.venv/bin/hf --help` ou deixe o Make chamá-lo automaticamente.

## Fluxo mínimo no RunPod

```bash
git clone git@github.com:yanwerneck/llm_local_inference_masters.git
cd llm_local_inference_masters/chatbot-runtime-bench
make prepare-benchmark MODEL_SIZE=7B
make prepare-ollama MODEL_SIZE=7B
make bench MODEL_SIZE=7B
```

O alvo `bench` é um alias de `bench-all`.

## Modelo usado

O modelo 7B oficial desta rodada é [`arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf`](https://huggingface.co/arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf). O `make prepare-benchmark MODEL_SIZE=7B` baixa esse GGUF e o tokenizer base para o SSD. O tokenizer não é o peso: ele é necessário para construir prompts e contar tokens comparavelmente.

Para a rodada 14B:

```bash
make prepare-benchmark MODEL_SIZE=14B
make prepare-ollama MODEL_SIZE=14B
make bench MODEL_SIZE=14B
```

Se o arquivo estiver em outro caminho:

```bash
make prepare-benchmark MODEL_SIZE=14B \
  MODEL_14B_GGUF=/workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
```

O 14B pode exceder a VRAM da RTX 3090. Isso é uma falha experimental válida: preserve o log, sem substituir automaticamente pelo 7B.

## O que cada alvo faz

### `make prepare-benchmark`

Executa a preparação completa, nesta ordem:

1. baixa o GGUF selecionado para o SSD;
2. cria as pastas necessárias;
3. baixa `config.json`, `tokenizer_config.json`, tokenizer e arquivos especiais;
4. verifica assinatura `GGUF`, tamanho e SHA-256;
5. cria `results/`;
6. instala `requirements.txt` no Python indicado por `PYTHON`;
7. valida imports do cliente (`guidellm`, `httpx`, `psutil`, `transformers`);
8. valida os JSONs do modelo selecionado;
9. deixa a validação dos executáveis para o alvo específico de cada runtime.

Não inicia servidores e não mede desempenho. Para validar apenas um runtime:

```bash
make prepare-vllm MODEL_SIZE=7B
make prepare-llama MODEL_SIZE=7B
make prepare-ollama MODEL_SIZE=7B
```

Assim, a ausência de vLLM não impede preparar o modelo para llama.cpp ou Ollama.

`prepare-llama` assume que a imagem do pod já fornece `llama-server` no `PATH` e valida esse executável. O template chama simplesmente `llama-server`, portanto não depende de um caminho específico de `/workspace`. Se a imagem não tiver o binário, use `make build-llama` explicitamente (com CMake e toolkit CUDA) ou instale o runtime no ambiente do pod.

O mesmo princípio vale para os três runtimes: os templates chamam `vllm`, `llama-server` e `ollama` pelo `PATH`, sem assumir `/workspace/vllm-runtime/.venv` ou `/usr/local/bin/ollama`. Se a imagem usar outro local, coloque o diretório no `PATH` ou sobrescreva `VLLM_BIN`, `LLAMA_SERVER_BIN` ou `OLLAMA_BIN`.

```bash
make prepare-benchmark MODEL_SIZE=7B
make prepare-benchmark MODEL_SIZE=14B
```

O cliente usa por padrão `/workspace/chatbot-runtime-bench/.venv`. A criação e instalação podem ser executadas isoladamente:

```bash
make install-benchmark
```

Para usar outro local ou Python:

```bash
make install-benchmark \
  BENCH_VENV=/workspace/meu-bench/.venv \
  PYTHON=/workspace/meu-bench/.venv/bin/python
```

Não instale `requirements.txt` no venv do vLLM: ele pode trocar `torch`, `transformers` e dependências CUDA do runtime.

Para executar partes isoladas:

```bash
make download-model MODEL_SIZE=7B
make download-tokenizer MODEL_SIZE=7B
make verify-gguf MODEL_SIZE=7B
```

### `make prepare-ollama`

Cria o alias `qwen7b-q8-gguf` ou `qwen14b-q8-gguf` a partir do GGUF local usando um `Modelfile` temporário. Não executa `ollama pull`.

### Benchmarks individuais e agregado

```bash
make bench-vllm MODEL_SIZE=7B
make bench-llama MODEL_SIZE=7B
make bench-ollama MODEL_SIZE=7B
make bench MODEL_SIZE=7B
```

`bench` é um alias de `bench-all`. Cada alvo individual executa a bateria base e, imediatamente depois, o sweep de KV. O agregador tenta os três runtimes, continua se um falhar e retorna erro ao final se houver falha.

### Smoke test

```bash
make smoke-vllm MODEL_SIZE=7B
```

O smoke usa poucas requisições e somente `short`; serve para diagnosticar instalação, porta, tokenizer e carregamento. Não é resultado final.

## O que o benchmark mede

Por runtime, cenário e repetição, são executadas 50 requisições de medição, 3 repetições e 3 aquecimentos. A execução é síncrona: uma requisição ativa por vez.

| Cenário | Entrada aproximada | Saída máxima |
|---|---:|---:|
| `short` | 256 tokens | 128 tokens |
| `medium` | 2048 tokens | 128 tokens |
| `long` | 8192 tokens | 128 tokens |

Métricas principais:

- **Time To First Token (ms)**: do POST até o primeiro conteúdo/token observado; p50, p95 e p99 resumem requisições diferentes.
- **Tokens/s**: velocidade durante o decode, calculada pelos intervalos entre tokens; não inclui TTFT.
- **Effective tokens/s**: tokens de saída divididos pela duração total, incluindo TTFT.

Warmup, primeira resposta, medição formal e encerramento são fases distintas. Falhas e incompletas não entram nas velocidades válidas.

## Sweep de KV cache

O sweep faz parte de `bench-vllm`, `bench-llama` e `bench-ollama`. Para cada runtime, o servidor é reiniciado com contexto de 1024, 2048, 3072 tokens e assim por diante, até a primeira falha de inicialização ou memória.

No vLLM também é coletado o gauge de ocupação do pool KV em `/metrics`. Esse percentual não é percentual de VRAM nem bytes físicos; a VRAM é observada separadamente pelo `nvidia-smi`.

## Parâmetros dos runtimes

Argumentos extras são anexados ao `argv` e registrados no `lifecycle.json`:

```bash
make bench-vllm MODEL_SIZE=7B \
  VLLM_EXTRA_ARGS='--gpu-memory-utilization 0.85 --enforce-eager'

make bench-llama MODEL_SIZE=7B \
  LLAMA_EXTRA_ARGS='--flash-attn auto --threads 8'

make bench-ollama MODEL_SIZE=7B OLLAMA_EXTRA_ARGS=''
```

O Make não corrige flags inválidas. A falha deve aparecer no `server.log`.

Variáveis principais:

```text
MODEL_SIZE=7B|14B
MODEL_7B_GGUF / MODEL_14B_GGUF
HF_GGUF_REPO / HF_GGUF_FILENAME
HF_TOKENIZER_MODEL
VLLM_EXTRA_ARGS / LLAMA_EXTRA_ARGS / OLLAMA_EXTRA_ARGS
BENCH_REQUESTS / BENCH_REPETITIONS / BENCH_WARMUP
```

## Telemetria e resultados

Cada execução cria `results/<timestamp>/` com:

- `summary.json`, `summary.csv`, `summary.html`: resultados por fase, cenário e repetição;
- JSONs brutos e CSVs por requisição;
- `gpu.csv`: VRAM, utilização, temperatura e potência aproximadamente a cada segundo;
- `system.csv`: CPU, RAM, espaço e I/O host-wide;
- `events.csv`: timestamps das fases;
- `telemetry-summary.json`: médias e máximos por fase;
- `kv-cache.csv`: ocupação KV quando o endpoint existe;
- `server.log`: stdout/stderr do processo iniciado;
- `manifest.json` e `lifecycle.json`: versões, argumentos, modelo e falhas.

Os eventos relacionam picos de VRAM/CPU a startup, primeira requisição, aquecimento, medição ou encerramento. O instrumento não mede diretamente o tempo de cada transferência PCIe/RAM↔VRAM; isso exige Nsight/CUDA instrumentation.

## Regras de validade

- mesmo GGUF, tokenizer, prompt, temperatura e limite de saída nos três runtimes;
- uma GPU limpa e nenhum servidor concorrente;
- nenhum download durante `bench`;
- smoke não é comparável à bateria formal;
- OOM, erro de loader, API ou tokenizer são preservados como falhas;
- comparação somente entre cenário, repetição e fase equivalentes.

## Diagnóstico rápido

```bash
make prepare-benchmark MODEL_SIZE=7B
make smoke-vllm MODEL_SIZE=7B
```

Se o servidor encerrar:

```bash
cat results/<timestamp>/server.log
cat results/<timestamp>/lifecycle.json
nvidia-smi
```

Uma falha é diagnóstico, não desempenho. Corrija a causa e repita o mesmo protocolo.
