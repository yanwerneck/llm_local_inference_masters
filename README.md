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
git clone https://github.com/yanwerneck/llm_local_inference_masters.git
cd /workspace/llm_local_inference_masters
```

O repositório é público, portanto HTTPS é o caminho mais simples para clonar no Pod e não exige chave SSH. O `hf` é instalado dentro do venv pelo `make install-benchmark`; não é necessário instalar um `hf` separado no sistema. Depois da instalação, verifique com `./.venv/bin/hf --help` ou deixe o Make chamá-lo automaticamente.

### SSH no Pod (somente se precisar fazer push)

O erro `Permission denied (publickey)` significa que o Pod não tem uma chave SSH autorizada na sua conta GitHub. Para configurar uma chave própria no Pod:

```bash
ssh-keygen -t ed25519 -C "runpod-github-key"
cat ~/.ssh/id_ed25519.pub
```

Adicione o conteúdo exibido em **GitHub → Settings → SSH and GPG keys → New SSH key**. Depois valide:

```bash
ssh -T git@github.com
```

Se o repositório já foi clonado por HTTPS e você quer fazer push por SSH:

```bash
cd /workspace/llm_local_inference_masters
git remote set-url origin git@github.com:yanwerneck/llm_local_inference_masters.git
```

Não copie a chave privada do seu computador pessoal para um Pod descartável. Ao terminar, remova a chave do Pod ou revogue-a no GitHub.

## Fluxo mínimo no RunPod

```bash
git clone https://github.com/yanwerneck/llm_local_inference_masters.git
cd llm_local_inference_masters
make prepare-benchmark MODEL_SIZE=7B
make prepare-ollama MODEL_SIZE=7B
make bench MODEL_SIZE=7B
```

O alvo `bench` é um alias de `bench-all`.

## Modelo usado

O modelo 7B oficial desta rodada é [`arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf`](https://huggingface.co/arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf). O `make prepare-benchmark MODEL_SIZE=7B` baixa esse GGUF e o tokenizer base para o SSD. O tokenizer não é o peso: ele é necessário para construir prompts e contar tokens comparavelmente.

O download do tokenizer é feito pelo `huggingface_hub.snapshot_download` com filtros explícitos, e não pela combinação ambígua de nomes posicionais e `--include` da CLI `hf`. A preparação falha se `config.json`, `tokenizer_config.json` ou `tokenizer.json` não estiverem presentes; isso evita iniciar o llama.cpp com um diretório de tokenizer incompleto.

As configurações vLLM deste repositório usam `--gpu-memory-utilization 1.0`, isto é, disponibilizam 100% do orçamento de VRAM ao executor. Esse parâmetro não força `GPU-Util=100%`: a utilização computacional continua dependendo da carga, e OOMs continuam sendo preservados como falhas.

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

Antes de iniciar o vLLM com GGUF, `prepare-vllm` executa `make check-vllm-gguf`. Essa verificação importa `vllm` e `vllm_gguf_plugin` usando o Python pertencente ao executável `vllm`. Se o plugin não estiver nesse ambiente, a preparação para com a instrução explícita:

```bash
/caminho/do/venv-do-vllm/bin/python -m pip install vllm-gguf-plugin
```

O erro `config file ... .gguf is not a valid JSON file` significa que o plugin não foi carregado; não significa que a assinatura GGUF esteja inválida. llama.cpp não precisa desse plugin: `llama-server` lê GGUF nativamente. Ollama também lê GGUF nativamente, mas precisa do daemon e do alias criado por `prepare-ollama`.

```bash
make prepare-benchmark MODEL_SIZE=7B
make prepare-benchmark MODEL_SIZE=14B
```

O cliente usa por padrão `.venv` na raiz do repositório. A criação e instalação podem ser executadas isoladamente:

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

Verifica que o daemon Ollama está acessível e cria o alias correspondente ao tamanho escolhido (`qwen7b-q8-gguf` ou `qwen14b-q8-gguf`) a partir do GGUF local usando um `Modelfile` temporário. Não executa `ollama pull`.

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

As taxas oficiais são calculadas no cliente, por requisição, usando `time.perf_counter()` e a resposta SSE. O benchmark registra `request_start_time`, `first_token_time`, `request_end_time`, `time_to_first_token_seconds`, `generation_time_seconds`, `end_to_end_latency_seconds`, `completion_tokens`, `prompt_tokens`, `total_tokens`, `decode_tokens_per_second` e `end_to_end_tokens_per_second`. As fórmulas são: TTFT = primeiro evento de conteúdo − início; geração = último evento de conteúdo − primeiro; decode tokens/s = tokens de saída ÷ geração; tokens/s efetivos = tokens de saída ÷ latência total. O throughput periódico `Avg generation throughput` do vLLM é apenas telemetria agregada em janelas do servidor e nunca entra nessas métricas.

Como a API SSE não garante um evento por token, `first_token_time` e `request_end_time` são observados no cliente e `generation_time_seconds` usa o primeiro/último evento SSE com conteúdo. A contagem de tokens vem somente de `usage`; se `usage` faltar, contagens e taxas ficam `null`, sem estimativa. Para uma saída de um token, `inter_token_latency` também fica `null`.

Warmup, primeira resposta, medição formal e encerramento são fases distintas. Falhas e incompletas não entram nas velocidades válidas.

## Sweep de KV cache

O sweep faz parte de `bench-vllm`, `bench-llama` e `bench-ollama`. Para cada runtime, o servidor é reiniciado com contexto de 1024, 2048, 3072 tokens e assim por diante, até a primeira falha de inicialização ou memória.

No vLLM também é coletado o gauge de ocupação do pool KV em `/metrics`. Esse percentual não é percentual de VRAM nem bytes físicos; a VRAM é observada separadamente pelo `nvidia-smi`.

## Parâmetros dos runtimes

Argumentos extras são anexados ao `argv` e registrados no `lifecycle.json`:

```bash
make bench-vllm MODEL_SIZE=7B \
  VLLM_EXTRA_ARGS='--gpu-memory-utilization 1.0 --enforce-eager'

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
- `telemetry-timeseries.csv`: série temporal da GPU com `elapsed_s`, percentual de VRAM ocupada e fases, pronta para gráficos;
- `system.csv`: CPU, RAM, espaço e I/O host-wide;
- `events.csv`: timestamps das fases;
- `telemetry-summary.json`: médias e máximos por fase;
- `kv-cache.csv`: ocupação KV quando o endpoint existe;
- `server.log`: stdout/stderr do processo iniciado;
- `manifest.json` e `lifecycle.json`: versões, argumentos, modelo e falhas.

Os eventos relacionam picos de VRAM/CPU a startup, primeira requisição, aquecimento, medição ou encerramento. O instrumento não mede diretamente o tempo de cada transferência PCIe/RAM↔VRAM; isso exige Nsight/CUDA instrumentation.

### Baixar resultados do pod

Execute no computador local, não dentro do pod:

```bash
make pull-results POD_SSH=root@HOST_DO_POD
```

Com uma porta SSH diferente:

```bash
make pull-results POD_SSH=root@HOST_DO_POD POD_PORT=2222 \
  REMOTE_RESULTS_DIR=/workspace/llm_local_inference_masters/results \
  LOCAL_RESULTS_DIR=results-pod
```

O alvo usa `scp` e copia todas as execuções para `LOCAL_RESULTS_DIR`. Ele não apaga arquivos locais.

### Profiling para memory-bound/compute-bound

`gpu.csv` é uma série temporal de baixa intrusão. Para um roofline quantitativo, use uma execução separada com Nsight Systems (`nsys`) e Nsight Compute (`ncu`), pois eles medem kernels, FLOPs, bytes e duração com overhead. Verifique a instalação no pod com:

```bash
make check-profilers
```

Não misture Nsight com a bateria oficial de TTFT/Tokens/s: o profiler altera o tempo observado. Registre esse profiling como experimento complementar, com o mesmo modelo, prompt e contexto.

#### Makefile de profiling separado

O arquivo `Makefile.profiling` é deliberadamente separado do Makefile principal. Use-o em outro ambiente/execução, com `nsys` ou `ncu` instalados. Ele não instala nem remove o Nsight automaticamente e não deve ser usado para produzir os números oficiais de latência.

Exemplos no pod:

```bash
make -f Makefile.profiling check
make -f Makefile.profiling profile-vllm-nsys MODEL_SIZE=7B
```

O servidor fica em primeiro plano. Em outro terminal do pod, envie uma única requisição curta:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen7b-q8-gguf","messages":[{"role":"user","content":"Explique RAM e VRAM em duas frases."}],"max_tokens":64,"temperature":0,"stream":false}'
```

Depois interrompa o servidor com `Ctrl-C`. O relatório fica em `results/profiling/`. Para Nsight Compute, use `profile-vllm-ncu` ou `profile-llama-ncu`, faça uma única requisição e interrompa após os kernels desejados. O resultado `.ncu-rep` contém os contadores para estimar FLOPs, bytes de DRAM, FLOP/s, GB/s e intensidade aritmética.

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
