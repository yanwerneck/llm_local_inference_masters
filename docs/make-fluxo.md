# Fluxo completo do Makefile

Este documento explica o que acontece quando cada alvo é executado. O objetivo é separar preparação, lançamento, medição e diagnóstico.

## 1. Seleção do experimento

`MODEL_SIZE` aceita `7B` ou `14B`. O Make deriva o GGUF, tokenizer, config JSON e launch JSON correspondentes. O padrão é `7B`, usando o artefato `arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf`.

```bash
make prepare-benchmark MODEL_SIZE=7B
make prepare-benchmark MODEL_SIZE=14B
```

Os caminhos podem ser sobrescritos:

```bash
make prepare-benchmark MODEL_SIZE=7B \
  MODEL_7B_GGUF=/workspace/models/arthur/qwen7b.gguf \
  MODEL_7B_TOKENIZER=/workspace/models/qwen7b-tokenizer
```

## 2. Preparação

`prepare-benchmark` depende de `install-benchmark`, `download-model`, `download-tokenizer` e `verify-gguf`.

### `install-benchmark`

Cria `BENCH_VENV` usando `SYSTEM_PYTHON` quando necessário, atualiza o pip e instala `requirements.txt`. O Python efetivamente usado pelos demais alvos é `PYTHON`. Esse ambiente é do cliente do benchmark; não o misture com o venv do vLLM ou com o ambiente de conversão do llama.cpp.

### `download-model`

Cria o diretório pai e executa `hf download` com `HF_HUB_OFFLINE=0`. O repositório e o nome do arquivo vêm de `HF_GGUF_REPO` e `HF_GGUF_FILENAME`. O alvo verifica que o arquivo chegou exatamente ao caminho esperado.

### `download-tokenizer`

Cria `TOKENIZER_DIR` e baixa apenas `config.json`, `tokenizer*`, `special_tokens_map.json` e `chat_template.jinja`. Não baixa pesos do checkpoint base.

### `verify-gguf`

Verifica presença, tamanho, assinatura binária `GGUF` e imprime SHA-256. Não interpreta os tensores e não inicia GPU.

### Validações finais

O alvo instala `requirements.txt` no `PYTHON` escolhido, importa as bibliotecas do cliente e valida JSONs. A presença do runtime é verificada separadamente por `prepare-vllm`, `prepare-llama` ou `prepare-ollama`; assim, preparar o modelo não é bloqueado por um runtime que ainda não foi instalado. No caso do llama.cpp, `prepare-llama` procura o `llama-server` já instalado no `PATH`, como ocorre na imagem do pod, e o launch JSON usa o comando `llama-server`. A compilação local fica disponível somente quando necessária, via `make build-llama`.
Os três templates seguem esse mesmo contrato: chamam `vllm`, `llama-server` e `ollama` pelo `PATH`, sem assumir caminhos internos de venv ou `/usr/local/bin`. Use `VLLM_BIN`, `LLAMA_SERVER_BIN` ou `OLLAMA_BIN` para sobrescrever o executável quando a imagem utilizar outro local.

## 3. Preparação do Ollama

`prepare-ollama` usa o arquivo GGUF local em um `Modelfile` temporário e executa `ollama create`. O alias é `qwen7b-q8-gguf` ou `qwen14b-q8-gguf`. Não há `ollama pull`: o modelo deve vir do SSD informado.

## 4. Bateria formal

`bench` chama `bench-all`. O agregador executa `bench-vllm`, `bench-llama` e `bench-ollama`, guarda um código de saída para cada um, continua após falhas e retorna código diferente de zero no final se algum falhou.

Cada alvo individual:

1. verifica o mesmo GGUF local;
2. inicia o runtime pelo launcher JSON;
3. espera `/v1/models` responder com o alias esperado;
4. mede a primeira requisição sem aquecimento oculto;
5. executa warmup e medida para `short`, `medium` e `long`;
6. mede TTFT, Tokens/s e velocidade efetiva;
7. coleta GPU, CPU, RAM, SSD, eventos e KV quando disponível;
8. encerra somente o grupo de processos criado;
9. executa o sweep de contexto/KV.

São 50 requisições por cenário, 3 repetições e 3 aquecimentos. O perfil é síncrono, portanto não é um benchmark de concorrência.

## 5. Argumentos experimentais

Os launchers JSON fornecem a configuração base. Variáveis `VLLM_EXTRA_ARGS`, `LLAMA_EXTRA_ARGS` e `OLLAMA_EXTRA_ARGS` são acrescentadas ao argv sem serem sanitizadas ou corrigidas.

```bash
make bench-vllm MODEL_SIZE=7B \
  VLLM_EXTRA_ARGS='--gpu-memory-utilization 0.80 --enforce-eager'
```

Esses argumentos aparecem no `lifecycle.json`. Se o runtime rejeitar a combinação, a execução falha e o `server.log` é a evidência.

## 6. Sweep de KV

Depois da bateria base, `run_kv_sweep.py` copia o launcher para uma pasta temporária, ajusta o limite de contexto e chama `bench.py` novamente em 1024, 2048, 3072, … tokens. Cada ponto tem seu próprio servidor e diretório de resultado.

- vLLM: altera `--max-model-len`;
- llama.cpp: altera `--ctx-size`;
- Ollama: mantém a política configurada externamente, pois `ollama serve` não possui uma flag universal equivalente.

O primeiro código de saída diferente de zero encerra o sweep. Nenhum ponto falho é convertido em zero ou substituído por outro modelo.

## 7. Onde estão os resultados

Cada execução tem um timestamp em `results/`. Os arquivos importantes são:

| Arquivo | Função |
|---|---|
| `summary.html` | leitura humana dos resultados |
| `summary.json` | métricas estruturadas |
| `gpu.csv` | amostras NVIDIA |
| `system.csv` | CPU/RAM/disco do host |
| `events.csv` | fases e timestamps |
| `telemetry-summary.json` | agregação por fase |
| `kv-cache.csv` | gauge KV do servidor, se existir |
| `server.log` | log do runtime iniciado |
| `lifecycle.json` | readiness, primeira resposta, argv e encerramento |
| `telemetry-timeseries.csv` | série temporal da GPU com `elapsed_s`, VRAM ocupada e utilização |

## 8. O que não é medido diretamente

`nvidia-smi` mostra VRAM total usada pela GPU, não bytes exclusivamente do KV. CPU/RAM/SSD são contadores observacionais do host. O benchmark não mede o tempo de cada cópia PCIe ou RAM↔VRAM; isso exige Nsight Systems/Compute ou instrumentação CUDA no runtime.

Para copiar os resultados do pod para o computador local, execute localmente: `make pull-results POD_SSH=root@HOST_DO_POD`. O alvo usa `scp`, aceita `POD_PORT`, `REMOTE_RESULTS_DIR` e `LOCAL_RESULTS_DIR`, e não apaga resultados locais. `make check-profilers` verifica `nsys` e `ncu`; profiling é complementar e deve ser executado separadamente, pois altera a latência.

## 9. Sequência recomendada

```bash
make prepare-benchmark MODEL_SIZE=7B
make prepare-ollama MODEL_SIZE=7B
make smoke-vllm MODEL_SIZE=7B
make bench-all MODEL_SIZE=7B
```

Só altere parâmetros depois de obter uma execução base válida. Cada alteração deve ser registrada na linha do Make e no `lifecycle.json`.
