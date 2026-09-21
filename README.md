# Chatbot Runtime Bench · um usuário

Repositório `yanwerneck/llm_local_inference_masters`: código do benchmark e materiais de estudo de inferência local. Abra [index.html](index.html) no navegador para navegar pelos guias offline.

## Materiais incluídos

- [Método do benchmark, incluindo inicialização e primeira resposta](docs/metodologia.html).
- [Git no RunPod](docs/git-runpod.html).
- [Guia de RunPod e vLLM](materiais/runpod-vllm-lab/index.html).
- [Arquitetura, gargalos e parâmetros de inferência](materiais/runpod-vllm-lab/estudo.html).
- [DDR, GDDR6X e HBM](materiais/runpod-vllm-lab/memorias.html).
- [Interface simples de chatbot em Python](materiais/runpod-vllm-lab/chat.html).
- [Revisão de privacidade antes da publicação](docs/seguranca-publicacao.md).

Os guias anteriores estão em `materiais/`, com seus fontes e scripts de geração. Suas instruções de instalação referem-se aos ambientes do servidor e do chat; **não misture essas dependências com o venv do benchmark**. Os HTMLs são arquivos estáticos: baixe/clonar e abra localmente; visualizar um arquivo no GitHub não ativa GitHub Pages.

Benchmark de latência HTTP com streaming para **vLLM, Ollama e llama-server**, usando **GuideLLM 0.7.4** nos blocos sequenciais e um observador HTTP simples para a inicialização e a primeira resposta. A versão 0.3 cobre início, aquecimento, TTFT, tokens/s, faixas de contexto/KV e GPU no relatório.

**Na versão 0.3, pesos já disponíveis no SSD são pré-requisito; download não faz parte do benchmark.** Não instala o runtime. Com `--launch`, inicia o comando fornecido em modo Hugging Face offline e encerra somente esse processo ao final. Não faz teste de concorrência, qualidade ou perplexidade. Veja a [explicação detalhada do código](docs/codigo-explicado.md).

Todo `run` exige `--local-model-path`: pasta com pesos HF, arquivo GGUF ou blob local de pesos do Ollama. Verificamos presença/tamanho e shards declarados antes do relógio, sem ler integralmente os pesos. Isso não prova o SSD físico, integridade ou vínculo com um servidor preexistente. Configure o runtime para o mesmo artefato local. Não use wrappers que baixem arquivos: variáveis offline de HF não são um firewall universal. Execução com download é inválida; não subtraímos tempos de internet.

Comece por [Como funciona o benchmark (HTML)](docs/metodologia.html). O tutorial de Git está separado: [HTML](docs/git-runpod.html) · [Markdown](docs/git-runpod.md).

## 1. Preparar o pod

Para testar um servidor existente, deixe-o funcionando em outro terminal. Para medir desde a partida, use `--launch`, explicado abaixo, com o servidor parado. Execute o cliente no mesmo pod para reduzir interferência da rede externa. Não rode outros servidores ou notebooks usando a GPU ao mesmo tempo.

Em uma imagem Ubuntu/Debian do RunPod, como root:

```bash
mkdir -p /workspace /workspace/models
apt-get update
apt-get install -y git python3-venv
cd /workspace
git clone https://github.com/yanwerneck/llm_local_inference_masters.git chatbot-runtime-bench
cd chatbot-runtime-bench
```

`apt-get update` atualiza o catálogo de pacotes; `install` instala Git e suporte a ambientes Python; `clone` baixa o código para a pasta local `chatbot-runtime-bench`, independentemente do nome do repositório remoto. Leia o tutorial separado para detalhes.

Use Python 3.12 no pod, de preferência. O cliente foi testado localmente com Python 3.14; os testes de integração não dependem de CUDA.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

O primeiro comando cria um ambiente **separado do vLLM**; `source` o ativa só neste terminal. O pip instala dependências dentro dele. O PyTorch CPU é usado pelo carregador de dados do GuideLLM, não para executar o modelo; instalá-lo antes evita baixar bibliotecas CUDA desnecessárias no Linux. `pip check` procura incompatibilidades entre pacotes instalados.

As dependências diretas centrais estão fixadas. Isso não é um lockfile completo e multiplataforma: cada execução salva todas as versões realmente instaladas em `client-packages.json`. Compare esse arquivo entre integrantes. Não atualize pacotes no meio da bateria experimental.

## 2. Baixar o GGUF e o tokenizer para o SSD

Modelo desta rodada: `arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf`. O GGUF contém apenas os pesos; use o tokenizer do modelo base Qwen2.5. O vLLM GGUF requer `vllm-gguf-plugin` e é experimental. Se não carregar, preserve o `server.log`, marque a execução como falha e investigue a incompatibilidade; não troque silenciosamente a representação.

```bash
source .venv/bin/activate
python -m pip install --upgrade huggingface_hub
mkdir -p /workspace/models /workspace/models/Qwen2.5-14B-tokenizer
hf download arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf \
  Qwen2.5-14B-Instruct-Q8_0.gguf \
  --revision main \
  --local-dir /workspace/models
hf download Qwen/Qwen2.5-14B-Instruct \
  --include 'config.json' 'tokenizer*' 'special_tokens_map.json' 'chat_template.jinja' \
  --revision main \
  --local-dir /workspace/models/Qwen2.5-14B-tokenizer
ls -lh /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
sha256sum /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
python -c 'import json; print("tokenizer model_type:", json.load(open("/workspace/models/Qwen2.5-14B-tokenizer/config.json"))["model_type"])'
```

`huggingface_hub` instala o comando `hf`; o primeiro `hf download` baixa somente o arquivo GGUF para o volume persistente, e o segundo baixa os arquivos de tokenizer **e o `config.json` mínimo exigido pelo vLLM**, incluindo `model_type: qwen2`. Nenhum servidor deve ser iniciado durante essa etapa. `ls` confirma o tamanho no SSD, `sha256sum` produz a impressão digital que deve ser registrada em `configs/*.json` e o último comando confirma que o vLLM conseguirá reconhecer a arquitetura do tokenizer.

Não é necessário executar `prepare-tokenizer` nem preencher nenhum `SHA_REGISTRADO_NO_SOURCE_JSON`: o comando `hf download` acima já colocou o tokenizer no caminho usado pelos três runtimes. O hash que precisamos controlar nesta rodada é o SHA-256 do GGUF. Compartilhe **a mesma pasta** de tokenizer entre os integrantes. O servidor também aplica seu chat template; compare os templates e as contagens reais retornadas, não só os nomes dos modelos.

## 3. Preparar a configuração do benchmark

Neste ponto ainda não existe servidor. Não execute `curl` agora: primeiro escolha um runtime e siga a seção correspondente em **4. Smoke test**, que mostra o comando exato para iniciar o servidor. Depois que o processo estiver em foreground e o log indicar que a API está disponível, o `curl` de cada runtime confirma o ID servido.

Edite o JSON com seu editor ou com `nano configs/vllm.json` (instale `nano` se não estiver disponível). Cada campo tem uma finalidade:

| Campo | O que registrar |
|---|---|
| `base_url` | Raiz HTTP do servidor; não coloque `/chat/completions`. |
| `model` | ID exatamente como aparece em `/v1/models`; os exemplos usam `qwen14b-q8-gguf`. |
| `tokenizer` | Pasta local; caminhos relativos são relativos ao diretório em que você executa o comando. |
| `context_window` | Limite realmente configurado no runtime. Mudar este JSON **não** reconfigura o servidor. |
| `cache_policy` | Política de reutilização entre requisições, após verificá-la. Ex.: `disabled-confirmed` ou `enabled-recorded: detalhes`. |
| `runtime_version` | Versão do vLLM/Ollama ou commit do llama.cpp. |
| `model_artifact` | Revisão do checkpoint ou SHA-256 do GGUF e tipo exato de quantização. |
| `server_command` | Comando/configuração de inicialização, **sem chaves ou senhas**. |
| `notes` | Alteração experimental, chat template, defaults de geração relevantes, particularidades do pod. |

Se a API exige autenticação, leia a chave sem gravá-la no histórico do shell Bash:

```bash
read -rs -p 'Chave da API: ' BENCH_API_KEY
export BENCH_API_KEY
```

A tecla Enter encerra a leitura. O Python obtém o valor do ambiente e o remove dos arquivos de configuração/resultado que grava. Não coloque segredos em `notes` ou `server_command`. Os resultados contêm prompts, respostas e detalhes do ambiente; revise antes de compartilhar.

## 4. Smoke test: provar que o caminho funciona

### O carregamento do modelo entra onde?

Há duas modalidades. Sem `--launch`, o benchmark conecta a um servidor já existente: **não mede a criação do processo nem pode afirmar quando os pesos foram carregados**. Ele mede apenas a primeira requisição observada por este cliente. Com `--launch`, o próprio benchmark inicia o comando em foreground e mede:

1. `process_to_api_observed_s`: criação do processo até `/v1/models` listar o alias. Isso mede prontidão HTTP, mas não prova que os pesos já estão residentes.
2. `process_to_first_content_s`: criação do processo até o primeiro conteúdo da primeira geração. Este é o indicador principal do custo de carregamento tardio, pois inclui leitura dos pesos locais, alocação, inicialização de kernels e compilação que ocorram antes/durante a primeira geração.
3. `process_to_response_end_s`: criação do processo até o fim da primeira resposta; inclui também o decode dessa resposta.

Para os três runtimes, execute o processo real do servidor via um arquivo `--launch` diferente, sempre parado antes do teste:

| Runtime | Comando foreground a colocar no arquivo `--launch` | Marco de carregamento |
|---|---|---|
| vLLM | `vllm serve ...` com o caminho local do GGUF e tokenizer correspondente | processo → primeiro conteúdo; `/v1/models` pode anteceder a carga completa |
| llama.cpp | `llama-server -m /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf ...` | processo → primeiro conteúdo; o servidor normalmente carrega o GGUF no startup |
| Ollama | `ollama serve` | processo → primeiro conteúdo, porque `ollama serve` pode ficar pronto antes de `ollama run` carregar o modelo |

O arquivo deve conter apenas um array JSON de argumentos, sem `source`, `&`, `docker -d`, pipes ou redirecionamentos. Para Ollama, o benchmark precisa conseguir alcançar `/v1/models` e `/v1/chat/completions`; se `ollama serve` não resolver o modelo sozinho, use um wrapper foreground documentado que mantenha o processo e faça o preload local sem baixar arquivos. O wrapper deve receber as variáveis offline apropriadas e ser validado no `server.log`.

O download do modelo continua fora do benchmark: prepare os arquivos/blob no SSD antes de iniciar. Se o runtime baixar depois da criação do processo, a execução não é comparável entre integrantes e deve ser marcada como inválida, não “corrigida” subtraindo uma estimativa de internet.

### Protocolo por runtime: comandos exatos

Os três procedimentos abaixo usam o mesmo arquivo local:

```text
/workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
```

O diretório `/workspace/models` precisa existir e o arquivo precisa estar lá **antes** da medição. Rode `sha256sum /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf` e registre o resultado em `model_artifact` nas configurações. Não há troca automática de formato: se um runtime falhar ao carregar esse GGUF, preserve o erro e trate a execução como falha diagnóstica.

Em todos os casos, use dois terminais: o primeiro mantém o servidor em foreground; o segundo verifica a API e executa o benchmark. O benchmark deve ser executado com o servidor parado quando usar `--launch`.

#### A. vLLM

**Preparação (uma vez no pod):** o vLLM deve estar instalado no ambiente próprio do runtime. O cliente do benchmark fica em outro `.venv`.

```bash
mkdir -p /workspace/vllm-runtime
# Se o ambiente ainda não existir:
python3 -m venv /workspace/vllm-runtime/.venv
source /workspace/vllm-runtime/.venv/bin/activate
unset VLLM_VENV
python -m pip install -U vllm vllm-gguf-plugin
python -c 'import vllm; print(vllm.__version__)'
nvidia-smi
vllm serve --help > /workspace/vllm-runtime/vllm-serve-help.txt
```

O `vllm-gguf-plugin` é necessário para o caminho GGUF em versões que o exigem. O último comando salva a interface efetivamente instalada; se a importação falhar por incompatibilidade CUDA, não mascare o erro instalando outro modelo.

**Servidor exato (Terminal 1):**

```bash
source /workspace/vllm-runtime/.venv/bin/activate
vllm serve /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --tokenizer /workspace/models/Qwen2.5-14B-tokenizer \
  --served-model-name qwen14b-q8-gguf \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096
```

`serve` cria o processo; o caminho após `serve` é o GGUF; `--tokenizer` aponta para o tokenizer local; `--served-model-name` define o ID que aparecerá na API; `--host`/`--port` definem a API; `--max-model-len` fixa o contexto usado nesta rodada. O processo fica em foreground para que `server.log` preserve o carregamento, a alocação e qualquer erro.

**Verificação e benchmark (Terminal 2):**

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/metrics | head
source .venv/bin/activate
python bench.py run \
  --config configs/vllm.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --smoke --scenarios short
```

O primeiro `curl` confirma o ID `qwen14b-q8-gguf`; o segundo confirma se há métricas Prometheus. Sem `--launch`, esse smoke mede um servidor já iniciado. Para medir também a partida, pare o servidor e use o arquivo `configs/launch-vllm.example.json`, que contém exatamente o mesmo comando acima:

```bash
source .venv/bin/activate
python bench.py run \
  --config configs/vllm.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --launch configs/launch-vllm.example.json \
  --smoke --scenarios short --startup-timeout 1800
```

Nesse modo, `process_to_api_observed_s` é o processo até a API responder, enquanto `process_to_first_content_s` inclui o custo observado até o primeiro token: leitura do GGUF, alocação, kernels e eventuais compilações tardias. A API pronta não prova, sozinha, que os pesos já estão residentes.

#### B. llama.cpp (`llama-server`)

**Compilar (uma vez no pod):**

```bash
cd /workspace
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /workspace/llama.cpp/build --config Release -j "$(nproc)"
/workspace/llama.cpp/build/bin/llama-server --help > /workspace/llama-server-help.txt
```

`GGML_CUDA=ON` compila os kernels CUDA; `cmake --build` gera o executável; `llama-server --help` registra as opções da versão usada. Registre também o commit: `git -C /workspace/llama.cpp rev-parse HEAD`.

**Servidor exato (Terminal 1):**

```bash
/workspace/llama.cpp/build/bin/llama-server \
  -m /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --alias qwen14b-q8-gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 4096 \
  -ngl 99 \
  --parallel 1 \
  --metrics
```

`-m` seleciona o GGUF local; `--alias` define o ID da API; `-c` fixa o contexto; `-ngl 99` solicita que todas as camadas possíveis sejam descarregadas na GPU; `--parallel 1` mantém uma requisição simultânea; `--metrics` habilita métricas do servidor quando suportado. Se a versão compilada rejeitar uma flag, preserve a saída de erro e a versão em vez de trocar silenciosamente o comando.

**Verificação e benchmark (Terminal 2):**

```bash
curl http://127.0.0.1:8080/v1/models
source .venv/bin/activate
python bench.py run \
  --config configs/llamacpp.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --smoke --scenarios short
```

O `curl` confirma que o alias servido corresponde ao campo `model` de `configs/llamacpp.json`. Para medir a partida, pare o `llama-server` e crie `configs/launch-llamacpp.json` com este array JSON (cada item é um argumento separado):

```json
[
  "/workspace/llama.cpp/build/bin/llama-server",
  "-m", "/workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf",
  "--alias", "qwen14b-q8-gguf",
  "--host", "127.0.0.1",
  "--port", "8080",
  "-c", "4096",
  "-ngl", "99",
  "--parallel", "1",
  "--metrics"
]
```

Execute o ciclo de vida com:

```bash
source .venv/bin/activate
python bench.py run \
  --config configs/llamacpp.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --launch configs/launch-llamacpp.json \
  --smoke --scenarios short --startup-timeout 1800
```

Aqui o tempo processo → primeiro conteúdo inclui o carregamento do GGUF e a preparação da GPU feita no início pelo `llama-server`. O arquivo `server.log` é a evidência para saber se todas as camadas foram para a GPU, se houve offload parcial ou se o processo falhou.

#### C. Ollama

**Importar o mesmo GGUF local (uma vez no pod):** crie um `Modelfile` sem URL e sem `ollama pull`:

```bash
cat > /workspace/Modelfile-qwen14b <<'EOF'
FROM /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
PARAMETER num_ctx 4096
EOF
ollama create qwen14b-q8-gguf -f /workspace/Modelfile-qwen14b
ollama show qwen14b-q8-gguf
ollama list
```

`FROM` importa o arquivo que já está no SSD; `num_ctx` fixa o contexto do modelo; `ollama create` prepara o blob local antes do relógio. `ollama show` e `ollama list` confirmam o nome e a origem. Se o Ollama não aceitar o GGUF, preserve a saída e marque a execução como falha.

**Servidor exato (Terminal 1):**

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

`ollama serve` fica em foreground e expõe a API em `127.0.0.1:11434`. Ele pode responder à saúde antes de carregar o modelo; por isso o benchmark mede também o primeiro conteúdo, quando o custo de carregamento tardio aparece. Não execute `ollama pull` durante o benchmark.

**Verificação e benchmark (Terminal 2):**

```bash
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/v1/models
source .venv/bin/activate
python bench.py run \
  --config configs/ollama.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --smoke --scenarios short
```

O primeiro `curl` verifica o catálogo nativo; o segundo verifica a compatibilidade OpenAI usada pelo cliente. Preencha `configs/ollama.json` com `model: qwen14b-q8-gguf` e `base_url: http://127.0.0.1:11434`. Para medir a partida, pare o serviço e crie `configs/launch-ollama.json`:

```json
["ollama", "serve"]
```

O array não contém `export`, `&` ou redirecionamentos: o benchmark inicia somente esse processo em foreground. Como `OLLAMA_HOST` já foi exportada no Terminal 1, o processo lançado herda a porta 11434. Se quiser que o próprio benchmark seja o primeiro processo Ollama, exporte `OLLAMA_HOST=127.0.0.1:11434` no Terminal 2 antes do `bench.py` e mantenha a porta livre.

Rode:

```bash
source .venv/bin/activate
python bench.py run \
  --config configs/ollama.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --launch configs/launch-ollama.json \
  --smoke --scenarios short --startup-timeout 1800
```

O `ollama create` fica fora da medição; o `--launch` mede apenas o processo `ollama serve` e o carregamento tardio provocado pela primeira requisição. Registre `load_duration` retornado pelo Ollama quando disponível. Se o servidor ficar pronto, mas a primeira geração falhar ou o modelo for descarregado, a execução é falha, não um resultado válido.

#### Comparação justa dos três comandos

Para cada runtime, mantenha o mesmo GGUF, a mesma revisão do tokenizer quando aplicável, contexto 4096, uma requisição em andamento e prompts idênticos. Use os três comandos com `--launch` para comparar inicialização, e os três sem `--launch` somente para comparar operação contra servidores já aquecidos. Salve a versão (`vllm --version`, commit do llama.cpp, `ollama version`), o comando, o SHA-256 e os logs. Não combine resultados se um runtime tiver falhado, feito download durante a execução ou usado outro artefato.

Execute o mesmo smoke, trocando apenas a configuração e o arquivo `--launch`:

```bash
# vLLM
python bench.py run --config configs/vllm.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-vllm.example.json --smoke --scenarios short

# llama.cpp
python bench.py run --config configs/llamacpp.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-llamacpp.json --smoke --scenarios short

# Ollama
python bench.py run --config configs/ollama.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-ollama.json --smoke --scenarios short
```

Faz GET de disponibilidade (sem gerar texto), **uma primeira requisição já cronometrada**, três requisições de aquecimento, três medições GuideLLM e uma referência final com o mesmo prompt inicial. Sempre uma de cada vez. A primeira requisição também verifica SSE/usage; não há um POST oculto de preflight antes dela. O aquecimento agora aparece no resumo, identificado por `phase=warmup`.

O smoke aceita metadados ainda marcados como `PREENCHER`; **não é um resultado final**. Falhas de protocolo ou requisições incompletas geram saída diferente de zero e ficam registradas. Interromper com Ctrl+C preserva fases já concluídas.

Uma execução só pode ser analisada como válida se cada bloco terminar com a quantidade esperada de sucessos. Se o GuideLLM registrar, por exemplo, 2 de 3 requisições, trate o bloco como falho/incompleto e não como “validação real aprovada”; consulte o JSON bruto e o erro antes de repetir. Os testes simulados e um smoke concluído verificam o caminho do instrumento, mas não comprovam que cada runtime/modelo real está validado no pod.

Esse caso de amostra faltante continua sendo um diagnóstico do caminho GuideLLM/servidor, não uma correção já demonstrada para todos os runtimes. Até haver evidência reproduzível de execução completa, mantenha a bateria marcada como falha.

O cliente inclui uma compatibilidade estreita para o GuideLLM 0.7.4: após o encerramento sinalizado, ela drena por até cinco segundos uma atualização terminal real que tenha chegado atrasada à fila. Não repete requisições nem cria métricas e não substitui a guarda de contagem; se a atualização não chegar, o bloco continua falhando.

### 4.1. Medir desde a partida de qualquer runtime

Pare manualmente o servidor que você iniciou. O benchmark **não mata um servidor existente**: se a porta já estiver ocupada, recusa o lançamento. Use o arquivo `launch` correspondente ao runtime (`configs/launch-vllm.example.json`, `configs/launch-llamacpp.json` ou `configs/launch-ollama.json`) e mantenha nele apenas argumentos JSON, sem `source`, `&`, redirecionamentos ou comandos de shell.

Os exemplos contêm caminhos do pod e contexto 4096, mas não são tuning universal. O comando escrito em `server_command` continua sendo metadado: **só o arquivo passado com `--launch` é executado**. Não ponha segredos nesses arquivos; use variáveis de ambiente apropriadas ao runtime.

```bash
# vLLM
python bench.py run --config configs/vllm.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-vllm.example.json --smoke --scenarios short --startup-timeout 1800 --initial-state 'GGUF no SSD; caches não limpos'

# llama.cpp
python bench.py run --config configs/llamacpp.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-llamacpp.json --smoke --scenarios short --startup-timeout 1800 --initial-state 'GGUF no SSD; caches não limpos'

# Ollama
python bench.py run --config configs/ollama.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-ollama.json --smoke --scenarios short --startup-timeout 1800 --initial-state 'GGUF no SSD; caches não limpos'
```

Esse comando:

1. Inicia a amostragem da GPU e lança seu runtime em primeiro plano, sem shell.
2. Cronometra desde a criação do processo até `/v1/models` listar o modelo.
3. Envia o primeiro POST de inferência e mede TTFT, resposta total e tempo desde a partida até o primeiro conteúdo e fim da resposta.
4. Mede aquecimento e blocos GuideLLM, depois repete o prompt inicial como referência aquecida.
5. Salva os relatórios e encerra **somente o grupo do processo que criou**, inclusive se o benchmark falhar.

`--startup-timeout` limita a espera pela API. `--timeout` é o limite de espera de leitura HTTP, não um prazo total rígido da bateria. Durante a inicialização, veja `results/DATA/server.log`; os logs do runtime podem conter informações sensíveis e não recebem filtragem completa. Revise antes de compartilhar.

Só são aceitos destinos locais no modo `--launch`. Use executáveis diretos/foreground, não serviços já em execução, `docker -d` ou wrappers que se desconectam do processo. Para Ollama e llama.cpp, crie arrays equivalentes com o comando correto do seu ambiente. Um Ollama novo pode carregar o modelo somente na primeira geração; a métrica processo → primeiro conteúdo captura esse custo, enquanto processo → API não prova residência dos pesos.

**Uma execução produz uma amostra de inicialização.** `--repetitions 3` repete blocos GuideLLM sem reiniciar o servidor. Para três partidas, execute o comando com `--launch` três vezes, preferindo `--repetitions 1`; cada chamada gera outra pasta e encerra seu próprio servidor. Não tire um p95 de cold start de uma única partida.

### 4.2. O que significa “frio”?

Um processo novo não garante cache de disco ou compilação CUDA frios. `--initial-state` registra as condições, **não apaga caches**. Sem `--launch`, medimos a primeira requisição deste cliente, mas não sabemos se o servidor já foi aquecido por alguém.

Downloads, provisionamento e instalação ficam na preparação, fora do experimento. Carregamento dos pesos locais, inicialização de kernels e compilação continuam dentro da medição. Separar SSD, PCIe e compilação exige logs/profiling; este cliente não inventa essa decomposição.

O prompt inicial padrão é uma pergunta em português sobre RAM/VRAM com limite de 128 tokens de saída. Para usar seu próprio caso de chatbot:

```bash
python bench.py run --config configs/vllm.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-vllm.example.json --first-prompt-file pergunta.txt --scenarios short --requests 30 --repetitions 1
```

Para repetir a mesma pergunta nos outros runtimes:

```bash
python bench.py run --config configs/llamacpp.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-llamacpp.json --first-prompt-file pergunta.txt --scenarios short --requests 30 --repetitions 1
python bench.py run --config configs/ollama.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --launch configs/launch-ollama.json --first-prompt-file pergunta.txt --scenarios short --requests 30 --repetitions 1
```

Crie `pergunta.txt` em UTF-8 com seu editor e use o mesmo conteúdo nos três runtimes. Esse arquivo não passa por truncamento automático; confirme que cabe no contexto. A referência final repete exatamente esse prompt e pode se beneficiar de cache de prefixo: documente a política ao comparar frio/quente.

### 4.3. Como ler as velocidades

`TTFT` é o tempo do envio até o primeiro conteúdo não vazio observado. `decode_tokens_s` é a velocidade média depois desse primeiro conteúdo (`1000 / mean_itl_ms`), usando o intervalo entre o primeiro e o último token; não inclui o tempo inicial. `effective_tokens_s` é a saída dividida pela duração total da requisição e inclui TTFT. Portanto uma resposta pode ter decode rápido e velocidade efetiva menor por causa do prefill/espera inicial. Nenhuma dessas colunas é throughput agregado de vários usuários.

As faixas de contexto e `--kv-bytes-per-token` são um proxy lógico da carga: estimam bytes por token a partir da arquitetura e do dtype informados, não medem bytes físicos alocados. Já `--collect-kv-metrics` registra, quando o servidor expõe o gauge, a fração ocupada do pool KV; isso também não é percentual de VRAM. `nvidia-smi` mostra memória total usada na GPU, sem atribuí-la exclusivamente ao KV ou ao runtime.

## 5. Primeira bateria: curta e média

Depois de preencher os metadados e confirmar o smoke de cada runtime, rode a bateria curta/média. Pare os três servidores manuais antes de usar `--launch`.

```bash
python bench.py run --config configs/vllm.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --scenarios short medium --requests 30 --repetitions 3
```

Repita exatamente a mesma bateria para os outros runtimes:

```bash
python bench.py run --config configs/llamacpp.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --scenarios short medium --requests 30 --repetitions 3
python bench.py run --config configs/ollama.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --scenarios short medium --requests 30 --repetitions 3
```

Isso faz 30 medições por cenário por repetição: **180 requisições na fase measure**, mais aquecimento medido, primeira resposta e referência final. Há apenas **uma requisição em andamento**, não 30 usuários. As três repetições ajudam a observar variação entre blocos; não são três réplicas independentes de hardware nem três partidas.

As sementes variam por cenário/repetição e são iguais entre runtimes quando os argumentos são iguais. O aquecimento usa sementes diferentes. A ordem dos cenários gira entre repetições para reduzir, sem eliminar, efeitos de posição e aquecimento térmico.

## 6. Cenário longo, só depois

```bash
python bench.py run --config configs/vllm.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --scenarios short medium long --requests 100 --repetitions 3
```

Para llama.cpp e Ollama, use os mesmos argumentos e apenas troque a configuração:

```bash
python bench.py run --config configs/llamacpp.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --scenarios short medium long --requests 100 --repetitions 3
python bench.py run --config configs/ollama.json --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf --scenarios short medium long --requests 100 --repetitions 3
```

São **900 medições**; isso pode levar bastante tempo e consumir horas cobradas no RunPod. Primeiro valide os cenários menores. O longo usa aproximadamente 8192 tokens de conteúdo e teto de 128 tokens de saída. O guard exige `context_window >= 8576`, reservando 256 tokens para template; essa margem não comprova que o modelo cabe na VRAM. Configure e valide o servidor, por exemplo com contexto de 9216 ou maior se houver memória, antes de mudar o JSON.

Não suponha que a RTX 3090 comporte qualquer contexto com um modelo de 14B em 8 bits. Reduza o escopo se faltar memória e registre o cenário como não suportado; não o omita silenciosamente da comparação.

## 7. Ollama e llama.cpp

### Grade de contexto/KV (inclusive no vLLM)

```bash
python bench.py run --config configs/vllm.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --input-tokens 256 512 1024 2048 3072 \
  --collect-kv-metrics --requests 30 --repetitions 3
```

Rode a mesma grade para cada runtime, mantendo o servidor correspondente iniciado e a porta/configuração corretas:

```bash
python bench.py run --config configs/llamacpp.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --input-tokens 256 512 1024 2048 3072 \
  --requests 30 --repetitions 3

python bench.py run --config configs/ollama.json \
  --local-model-path /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf \
  --input-tokens 256 512 1024 2048 3072 \
  --requests 30 --repetitions 3
```

`--input-tokens` substitui os cenários fixos. O template soma tokens; as faixas no HTML usam a entrada real. `--collect-kv-metrics` coleta ocupação real do pool KV em `/metrics`, quando disponível no vLLM; não é percentual da VRAM. Mantenha a coleta igual entre execuções, pois tem custo.

O parâmetro `--collect-kv-metrics` só produz dados quando o runtime expõe o gauge esperado; mantenha-o no vLLM e registre a ausência nos outros dois. A grade de `--input-tokens`, TTFT, tokens/s, tokens reais, erros e GPU continua comparável entre os três.

Para adicionar MiB de KV lógico **estimado**, use `--kv-bytes-per-token N`: `N = 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento`, para uma sequência e atenção completa. Use arquitetura e dtype real do cache, não os bits dos pesos. Sem o coeficiente, não inventamos MiB; reserva, blocos e arquiteturas diferentes não são cobertos pela estimativa.

Use os mesmos comandos com `--config configs/ollama.json` ou `--config configs/llamacpp.json`. Preencha o ID real do modelo, a versão e o artefato. O servidor deve expor `/v1/models` e `/v1/chat/completions` com streaming e usage.

- Ollama: configure o contexto no modelo/servidor e mantenha o modelo carregado durante a bateria. Não assuma que o JSON deste cliente muda `num_ctx`.
- llama.cpp: execute `llama-server`, não o CLI interativo; confira o alias do modelo e o chat template.
- vLLM: use o servidor que você já instalou. Não instale o runtime no venv do benchmark.

Use o mesmo caminho/hash do GGUF nas três configurações. Se isso não for possível, não combine as tabelas nem atribua a diferença exclusivamente ao runtime.

## 8. Onde estão os resultados?

Cada execução cria `results/DATA_UTC/`, sem sobrescrever execuções anteriores:

| Arquivo | Conteúdo |
|---|---|
| `lifecycle.html` / `lifecycle.json` | Inicialização, primeira resposta, referência final e estado observado. |
| `first-request.json` / `warm-reference.json` | Tempos, prompt, resposta e offsets de eventos SSE dessas requisições. Preserva tempos parciais se falharem. |
| `server.log` | stdout/stderr do runtime, somente com `--launch`; examine antes de compartilhar. |
| `summary.html` | Relatório integrado: partida, TTFT, tokens/s, faixas de contexto/KV e GPU por fase. |
| `context-summary.json` | Velocidades por faixa real de entrada; MiB lógicos estimados quando configurados. |
| `gpu-summary.json` | Memória, utilização, temperatura e potência por GPU/fase. |
| `kv-cache.csv` | Ocupação do pool KV via `/metrics`, se solicitada. |
| `summary.csv` / `summary.json` | Uma linha por fase, cenário e repetição; filtre `phase` na análise. |
| `r1-short-measure.json` | Relatório bruto GuideLLM, requisições, tempos, textos e contagens. |
| `*-requests.csv` | Uma linha por requisição, incluindo status, para análise no R/Python. |
| `r1-short-warmup.json` | Aquecimento, separado. |
| `*-config.json` | Argumentos da fase, com perfil síncrono e sementes. |
| `manifest.json` | Configuração, status final, hash do tokenizer e ambiente. |
| `client-packages.json` | Versões reais do cliente. Não confundir com versões do runtime em outro venv. |
| `gpu.csv` | Amostras locais de memória, utilização, temperatura e potência, aproximadamente a cada segundo. |
| `gpu-before.json` / `gpu-after.json` | Snapshots de `nvidia-smi`; ausência de GPU não impede testar o cliente. |

Baixe a pasta de resultados para o computador e abra `summary.html`. Não precisa servir a página publicamente. O monitor coleta **todas as GPUs visíveis** e não identifica processos; o máximo observado não é um pico exato nem memória exclusivamente atribuível ao runtime. Se usar servidor remoto, esse monitor mede o host do cliente, não o servidor.

Para comparar: confira `status=complete`, ausência de erros, hashes iguais de tokenizer e de requisições (`requests_sha256`), tamanhos reais semelhantes de saída, mesma GPU e política de cache documentada. Não compare diretamente linhas com saídas de comprimentos muito diferentes.

## 9. Testes do projeto

```bash
python -m unittest discover -s tests -v
python tests/integration_mock.py
```

O primeiro testa validação, estatísticas, lançamento seguro e cronometria inicial com servidor simulado. O segundo usa o **GuideLLM instalado**, um tokenizer local de teste e HTTP/SSE; verifica concorrência máxima 1, primeira resposta e aquecimento identificado no resumo. Não mede desempenho de GPU, não valida os três runtimes reais e não substitui o smoke no RunPod.

## Referências

- [GuideLLM: código e releases](https://github.com/vllm-project/guidellm) — ferramenta aberta do ecossistema vLLM, não uma certificação ou padrão universal de benchmark.
- [Perfil synchronous e execução](https://vllm-project.github.io/guidellm/0.7.0/getting-started/benchmark/) — explicitamos o perfil; o default sweep não é adequado a este trabalho.
- [Backends HTTP](https://vllm-project.github.io/guidellm/0.7.0/guides/backends/) e [datasets](https://vllm-project.github.io/guidellm/0.7.0/guides/datasets/).
- [Compatibilidade OpenAI do Ollama](https://docs.ollama.com/api/openai-compatibility).
- [Código das métricas, versão 0.7.4](https://github.com/vllm-project/guidellm/blob/v0.7.4/src/guidellm/schemas/request_stats.py).

## Limites da validação

Este repositório inclui testes locais, mas não contém resultados reais da RTX 3090 e não afirma que os três servidores já foram validados no seu pod. Faça o smoke em cada um. Consulte [metodologia](docs/metodologia.html) antes de interpretar as tabelas.
