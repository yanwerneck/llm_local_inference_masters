# Bench explicado

## A ideia em uma frase

O benchmark mede a experiência de **uma pessoa usando um chatbot**, uma requisição por vez, observando o servidor por HTTP/SSE e registrando latência, tokens, memória e utilização da GPU.

Ele não é um teste de usuários simultâneos, não mede qualidade das respostas e não executa a rede neural diretamente. Quem executa a inferência é o runtime: vLLM, llama.cpp ou Ollama. O projeto prepara o ambiente, inicia o runtime quando solicitado, envia a carga e organiza as evidências.

## O mapa completo

```text
prepare-<runtime>
        │ modelo + tokenizer + configs + runtime validado
        ▼
smoke-<runtime> ── diagnóstico rápido
        │
        ▼
bench-<runtime>
        ├── bateria base: short / medium / long
        │     └── independent, replay ou closed-loop
        └── sweep de contexto/KV: 1024, 2048, 3072, ...
```

`bench-all` executa essa combinação para vLLM, llama.cpp e Ollama. `smoke-all` executa somente o diagnóstico rápido dos três.

## Não confunda os três conceitos

| Conceito | O que é | O que muda | Para que serve |
|---|---|---|---|
| Bateria base | `bench-<runtime>` chamando `bench.py` | cenários, repetições e fases no mesmo processo | medir TTFT, decode e latência ponta a ponta |
| `replay` | modo `BENCH_MODE=replay` da bateria base | envia histórico com respostas fixas do fixture | repetir exatamente uma conversa conhecida |
| `closed-loop` | modo conversacional da bateria base | coloca a resposta real no turno seguinte | estudar crescimento de histórico com respostas reais |
| Sweep de contexto/KV | `kv-sweep-<runtime>`, incluído no alvo formal | reinicia o servidor e aumenta o contexto a cada ponto | observar custo de contexto, VRAM, TTFT e falhas |

`replay` não é o sweep. Replay é uma política de mensagens; sweep é uma série de execuções com tamanhos de contexto diferentes. É possível usar `SWEEP_MODE=replay`, mas ainda serão dois eixos experimentais diferentes.

### Como funciona o modo `independent`

`independent` é o modo padrão da bateria base. Cada requisição contém somente uma mensagem do usuário, sem `system`/`assistant` acumulados de requisições anteriores:

```text
requisição 1: [user: prompt short]
requisição 2: [user: prompt short]
requisição 3: [user: prompt short]
```

O cliente não coloca a resposta da requisição 1 na requisição 2 e não envia um histórico conversacional. Para cada cenário, o prompt sintético é construído uma vez a partir da frase-base e reutilizado nas requisições daquele bloco; isso mantém a carga textual controlada. `short`, `medium` e `long` usam tamanhos de entrada diferentes.

Isso é independência **da conversa**, não garantia de estado frio. O servidor continua vivo durante warmup e measure e pode manter pesos na GPU, caches de prefixo, alocadores e estruturas internas. Assim, `independent` responde “quanto custa servir mensagens isoladas em um processo aquecido?”, não “quanto custa iniciar um runtime frio a cada requisição?”. Para cold starts, o processo precisa ser reiniciado entre execuções, como no sweep ou em chamadas separadas com `--launch`.

No `independent`, a resposta nunca altera a próxima entrada. No `replay`, o fixture fornece o histórico e respostas `assistant` fixas; no `closed-loop`, as respostas reais entram no turno seguinte. Essa é a diferença operacional entre os três modos.

## O que acontece em uma execução

### 1. Preparação — fora do relógio

`prepare-vllm`, `prepare-llama` e `prepare-ollama` dependem da etapa comum `prepare-benchmark`. Ela confere modelo, tokenizer, configs, imports, assinatura e arquivos locais. A preparação não inicia uma medição e não entra na latência.

```bash
make prepare-all MODEL_SIZE=7B
```

O alvo por runtime já chama a etapa comum. Não é necessário executar `prepare-benchmark` manualmente antes.

### 2. Partida e readiness

Com `--launch`, o benchmark inicia o processo do runtime em foreground e registra o grupo criado. Em seguida consulta `/v1/models` até encontrar o alias esperado. Isso mede `process_to_api_observed_s`, mas **API disponível não significa necessariamente modelo já residente na GPU**.

Se o servidor já existia e `--launch` não foi usado, a idade do processo e o carregamento dos pesos são desconhecidos. O cliente mede apenas a primeira requisição que ele observou.

### 3. Primeira resposta

O primeiro POST é enviado sem uma geração anterior escondida. O cliente mede:

- tempo do processo até o primeiro conteúdo;
- TTFT da requisição;
- tempo total até o fim do stream;
- `usage`, quando o servidor fornece contagem de tokens.

Essa é uma **sonda de primeira resposta**, registrada em `first-request.json` e no lifecycle. Ela não é misturada ao p50 da fase `measure`.

### 4. Warmup

O warmup envia requisições reais antes da medição formal. Elas podem aquecer compilação, buffers, alocadores, HTTP/SSE, tokenizer, prefixos e estruturas KV. Por padrão são três por cenário e repetição.

Warmup não é TTFT:

```text
TTFT  = uma métrica: POST → primeiro conteúdo daquela requisição
warmup = uma fase: requisições usadas para preparar o estado do processo
```

O warmup não limpa caches, não prova estabilidade e não entra na fase `measure`. Seus dados ficam registrados como `phase=warmup` para investigar tendência. Com `BENCH_WARMUP=0`, a execução fica mais barata, mas mais dependente do estado inicial.

### 5. Medição oficial

Na fase `measure`, o cliente envia uma requisição por vez. Para cada resposta bem-sucedida calcula TTFT, intervalo entre tokens, tokens/s de decode, tokens/s efetivos e latência total. A bateria formal padrão usa 50 requisições, três repetições e três warmups por cenário.

`short`, `medium` e `long` representam aproximadamente 256, 2048 e 8192 tokens de entrada, com teto de 128 tokens de saída. O servidor informa os comprimentos reais; esses valores devem ser conferidos antes da comparação.

### 6. Referência final e cleanup

O prompt inicial é repetido no final como referência aquecida. Depois, o launcher encerra somente o processo criado pelo benchmark e fecha os coletores. A referência ajuda a comparar o primeiro acesso com um estado posterior, mas pode aproveitar cache de prefixo; não isola sozinha um único custo.

## O que é TTFT, decode e latência total?

```text
envio do POST ───── primeiro conteúdo ───── últimos tokens ───── fim HTTP
                  ←──── TTFT ────→
                  ←──────── geração/decode ────────→
←──────────────── latência ponta a ponta ───────────────────────→
```

| Métrica | Fórmula conceitual | Inclui TTFT? |
|---|---|---:|
| TTFT | primeiro conteúdo − início do POST | é o próprio intervalo |
| Decode tokens/s | tokens de saída ÷ tempo entre primeiro e último token | não |
| Tokens/s efetivos | tokens de saída ÷ latência total | sim |
| Latência ponta a ponta | fim da resposta − início do POST | sim |

Uma configuração pode ter TTFT menor e decode mais lento. Também pode gerar menos tokens e parecer mais rápida na taxa efetiva. Por isso o relatório preserva contagens de entrada/saída e não compara uma métrica isolada.

## O segundo bench: sweep de contexto/KV

O sweep não continua a mesma amostra estatística da bateria base. Ele reinicia o runtime em cada ponto para alterar o limite de contexto e medir o efeito de cargas crescentes:

```text
contexto 1024 → inicia servidor → executa bench.py → encerra
contexto 2048 → inicia servidor → executa bench.py → encerra
contexto 3072 → inicia servidor → executa bench.py → encerra
```

Cada ponto tem startup, primeira resposta, warmup, medição, telemetria e diretório próprio. Por isso o sweep é mais lento: o custo dominante é reiniciar e carregar o runtime repetidamente, não apenas incrementar 1024 tokens.

Para exploração rápida:

```bash
make quick-sweep-vllm MODEL_SIZE=7B SWEEP_MAX_CONTEXT=1024
```

Para uma grade mais espaçada, altere `SWEEP_STEP=2048` ou `4096`. Para investigar uma região específica, use passo menor. O ponto falho não vira zero nem é substituído por outro modelo.

## Smoke, base e sweep: quando usar cada um

| Comando | Duração | O que valida | É resultado formal? |
|---|---:|---|---:|
| `smoke-<runtime>` | curta | processo, readiness, API, tokenizer, streaming e cleanup | não |
| `bench-<runtime>` | longa | bateria base + sweep do runtime | sim, se todos os blocos forem completos |
| `quick-sweep-<runtime>` | curta por ponto | integração do launcher e contexto | não; é exploratório |
| `bench-all` | mais longa | os três runtimes, continuando após falhas | sim, por runtime e protocolo |

## Como ler os resultados

Cada execução nova fica em:

```text
results/<runtime>/<timestamp>/<nome-humano>/
├── html/   summary.html, lifecycle.html e gráficos SVG
├── json/   manifestos, métricas e respostas brutas
├── csv/    GPU, CPU/RAM, eventos e requests
└── logs/   server.log e telemetry.log
```

Comece por `html/summary.html`. Depois confira `json/manifest.json` para status, configuração e contagens; `csv/gpu.csv` para memória/utilização; e `logs/server.log` para startup e erros. Uma mensagem no log de shutdown depois de HTTP 200 não deve ser classificada automaticamente como falha: confira o status do manifesto e o código do alvo.

## Perguntas que o benchmark responde

- O runtime está pronto e serve o alias esperado?
- Qual é o TTFT da primeira resposta e das requisições aquecidas?
- Como o tempo entre tokens e a latência total mudam com o cenário?
- O histórico conversacional real ou reproduzido muda a carga?
- Como memória GPU, GPU-utilização e TTFT se comportam quando o contexto cresce?
- Em qual ponto o runtime falha por startup, contexto ou memória?

Ele não responde sozinho quanto tempo cada kernel CUDA levou, se a GPU está memory-bound, qual runtime tem melhor qualidade de resposta ou como o sistema atende muitos usuários simultâneos. Essas perguntas exigem profiling, avaliação de qualidade ou um benchmark concorrente separado.
