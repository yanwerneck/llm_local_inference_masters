# Um usuário. Três runtimes. Um instrumento.

Este estudo pergunta: **como o runtime e seus parâmetros mudam o tempo de resposta de um chatbot para uma pessoa?** A carga de múltiplos usuários fica para o próximo trabalho.

## 1. Quem faz o quê?

```text
Este projeto → GuideLLM → API HTTP do runtime → modelo na GPU
                            ↓ respostas em streaming
               tempos + contagens + relatórios
```

O runtime pode ser vLLM, Ollama ou llama-server. O GuideLLM organiza e mede os blocos sequenciais. Nosso código também observa a partida do processo (quando solicitada) e cronometra o primeiro stream HTTP com relógio monotônico. Isso evita aquecer o servidor antes de medir o primeiro contato. Essas métricas iniciais são próprias do cliente; não as apresentamos como métricas internas de kernels nem como resultados produzidos pelo GuideLLM.

Escolhemos [GuideLLM](https://github.com/vllm-project/guidellm) por ser uma ferramenta existente no ecossistema de serving de LLMs e usar um protocolo comum. Não há a pretensão de que este projeto seja um padrão industrial certificado. **Ferramenta padronizada não torna automaticamente o experimento comparável:** os controles continuam sendo responsabilidade do grupo.

## 2. O que “um usuário” significa aqui?

Uma mensagem é enviada; a resposta termina; só então enviamos a próxima. Não há requisições sobrepostas. Isso é uma carga sequencial fechada, sem tempo de reflexão entre mensagens. Não é um robô simulando toda a rotina humana, apenas um instrumento para medir serviço individual.

O perfil GuideLLM é explicitamente `synchronous`. Não usamos o default `sweep`, nem taxa de chegada Poisson, nem perfil throughput. A documentação do [perfil sequencial](https://vllm-project.github.io/guidellm/0.7.0/getting-started/benchmark/) explica essa distinção.

Uma execução do cliente por vez não impede interferência externa: outro terminal, processo ou usuário no mesmo hardware pode contaminar o experimento. A trava de arquivo evita duas execuções deste projeto usando a mesma pasta de resultados, mas não controla o pod inteiro.

## 3. Qual carga será usada?

O próprio GuideLLM gera texto sintético com um tokenizer fixado. Usamos aproximadamente 256, 2048 e, opcionalmente, 8192 tokens de conteúdo, com teto de 128 tokens na resposta. O [gerador de datasets](https://vllm-project.github.io/guidellm/0.7.0/guides/datasets/) fornece uma carga repetível por semente.

| Cenário | Conteúdo de entrada | Teto de saída | Pergunta experimental |
|---|---:|---:|---|
| short | ~256 tokens | 128 | Quanto demora uma interação curta? |
| medium | ~2048 tokens | 128 | Como aumenta o custo de processar contexto? |
| long | ~8192 tokens | 128 | O que acontece com uma entrada grande, se couber? |

Esses números são **nosso desenho**, não limites oficiais de classificação de chatbot. Tokens de template e mensagens de sistema podem aumentar a entrada total. O servidor devolve as contagens reais, que devem ser analisadas.

O texto não é um dataset validado de perguntas em português, nem uma conversa real com histórico crescente. Cada requisição é independente. O cenário longo **representa o volume de contexto**, mas não avalia reaproveitamento de uma conversa em vários turnos. Esse escopo evita misturar dois fenômenos no primeiro experimento.

O modelo pode encerrar antes de 128 tokens. Não forçamos ignore-EOS porque não é um parâmetro portátil entre os três servidores. Registrar saída efetiva é obrigatório: uma configuração pode parecer mais rápida apenas por gerar menos texto. Textos sintéticos também podem provocar respostas curtas ou estranhas; confirme os comprimentos no piloto antes de investir na bateria inteira.

## 4. Como uma execução acontece

1. Validamos configuração, versão do instrumento e arquivos do tokenizer, antes de iniciar o runtime.
2. Com `--launch`, lançamos o processo e medimos até a API listar o modelo. Sem essa opção, a partida anterior é desconhecida e não recebe um tempo inventado.
3. Enviamos **a primeira requisição já cronometrada**, que também valida streaming e usage. Não há teste de geração anterior.
4. Para cada cenário/repetição, medimos o bloco de aquecimento e o bloco posterior, identificados separadamente no resumo.
5. Repetimos o mesmo prompt inicial como referência final aquecida, salvamos os resultados e encerramos somente o processo iniciado por nós.

**Nada disso é descartado:** inicialização e primeira resposta ficam em `lifecycle.html`; aquecimento e blocos posteriores em `summary.html`, com a coluna `phase`. Separar fases não é excluí-las. Não as misturamos em uma única média porque respondem perguntas diferentes.

```text
PARTIDA DO PROCESSO  (--launch)
        │ inicialização, possível carga/compilação/download
        ▼
API LISTA O MODELO   (não necessariamente já residente na GPU)
        │ primeiro POST medido; possível carregamento tardio
        ▼
PRIMEIRO CONTEÚDO → FIM DA PRIMEIRA RESPOSTA
        │
        ▼
AQUECIMENTO MEDIDO → BLOCOS POSTERIORES → REFERÊNCIA FINAL
```

### As métricas iniciais

| Métrica | O que cobre |
|---|---|
| `process_to_api_observed_s` | Da chamada de criação do processo até o primeiro GET válido que lista o modelo. |
| `first_request.ttft_ms` | Do envio do primeiro POST até conteúdo não vazio; não termina ao receber apenas o campo role. |
| `first_request.e2e_s` | Duração total do primeiro stream, incluindo custo inicial que ocorra ali. |
| `process_to_first_content_s` | Da partida até aparecer o primeiro texto: medida útil para comparar runtimes com carregamento antecipado ou tardio. |
| `process_to_response_end_s` | Da partida até terminar a primeira resposta. |
| `warm_reference` | O mesmo prompt inicial repetido ao final, com o mesmo cliente observador. |

O polling de disponibilidade usa intervalos de até 0,5 s, além do tempo de rede/timeout dos GETs. Portanto, medimos **quando observamos a API disponível**, não o instante interno exato da prontidão. Os intervalos processo → conteúdo/fim incluem essa detecção e pequenos custos do cliente entre etapas. Nenhuma diferença de poucos milissegundos nesse intervalo deve ser atribuída automaticamente ao runtime.

O tempo até a API responder não é sozinho uma comparação justa de carregamento. Alguns servidores só carregam o modelo quando recebem a primeira geração. Compare também processo → primeiro conteúdo. O Ollama oferece métricas nativas como `load_duration`, mas elas não são um campo comum a todas as APIs; veja a [documentação de métricas do Ollama](https://docs.ollama.com/api/usage). Não fazemos uma segunda chamada nativa antes da primeira medição, pois ela alteraria o estado a observar.

### Processo novo não significa todos os caches frios

Há pelo menos quatro estados distintos: processo novo com pesos locais; processo novo precisando baixar pesos; servidor ativo com modelo não residente; servidor e modelo já aquecidos. Cache de disco do sistema operacional e cache de compilação podem sobreviver à reinicialização. Registre em `--initial-state` quais condições existem. O script **não apaga caches, não descarrega modelos de servidores alheios e não reinicia o pod**.

Sem `--launch`, a primeira requisição é apenas a primeira que este cliente observou. Não a rotule como cold start comprovado. Com `--launch`, sabemos o início do processo, mas não o estado de todos os caches. Uma porta já ocupada bloqueia o lançamento sem matar o processo existente.

Provisionamento do pod, instalação de dependências e downloads anteriores ao processo estão fora do relógio. Download feito pelo próprio runtime depois da partida entra no intervalo, mas não ganha uma decomposição automática. Separar carregamento de pesos, transferência PCIe, compilação e captura de grafos exige logs ou profiling específicos: não podemos deduzir esses tempos isolados apenas olhando a API.

### Repetição de cold start e referência aquecida

Uma chamada do programa produz uma amostra de partida. `--repetitions` repete cenários sem reiniciar; não transforma três blocos em três cold starts. Para estudar a variabilidade inicial, repita a chamada completa com `--launch` em condições documentadas, uma por vez. O processo iniciado é encerrado ao final de cada chamada.

A pergunta inicial é texto real em português, distinta da carga sintética dos cenários. Pode ser substituída por `--first-prompt-file`. Comparamos seu primeiro acesso com uma referência usando **o mesmo texto** no final; não comparamos diretamente essa pergunta curta com o cenário sintético longo. A referência final pode aproveitar cache de prefixo, portanto a diferença não isola automaticamente apenas inicialização de kernels. Ambos os resultados ficam disponíveis para interpretação.

Aquecer com três requisições é um ponto de partida, não prova de estabilidade. Observe se ainda há tendência na sequência temporal. Se necessário, aumente `--warmup` igualmente para todas as configurações. Não use o conjunto medido inteiro como aquecimento: isso pode popular caches com os mesmos prompts.

## 5. O relógio começa e termina onde?

```text
envio ───────── primeiro conteúdo ── conteúdo ── conteúdo ── fim HTTP
      ← TTFT →                  ← geração →
      ←──────────── latência ponta a ponta ───────────────→
```

As medições são observadas pelo cliente. Incluem custos do serviço e transporte HTTP; não são cronômetros internos de kernels CUDA. Rodar cliente e servidor no mesmo pod reduz variação de rede, mas pode introduzir competição pela CPU. Mantenha a topologia igual entre testes.

| Coluna do resumo | Interpretação | Unidade |
|---|---|---|
| `ttft_ms_p50/p95` | Distribuição da espera pelo primeiro conteúdo/token observado | ms |
| `e2e_s_p50/p95` | Distribuição do tempo total de requisição | s |
| `mean_itl_ms_p50/p95` | Distribuição da média de tempo entre tokens de cada requisição | ms |
| `prompt_tokens_p50/p95` | Comprimento real de entrada registrado | tokens |
| `output_tokens_p50/p95` | Comprimento real da saída registrada | tokens |
| `successful/errored/incomplete` | Contagens de resultados por status | requisições |

**Atenção ao nome TPOT.** Na versão fixada do GuideLLM, `time_per_output_token_ms` inclui o tempo inicial. Já `inter_token_latency_ms` é calculado como `(último token − primeiro token)/(número de tokens − 1)`. Usamos essa segunda medida no resumo como `mean_itl_ms`, evitando dar o mesmo nome a definições diferentes. Veja o [código da versão 0.7.4](https://github.com/vllm-project/guidellm/blob/v0.7.4/src/guidellm/schemas/request_stats.py).

Um evento SSE pode carregar mais de um token. Por isso exigimos suporte a usage na primeira requisição medida e mantemos as contagens do servidor no instrumento. Se usage estiver ausente, salvamos os tempos iniciais parciais e marcamos falha, sem fabricar contagens. Os timestamps continuam limitados pela granularidade dos eventos e buffering: não revelam exatamente quando cada token foi calculado na GPU. Verifique essa compatibilidade em cada versão do servidor.

O p95 de `mean_itl_ms` **não é o p95 de todas as pausas individuais do streaming**. Uma resposta pode ter uma pausa grande e ainda ter média aceitável. O relatório bruto preserva detalhes do instrumento para investigações posteriores; o resumo é deliberadamente menor.

## 6. O olhar estatístico

A unidade básica é a requisição. O resumo calcula percentis empíricos com interpolação linear, usando somente sucessos, separadamente por **fase**, cenário e repetição. Filtre `phase=measure` para analisar os blocos posteriores e `phase=warmup` para estudar o aquecimento. Falhas são mostradas, não convertidas em latência zero nem descartadas sem aviso. A primeira resposta tem valor individual, não um p95 calculado de uma amostra única.

O piloto padrão faz 30 requisições por cenário em três repetições. Serve para depurar e detectar efeitos grandes; com 30 valores, p95 depende de pouquíssimas observações. A coluna `p95_exploratory` marca blocos com menos de 100 sucessos. Isso não quer dizer que 100 garantam precisão: dependência temporal e variabilidade continuam importando.

Para a análise, examine séries temporais e dispersão, além da mediana. Não calcule o “p95 geral” tirando a média de p95 de blocos. Se combinar requisições, preserve os identificadores de cenário e repetição e explicite a população que está resumindo.

Repetir requisições no mesmo pod não cria réplicas independentes de hardware. Clocks, temperatura, estado de cache e vizinhos do host podem introduzir dependência. Para inferência mais rigorosa, desenhe blocos comparáveis, alterne a ordem das configurações e use métodos que respeitem essa estrutura. Não trate centenas de tokens da mesma resposta como centenas de réplicas independentes.

## 7. O que precisa ser mantido igual?

- GPU, quantidade de GPUs e condições do host; confira a CPU e a RAM também.
- Arquitetura, revisão do modelo e representação dos pesos, quando tecnicamente possível.
- Tokenizer, chat template e mensagens de sistema efetivas.
- Lista de cenários, sementes, quantidade de requisições e limite de saída.
- Temperatura e demais parâmetros de geração relevantes. Enviamos `temperature=0` e `top_p=1`; isso não garante igualdade bit a bit entre kernels.
- Estado de carregamento e política de reutilização de cache entre requisições.
- Versão do cliente e forma de acessar o servidor.

`requests_sha256` compara mensagens e limites de saída realmente enviados, desconsiderando o alias do modelo. Hashes diferentes significam que o teste não usou a mesma carga. Hashes iguais não provam que os runtimes aplicaram o mesmo template internamente.

**GPTQ e GGUF:** “8 bits” não identifica sozinho os mesmos pesos. Se vLLM usa GPTQ e os outros usam uma quantização GGUF distinta, o tratamento experimental inclui as duas diferenças. Não atribua tudo ao runtime. Registre o artefato no relatório e delimite a conclusão.

## 8. Cache: a principal armadilha silenciosa

O modelo estar carregado na GPU é desejável neste protocolo. Já reaproveitar resultados de um prefixo anterior pode mudar radicalmente o prefill. São coisas diferentes. Também não estamos propondo desativar o KV cache normal usado para gerar os tokens de uma resposta.

Para uma comparação de requisições independentes, tentem desativar reutilização entre requisições, quando o runtime permitir, e registrem como fizeram. Se não for possível, mantenham essa limitação explícita. O cliente não possui um botão universal de limpeza de cache para todos os servidores.

As sementes de aquecimento diferem das sementes de medição, e os prompts variam. Isso reduz repetição exata, mas **não garante cache frio**: templates e prefixos comuns ainda podem ser reaproveitados. `cache_policy` é uma declaração do operador, não uma medição automática.

## 9. Memória e saúde do pod

`gpu.csv` amostra memória ocupada, GPU utilization, temperatura e potência. Serve para procurar crescimento inesperado, aquecimento e interferência. O monitor é local ao cliente e não mede banda da GDDR6X, tráfego PCIe, FLOPs ou memória por processo.

O maior uso registrado é um **máximo amostrado**, que pode perder um pico entre leituras. A alocação reservada pelo runtime também não é igual ao volume de dados lidos. Não conclua “memory-bound” apenas porque a VRAM está quase cheia. Essa conclusão exige evidência adicional, por exemplo profiling e experimentos controlados.

## 10. Como explorar os parâmetros do vLLM

Primeiro execute a configuração que já funciona, sem tunar. Escolha um sintoma: TTFT alto com entrada longa, geração lenta ou falta de memória. Depois altere **uma decisão por vez**, mantendo o protocolo idêntico, e registre o comando exato do servidor.

Neste trabalho, priorize contexto máximo, política de cache e opções de execução que façam sentido para uma sequência. Reservar mais memória não aumenta fisicamente a banda da GPU. Não faça uma busca enorme em parâmetros de concorrência: eles são assunto do próximo trabalho.

Não incluímos flags de tuning automaticamente porque variam conforme versão e podem mudar o experimento. Consulte o `vllm serve --help` da instalação e a [referência oficial](https://docs.vllm.ai/en/latest/cli/serve/). Compare o baseline com uma mudança; se o efeito se mantiver, investigue por quê.

## 11. Checklist antes de chamar um resultado de melhor

1. Todas as fases terminaram sem erro? O manifest mostra `complete`?
2. Os comprimentos reais das respostas são comparáveis?
3. GPU, modelo, tokenizer, template, carga e cache estão documentados?
4. O ganho aparece em mais de uma repetição, sem tendência térmica óbvia?
5. A melhora é em TTFT, geração ou resposta total? Não são a mesma coisa.
6. Existe diferença de quantização impedindo atribuir o efeito só ao runtime?

Esse benchmark mede desempenho, **não qualidade**. Uma pequena bateria fixa de perguntas reais pode funcionar como checagem de regressão em outro relatório, sem misturar sua nota com as métricas de latência.

## Próximo passo

Volte ao [README](../README.md) para instalar e rodar. Faça primeiro o smoke em cada servidor. Os testes automatizados locais usam um servidor simulado: validam o instrumento, não antecipam o resultado na RTX 3090.
