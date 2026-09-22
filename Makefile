SHELL := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# Caminhos podem ser sobrescritos na chamada:
# make quantize-q8 LLAMA_CPP_DIR=/mnt/llama.cpp HF_MODEL_DIR=/mnt/qwen
BENCH_VENV ?= /workspace/chatbot-runtime-bench/.venv
SYSTEM_PYTHON ?= python3
PYTHON ?= $(BENCH_VENV)/bin/python
HF ?= hf
LLAMA_CPP_DIR ?= /workspace/llama.cpp
LLAMA_CPP_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build
LLAMA_VENV ?= /workspace/llama-cpp-venv
LLAMA_PYTHON ?= $(LLAMA_VENV)/bin/python
HF_MODEL_ID ?= Qwen/Qwen2.5-7B-Instruct
HF_REVISION ?= main
HF_MODEL_DIR ?= /workspace/models/Qwen2.5-7B-Instruct-original
HF_HUB_ENABLE_HF_TRANSFER ?= 0
HF_REPO_ID ?= yanwerneck/Qwen2.5-7B-Instruct-GGUF-Q8_0
HF_UPLOAD_FILENAME ?= Qwen2.5-7B-Instruct-Q8_0.gguf
GGUF_OUTPUT_DIR ?= /workspace/models/Qwen2.5-7B-Instruct-GGUF-Q8_0
QUANTIZE_THREADS ?= $(shell nproc 2>/dev/null || echo 1)
MODEL_SIZE ?= 7B
MODEL_7B_GGUF ?= /workspace/models/Qwen2.5-7B-Instruct-Q8_0/Qwen2.5-7B-Instruct-Q8_0.gguf
MODEL_14B_GGUF ?= /workspace/models/Qwen2.5-14B-Instruct-Q8_0.gguf
MODEL_7B_TOKENIZER ?= /workspace/models/Qwen2.5-7B-Instruct-original
MODEL_14B_TOKENIZER ?= /workspace/models/Qwen2.5-14B-tokenizer
VLLM_EXTRA_ARGS ?=
LLAMA_EXTRA_ARGS ?=
OLLAMA_EXTRA_ARGS ?=
ifeq ($(MODEL_SIZE),7B)
GGUF_FILE := $(MODEL_7B_GGUF)
TOKENIZER_DIR ?= $(MODEL_7B_TOKENIZER)
HF_GGUF_REPO ?= arthuravianna/Qwen2.5-7B-Instruct-Q8_0.gguf
HF_GGUF_FILENAME ?= Qwen2.5-7B-Instruct-Q8_0.gguf
HF_TOKENIZER_MODEL ?= Qwen/Qwen2.5-7B-Instruct
VLLM_MODEL_DIR ?= $(dir $(MODEL_7B_GGUF))
VLLM_CONFIG ?= configs/vllm-7b-gguf.json
VLLM_LAUNCH ?= configs/launch-vllm-7b-gguf.example.json
LLAMA_MODEL_DIR ?= $(dir $(MODEL_7B_GGUF))
LLAMA_CONFIG ?= configs/llamacpp-7b-gguf.json
LLAMA_LAUNCH ?= configs/launch-llamacpp-7b-gguf.json
OLLAMA_MODEL_DIR ?= $(dir $(MODEL_7B_GGUF))
OLLAMA_CONFIG ?= configs/ollama-7b-gguf.json
OLLAMA_LAUNCH ?= configs/launch-ollama-7b-gguf.json
else ifeq ($(MODEL_SIZE),14B)
GGUF_FILE := $(MODEL_14B_GGUF)
TOKENIZER_DIR ?= $(MODEL_14B_TOKENIZER)
HF_GGUF_REPO ?= arthuravianna/Qwen2.5-14B-Instruct-Q8_0.gguf
HF_GGUF_FILENAME ?= Qwen2.5-14B-Instruct-Q8_0.gguf
HF_TOKENIZER_MODEL ?= Qwen/Qwen2.5-14B-Instruct
VLLM_MODEL_DIR ?= $(MODEL_14B_GGUF)
VLLM_CONFIG ?= configs/vllm-14b-gguf.json
VLLM_LAUNCH ?= configs/launch-vllm-14b-gguf.json
LLAMA_MODEL_DIR ?= $(MODEL_14B_GGUF)
LLAMA_CONFIG ?= configs/llamacpp-14b-gguf.json
LLAMA_LAUNCH ?= configs/launch-llamacpp-14b-gguf.json
OLLAMA_MODEL_DIR ?= $(MODEL_14B_GGUF)
OLLAMA_CONFIG ?= configs/ollama-14b-gguf.json
OLLAMA_LAUNCH ?= configs/launch-ollama-14b-gguf.json
else
$(error MODEL_SIZE deve ser 7B ou 14B)
endif
VLLM_BIN ?= /workspace/vllm-runtime/.venv/bin/vllm
LLAMA_SERVER_BIN ?= /workspace/llama.cpp/build/bin/llama-server
OLLAMA_BIN ?= ollama
BENCH_SCENARIOS ?= short medium long
BENCH_REQUESTS ?= 50
BENCH_REPETITIONS ?= 3
BENCH_WARMUP ?= 3
BENCH_STARTUP_TIMEOUT ?= 1800

QUANTIZE_BIN := $(LLAMA_CPP_BUILD_DIR)/bin/llama-quantize
QUANTIZE_SCRIPT := scripts/quantize_hf_to_gguf_q8_0.sh

.PHONY: help check-tools clone-llama build-llama install-llama-python \
        download-source inspect-source quantize-q8 verify-gguf upload-hf \
        install-benchmark download-model download-tokenizer prepare-benchmark prepare-ollama smoke-vllm bench-vllm bench-llama bench-ollama bench-all bench kv-sweep docs clean-info

help:
	@printf '%s\n' \
		'Fluxo llama.cpp → GGUF Q8_0' \
		'' \
		'  make clone-llama          clona o llama.cpp em LLAMA_CPP_DIR' \
		'  make build-llama          compila o binário llama-quantize' \
		'  make install-llama-python instala dependências em venv separado' \
		'  make download-source      baixa o Qwen original para o SSD' \
		'  make inspect-source       mostra formato/metadados da fonte' \
		'  make quantize-q8          gera GGUF F16 e depois GGUF Q8_0' \
		'  make verify-gguf          confirma assinatura, tamanho e SHA-256' \
		'  make upload-hf            publica somente o GGUF no Hugging Face' \
		'  make install-benchmark    cria o venv do cliente e instala requirements.txt' \
		'  make download-model       baixa o GGUF selecionado para o SSD' \
		'  make download-tokenizer   baixa somente os arquivos do tokenizer' \
		'  make prepare-benchmark    instala cliente, cria pastas e valida modelo/configs/runtimes' \
		'  make prepare-ollama      cria o alias Ollama a partir do GGUF local, sem download' \
		'  make smoke-vllm           executa smoke do benchmark com GGUF local' \
		'  make bench-vllm            bateria formal: short/medium/long, 50x3, com telemetria' \
		'  make bench-llama           mesma bateria usando llama-server' \
		'  make bench-ollama          mesma bateria usando Ollama (modelo já criado localmente)' \
		'  make bench-all             executa vLLM, llama.cpp e Ollama; continua e resume falhas' \
		'  make kv-sweep              compatibilidade: KV sweep do vLLM (já incluído em bench-vllm)' \
		'  make docs                 regenera os HTMLs dos documentos' \
		'' \
		'Variáveis úteis:' \
		'  MODEL_SIZE=7B|14B, MODEL_7B_GGUF, MODEL_14B_GGUF' \
		'  VLLM_EXTRA_ARGS, LLAMA_EXTRA_ARGS, OLLAMA_EXTRA_ARGS' \
		'  LLAMA_CPP_DIR, HF_MODEL_DIR, GGUF_OUTPUT_DIR, QUANTIZE_THREADS' \
		'  VLLM_MODEL_DIR, VLLM_CONFIG, VLLM_LAUNCH' \
		'  LLAMA_MODEL_DIR, LLAMA_CONFIG, LLAMA_LAUNCH' \
		'  OLLAMA_MODEL_DIR, OLLAMA_CONFIG, OLLAMA_LAUNCH'


check-tools:
	@command -v $(PYTHON) >/dev/null || { echo 'Python ausente'; exit 1; }
	@command -v git >/dev/null || { echo 'Git ausente'; exit 1; }
	@command -v cmake >/dev/null || { echo 'CMake ausente'; exit 1; }

clone-llama: check-tools
	@test -d "$(LLAMA_CPP_DIR)/.git" || git clone https://github.com/ggml-org/llama.cpp.git "$(LLAMA_CPP_DIR)"

build-llama: clone-llama
	cmake -S "$(LLAMA_CPP_DIR)" -B "$(LLAMA_CPP_BUILD_DIR)" -DGGML_CUDA=ON
	cmake --build "$(LLAMA_CPP_BUILD_DIR)" --target llama-quantize -j"$(QUANTIZE_THREADS)"

install-llama-python: clone-llama
	$(PYTHON) -m venv "$(LLAMA_VENV)"
	"$(LLAMA_PYTHON)" -m pip install --upgrade pip
	"$(LLAMA_PYTHON)" -m pip install -r "$(LLAMA_CPP_DIR)/requirements.txt"

download-source:
	mkdir -p "$(HF_MODEL_DIR)"
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) download "$(HF_MODEL_ID)" --revision "$(HF_REVISION)" --local-dir "$(HF_MODEL_DIR)"

inspect-source:
	@test -f "$(HF_MODEL_DIR)/config.json" || { echo "config.json ausente em $(HF_MODEL_DIR)"; exit 1; }
	$(PYTHON) -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.loads((p/"config.json").read_text()); print("model_type:",c.get("model_type")); print("architectures:",c.get("architectures")); print("quantization_config:",c.get("quantization_config")); print("safetensors:",sorted(x.name for x in p.glob("*.safetensors"))); print("gguf:",sorted(x.name for x in p.glob("*.gguf")))' "$(HF_MODEL_DIR)"

quantize-q8: build-llama install-llama-python inspect-source
	chmod +x "$(QUANTIZE_SCRIPT)"
	PYTHON="$(LLAMA_PYTHON)" LLAMA_QUANTIZE_THREADS="$(QUANTIZE_THREADS)" "$(QUANTIZE_SCRIPT)" \
		--llama-cpp-dir "$(LLAMA_CPP_DIR)" \
		--hf-model-dir "$(HF_MODEL_DIR)" \
		--output-dir "$(GGUF_OUTPUT_DIR)" \
		--threads "$(QUANTIZE_THREADS)"

verify-gguf:
	@test -f "$(GGUF_FILE)" || { echo "GGUF ausente: $(GGUF_FILE)"; exit 1; }
	ls -lh "$(GGUF_FILE)"
	printf 'assinatura: '
	head -c 4 "$(GGUF_FILE)" | od -An -tc
	$(PYTHON) -c 'import sys; from pathlib import Path; p=Path(sys.argv[1]); data=p.read_bytes(); sys.exit("assinatura GGUF inválida") if data[:4] != b"GGUF" else print("assinatura GGUF válida")' "$(GGUF_FILE)"
	sha256sum "$(GGUF_FILE)"

install-benchmark:
	@echo '[PREPARE] criando/validando venv do cliente: $(BENCH_VENV)'
	@test -x "$(PYTHON)" || "$(SYSTEM_PYTHON)" -m venv "$(BENCH_VENV)"
	"$(PYTHON)" -m pip install --upgrade pip
	"$(PYTHON)" -m pip install -r requirements.txt

download-model:
	@echo '[PREPARE] baixando GGUF $(MODEL_SIZE): $(HF_GGUF_REPO)'
	mkdir -p "$(dir $(GGUF_FILE))"
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) download "$(HF_GGUF_REPO)" "$(HF_GGUF_FILENAME)" --revision "$(HF_REVISION)" \
		--local-dir "$(dir $(GGUF_FILE))"
	@test -f "$(GGUF_FILE)" || { echo "GGUF baixado em local inesperado; esperado: $(GGUF_FILE)"; exit 1; }

download-tokenizer:
	@echo '[PREPARE] baixando tokenizer local: $(HF_TOKENIZER_MODEL)'
	mkdir -p "$(TOKENIZER_DIR)"
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) download "$(HF_TOKENIZER_MODEL)" --revision "$(HF_REVISION)" \
		--include 'config.json' 'tokenizer*' 'special_tokens_map.json' 'chat_template.jinja' \
		--local-dir "$(TOKENIZER_DIR)"

prepare-benchmark: install-benchmark download-model download-tokenizer verify-gguf
	@echo '[PREPARE] criando diretório de resultados'
	mkdir -p results
	@echo '[PREPARE] validando imports do cliente'
	$(PYTHON) -c 'import guidellm, httpx, psutil, transformers; print("guidellm", guidellm.__version__, "httpx", httpx.__version__, "psutil", psutil.__version__, "transformers", transformers.__version__)'
	@echo '[PREPARE] validando tokenizer e JSONs'
	test -f "$(TOKENIZER_DIR)/tokenizer_config.json"
	$(PYTHON) -m json.tool "$(VLLM_CONFIG)" >/dev/null
	$(PYTHON) -m json.tool "$(VLLM_LAUNCH)" >/dev/null
	$(PYTHON) -m json.tool "$(LLAMA_CONFIG)" >/dev/null
	$(PYTHON) -m json.tool "$(LLAMA_LAUNCH)" >/dev/null
	$(PYTHON) -m json.tool "$(OLLAMA_CONFIG)" >/dev/null
	$(PYTHON) -m json.tool "$(OLLAMA_LAUNCH)" >/dev/null
	@echo '[PREPARE] verificando executáveis dos runtimes'
	test -x "$(VLLM_BIN)" || { echo "vLLM ausente: $(VLLM_BIN)"; exit 1; }
	test -x "$(LLAMA_SERVER_BIN)" || { echo "llama-server ausente: $(LLAMA_SERVER_BIN)"; exit 1; }
	command -v "$(OLLAMA_BIN)" >/dev/null || { echo "ollama ausente: $(OLLAMA_BIN)"; exit 1; }
	"$(VLLM_BIN)" --help >/dev/null
	"$(LLAMA_SERVER_BIN)" --help >/dev/null
	"$(OLLAMA_BIN)" --version
	@echo '[PREPARE] pronto: execute make bench-all (Ollama exige alias criado; veja make prepare-ollama)'

prepare-ollama: verify-gguf
	@echo '[PREPARE] criando qwen7b-q8-gguf a partir do GGUF local; nenhum download será feito'
	command -v "$(OLLAMA_BIN)" >/dev/null || { echo "ollama ausente: $(OLLAMA_BIN)"; exit 1; }
	printf 'FROM %s\nPARAMETER num_ctx 16384\n' "$(GGUF_FILE)" > /tmp/Modelfile.qwen7b
	"$(OLLAMA_BIN)" create qwen7b-q8-gguf -f /tmp/Modelfile.qwen7b
	"$(OLLAMA_BIN)" show qwen7b-q8-gguf >/dev/null
	@echo '[PREPARE] alias qwen7b-q8-gguf disponível'

upload-hf: verify-gguf
	@test -n "$${HF_TOKEN:-}" || { echo 'HF_TOKEN não está definido; exporte-o sem colar o valor no repositório.' >&2; exit 1; }
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) repo create "$(HF_REPO_ID)" --repo-type model --exist-ok
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER="$(HF_HUB_ENABLE_HF_TRANSFER)" \
		$(HF) upload "$(HF_REPO_ID)" "$(GGUF_FILE)" "$(HF_UPLOAD_FILENAME)" \
			--repo-type model --commit-message "Add llama.cpp Q8_0 GGUF"

smoke-vllm: verify-gguf
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run \
		--config "$(VLLM_CONFIG)" \
		--local-model-path "$(VLLM_MODEL_DIR)" \
		--launch "$(VLLM_LAUNCH)" \
		--launch-extra-args $(VLLM_EXTRA_ARGS) \
		--smoke --scenarios short --startup-timeout 1800

bench-vllm: verify-gguf
	@echo '[BENCH vLLM] início: short/medium/long + telemetria + KV sweep embutido'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run \
		--config "$(VLLM_CONFIG)" \
		--local-model-path "$(VLLM_MODEL_DIR)" \
		--launch "$(VLLM_LAUNCH)" \
		--scenarios $(BENCH_SCENARIOS) \
		--requests "$(BENCH_REQUESTS)" \
		--repetitions "$(BENCH_REPETITIONS)" \
		--warmup "$(BENCH_WARMUP)" \
		--collect-kv-metrics \
		--startup-timeout "$(BENCH_STARTUP_TIMEOUT)"
	@echo '[BENCH vLLM] bateria base concluída; iniciando sweep KV em passos de 1024'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label vllm \
		--config "$(VLLM_CONFIG)" --launch "$(VLLM_LAUNCH)" \
		--local-model-path "$(VLLM_MODEL_DIR)" --python "$(PYTHON)" \
		--launch-extra-args $(VLLM_EXTRA_ARGS) \
		--requests "$(BENCH_REQUESTS)" --repetitions "$(BENCH_REPETITIONS)" \
		--warmup "$(BENCH_WARMUP)" --startup-timeout "$(BENCH_STARTUP_TIMEOUT)" \
		--results results/kv-sweep-vllm

bench-llama: verify-gguf
	@echo '[BENCH llama.cpp] início: short/medium/long + telemetria + KV sweep embutido'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run --config "$(LLAMA_CONFIG)" \
		--local-model-path "$(LLAMA_MODEL_DIR)" --launch "$(LLAMA_LAUNCH)" \
		--launch-extra-args $(LLAMA_EXTRA_ARGS) \
		--scenarios $(BENCH_SCENARIOS) --requests "$(BENCH_REQUESTS)" \
		--repetitions "$(BENCH_REPETITIONS)" --warmup "$(BENCH_WARMUP)" \
		--collect-kv-metrics --startup-timeout "$(BENCH_STARTUP_TIMEOUT)"
	@echo '[BENCH llama.cpp] bateria base concluída; iniciando sweep KV em passos de 1024'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label llama.cpp \
		--config "$(LLAMA_CONFIG)" --launch "$(LLAMA_LAUNCH)" \
		--local-model-path "$(LLAMA_MODEL_DIR)" --python "$(PYTHON)" \
		--launch-extra-args $(LLAMA_EXTRA_ARGS) \
		--context-flag=--ctx-size \
		--requests "$(BENCH_REQUESTS)" --repetitions "$(BENCH_REPETITIONS)" \
		--warmup "$(BENCH_WARMUP)" --startup-timeout "$(BENCH_STARTUP_TIMEOUT)" \
		--results results/kv-sweep-llama

bench-ollama: verify-gguf
	@echo '[BENCH Ollama] início: short/medium/long + telemetria + KV sweep embutido'
	@echo '[BENCH Ollama] pré-condição: o alias qwen7b-q8-gguf já deve existir em ollama list; nenhum pull será feito'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) bench.py run --config "$(OLLAMA_CONFIG)" \
		--local-model-path "$(OLLAMA_MODEL_DIR)" --launch "$(OLLAMA_LAUNCH)" \
		--launch-extra-args $(OLLAMA_EXTRA_ARGS) \
		--scenarios $(BENCH_SCENARIOS) --requests "$(BENCH_REQUESTS)" \
		--repetitions "$(BENCH_REPETITIONS)" --warmup "$(BENCH_WARMUP)" \
		--collect-kv-metrics --startup-timeout "$(BENCH_STARTUP_TIMEOUT)"
	@echo '[BENCH Ollama] bateria base concluída; iniciando sweep KV em passos de 1024'
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py --runtime-label ollama \
		--config "$(OLLAMA_CONFIG)" --launch "$(OLLAMA_LAUNCH)" \
		--local-model-path "$(OLLAMA_MODEL_DIR)" --python "$(PYTHON)" \
		--launch-extra-args $(OLLAMA_EXTRA_ARGS) \
		--context-flag none \
		--requests "$(BENCH_REQUESTS)" --repetitions "$(BENCH_REPETITIONS)" \
		--warmup "$(BENCH_WARMUP)" --startup-timeout "$(BENCH_STARTUP_TIMEOUT)" \
		--results results/kv-sweep-ollama

bench-all: verify-gguf
	@set +e
	@echo '[BENCH ALL] 1/3 vLLM'
	@$(MAKE) --no-print-directory bench-vllm; vllm_status=$$?
	@echo '[BENCH ALL] 2/3 llama.cpp'
	@$(MAKE) --no-print-directory bench-llama; llama_status=$$?
	@echo '[BENCH ALL] 3/3 Ollama'
	@$(MAKE) --no-print-directory bench-ollama; ollama_status=$$?
	@echo "[BENCH ALL] status: vLLM=$$vllm_status llama.cpp=$$llama_status Ollama=$$ollama_status"
	@test "$$vllm_status" -eq 0 -a "$$llama_status" -eq 0 -a "$$ollama_status" -eq 0

bench: bench-all

kv-sweep: verify-gguf
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
		$(PYTHON) scripts/run_kv_sweep.py \
		--config "$(VLLM_CONFIG)" --launch "$(VLLM_LAUNCH)" \
		--local-model-path "$(VLLM_MODEL_DIR)" --python "$(PYTHON)" \
		--requests "$(BENCH_REQUESTS)" --repetitions "$(BENCH_REPETITIONS)" \
		--warmup "$(BENCH_WARMUP)" --startup-timeout "$(BENCH_STARTUP_TIMEOUT)"

docs:
	$(PYTHON) scripts/build_docs.py

clean-info:
	@printf 'Não há limpeza automática: pesos, GGUFs e resultados são evidências experimentais.\n'
