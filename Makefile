SHELL := /usr/bin/env bash

# Caminhos podem ser sobrescritos na chamada:
# make quantize-q8 LLAMA_CPP_DIR=/mnt/llama.cpp HF_MODEL_DIR=/mnt/qwen
PYTHON ?= python3
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
VLLM_MODEL_DIR ?= /workspace/models/Qwen2.5-7B-Instruct-GGUF-Q8_0
VLLM_CONFIG ?= configs/vllm-7b-gguf.json
VLLM_LAUNCH ?= configs/launch-vllm-7b-gguf.example.json
BENCH_SCENARIOS ?= short medium long
BENCH_REQUESTS ?= 50
BENCH_REPETITIONS ?= 3
BENCH_WARMUP ?= 3
BENCH_STARTUP_TIMEOUT ?= 1800

QUANTIZE_BIN := $(LLAMA_CPP_BUILD_DIR)/bin/llama-quantize
QUANTIZE_SCRIPT := scripts/quantize_hf_to_gguf_q8_0.sh
GGUF_FILE := $(GGUF_OUTPUT_DIR)/Qwen2.5-7B-Instruct-original-Q8_0.gguf

.PHONY: help check-tools clone-llama build-llama install-llama-python \
        download-source inspect-source quantize-q8 verify-gguf upload-hf \
        smoke-vllm bench-vllm kv-sweep docs clean-info

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
		'  make smoke-vllm           executa smoke do benchmark com GGUF local' \
		'  make bench-vllm            bateria formal: short/medium/long, 50x3, com telemetria' \
		'  make kv-sweep              contexto/KV 1024 em 1024 até a primeira falha de memória' \
		'  make docs                 regenera os HTMLs dos documentos' \
		'' \
		'Variáveis úteis:' \
		'  LLAMA_CPP_DIR, HF_MODEL_DIR, GGUF_OUTPUT_DIR, QUANTIZE_THREADS' \
		'  VLLM_MODEL_DIR, VLLM_CONFIG, VLLM_LAUNCH'


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
		--smoke --scenarios short --startup-timeout 1800

bench-vllm: verify-gguf
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
