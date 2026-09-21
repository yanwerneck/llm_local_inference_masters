SHELL := /usr/bin/env bash

# Caminhos podem ser sobrescritos na chamada:
# make quantize-q8 LLAMA_CPP_DIR=/mnt/llama.cpp HF_MODEL_DIR=/mnt/qwen
PYTHON ?= python3
HF ?= hf
LLAMA_CPP_DIR ?= /workspace/llama.cpp
LLAMA_CPP_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build
HF_MODEL_ID ?= Qwen/Qwen2.5-7B-Instruct
HF_REVISION ?= main
HF_MODEL_DIR ?= /workspace/models/Qwen2.5-7B-Instruct-original
GGUF_OUTPUT_DIR ?= /workspace/models/Qwen2.5-7B-Instruct-GGUF-Q8_0
QUANTIZE_THREADS ?= $(shell nproc 2>/dev/null || echo 1)
VLLM_MODEL_DIR ?= /workspace/models/Qwen2.5-7B-Instruct-GGUF-Q8_0
VLLM_CONFIG ?= configs/vllm.json
VLLM_LAUNCH ?= configs/launch-vllm.example.json

QUANTIZE_BIN := $(LLAMA_CPP_BUILD_DIR)/bin/llama-quantize
QUANTIZE_SCRIPT := scripts/quantize_hf_to_gguf_q8_0.sh
GGUF_FILE := $(GGUF_OUTPUT_DIR)/Qwen2.5-7B-Instruct-original-Q8_0.gguf

.PHONY: help check-tools clone-llama build-llama install-llama-python \
        download-source inspect-source quantize-q8 verify-gguf \
        smoke-vllm docs clean-info

help:
	@printf '%s\n' \
		'Fluxo llama.cpp → GGUF Q8_0' \
		'' \
		'  make clone-llama          clona o llama.cpp em LLAMA_CPP_DIR' \
		'  make build-llama          compila o binário llama-quantize' \
		'  make install-llama-python instala dependências do conversor HF→GGUF' \
		'  make download-source      baixa o Qwen original para o SSD' \
		'  make inspect-source       mostra formato/metadados da fonte' \
		'  make quantize-q8          gera GGUF F16 e depois GGUF Q8_0' \
		'  make verify-gguf          confirma assinatura, tamanho e SHA-256' \
		'  make smoke-vllm           executa smoke do benchmark com GGUF local' \
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
	$(PYTHON) -m pip install -r "$(LLAMA_CPP_DIR)/requirements.txt"

download-source:
	mkdir -p "$(HF_MODEL_DIR)"
	$(HF) download "$(HF_MODEL_ID)" --revision "$(HF_REVISION)" --local-dir "$(HF_MODEL_DIR)"

inspect-source:
	@test -f "$(HF_MODEL_DIR)/config.json" || { echo "config.json ausente em $(HF_MODEL_DIR)"; exit 1; }
	$(PYTHON) -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.loads((p/"config.json").read_text()); print("model_type:",c.get("model_type")); print("architectures:",c.get("architectures")); print("quantization_config:",c.get("quantization_config")); print("safetensors:",sorted(x.name for x in p.glob("*.safetensors"))); print("gguf:",sorted(x.name for x in p.glob("*.gguf")))' "$(HF_MODEL_DIR)"

quantize-q8: build-llama install-llama-python inspect-source
	chmod +x "$(QUANTIZE_SCRIPT)"
	LLAMA_QUANTIZE_THREADS="$(QUANTIZE_THREADS)" "$(QUANTIZE_SCRIPT)" \
		--llama-cpp-dir "$(LLAMA_CPP_DIR)" \
		--hf-model-dir "$(HF_MODEL_DIR)" \
		--output-dir "$(GGUF_OUTPUT_DIR)" \
		--threads "$(QUANTIZE_THREADS)"

verify-gguf:
	@test -f "$(GGUF_FILE)" || { echo "GGUF ausente: $(GGUF_FILE)"; exit 1; }
	ls -lh "$(GGUF_FILE)"
	printf 'assinatura: '
	head -c 4 "$(GGUF_FILE)" | od -An -tc
	$(PYTHON) -c 'import sys; from pathlib import Path; p=Path(sys.argv[1]); data=p.read_bytes(); raise SystemExit("assinatura GGUF inválida") if data[:4] != b"GGUF" else print("assinatura GGUF válida")' "$(GGUF_FILE)"
	sha256sum "$(GGUF_FILE)"

smoke-vllm: verify-gguf
	$(PYTHON) bench.py run \
		--config "$(VLLM_CONFIG)" \
		--local-model-path "$(VLLM_MODEL_DIR)" \
		--launch "$(VLLM_LAUNCH)" \
		--smoke --scenarios short --startup-timeout 1800

docs:
	$(PYTHON) scripts/build_docs.py

clean-info:
	@printf 'Não há limpeza automática: pesos, GGUFs e resultados são evidências experimentais.\n'
