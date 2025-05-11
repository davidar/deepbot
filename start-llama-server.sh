#!/bin/sh

MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/Mistral-Small-24B-Instruct-2501-GGUF/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"

../llama.cpp/build/bin/llama-server -m $MODEL_PATH -c 16384 --threads 8 -cb -np 8 -ngl 14
