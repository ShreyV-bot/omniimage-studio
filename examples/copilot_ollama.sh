#!/usr/bin/env sh
# Use any local OpenAI-compatible server (Ollama shown). Pull a tool-calling model first: ollama pull qwen2.5:7b
omni copilot --backend openai --base-url http://localhost:11434/v1 --model qwen2.5:7b --workdir out \
  "make a cold moody LUT and a short riser sound effect"
