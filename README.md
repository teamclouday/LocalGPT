# LocalGPT

My local LLM inference setup

### Setup

Install dependencies and download models
```
uv sync
uv run python .\download.py -i "Snowflake/snowflake-arctic-embed-m-v2.0" --exclude "*.onnx" "*.safetensors"
uv run python .\download.py -i "Snowflake/snowflake-arctic-embed-m-v2.0" -f "onnx/model.onnx"
uv run python .\download.py -i "unsloth/Qwen3-30B-A3B-GGUF" -f "Qwen3-30B-A3B-Q4_0.gguf"
```

Start docker
```
docker compose up -d
```