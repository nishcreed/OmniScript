# Persistent model assets.
export HF_HOME=/workspace/hf-cache
export NLTK_DATA=/workspace/nltk_data
export OMNISCRIPT_LAYOUT_MODEL=/workspace/models/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt

# Temporary package download cache.
export PIP_CACHE_DIR=/tmp/pip-cache

export OMNISCRIPT_MOCK=0
export OMNISCRIPT_DEVICE=cuda:0
export OMNISCRIPT_DTYPE=auto

export TESSERACT_CMD=/usr/bin/tesseract
export OMNISCRIPT_API_URL=http://127.0.0.1:8000

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$NLTK_DATA" /workspace/models