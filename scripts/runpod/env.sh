# Environment settings for the Runpod Pod.

export HF_HOME=/workspace/hf-cache
export PIP_CACHE_DIR=/workspace/pip-cache
export NLTK_DATA=/workspace/nltk_data

export OMNISCRIPT_MOCK=0
export OMNISCRIPT_DEVICE=cuda:0
export OMNISCRIPT_DTYPE=auto

export TESSERACT_CMD=/usr/bin/tesseract
export OMNISCRIPT_API_URL=http://127.0.0.1:8000

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$NLTK_DATA"