#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# This script is intended for the Linux Runpod container.
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: Linux required." >&2
    exit 1
fi

# Check the template environment before installing anything.
python3 - <<'PY'
import sys
import torch

if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit("Unsupported Python version: expected 3.11 or 3.12.")

if torch.__version__.split("+")[0] != "2.8.0":
    raise SystemExit("Unsupported PyTorch version: expected 2.8.0.")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable.")

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

source scripts/runpod/env.sh

# Install external tools used by OCR, audio, and image processing.
apt-get update
apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    python3-venv

# Reuse the template's installed CUDA-enabled PyTorch.
python3 -m venv --system-site-packages .runpod-venv
source .runpod-venv/bin/activate

python -m pip install -r requirements.txt

# Install resources used for sentence splitting.
python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab

# Check dependency compatibility.
python -m pip check

# Confirm that CUDA still works after installation.
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable after dependency installation.")

tensor = torch.ones(1, device="cuda")
torch.cuda.synchronize()

print("CUDA check passed:", torch.cuda.get_device_name(0))
PY

# Download all model checkpoints using the prepared environment.
python -m scripts.runpod.download_models

echo "Environment setup complete."
