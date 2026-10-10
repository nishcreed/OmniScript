# Deployment preparation reference — paused for guided learning

The automated deployment preparation made on 2026-10-09 was reverted at the
user's request. The user wants to implement and understand each change manually,
one step at a time, just as during the notebook migration. Do not automatically
reapply this patch or rebuild all deployment changes without a new request.

`changes.patch` preserves the complete proposal, including new files. It is a
reference, not a tested GPU release. The previous local tests passed, but no real
CUDA inference or Linux dependency installation was verified.

A full snapshot is also saved at:
`/private/tmp/omniscript-deployment-reference-20261009.tar.gz`.
The earlier `/tmp/omniscript-runpod.tar.gz` contains the reverted version and
should not be used for the guided deployment.

## Preserved working baseline

- FastAPI with mock mode and short user-facing error messages.
- Gradio at `/reader`, accepting PNG/JPEG files.
- The modular `src/inference/` package.
- Text and Base64 WAV audio returned together, with no retained output WAV.
- The original mock API and UI tests.

## Pending learning sequence

1. Add a precision setting to `src/config.py`; explain GPU VRAM and FP32/FP16/BF16.
2. Resolve device/dtype centrally and update cached loaders incrementally.
3. Set evaluation mode and verify image tensor dtype without casting token IDs.
4. Update the old IndicTransTokenizer dependency to maintained IndicTransToolkit
   plus AutoTokenizer; update preprocessing/decoding together.
5. Choose the Linux Python version and review/pin the inference dependency baseline.
6. Configure workspace caches and install system tools and NLTK resources on the Pod.
7. Download the exact DocLayNet checkpoint and other models; record model revisions.
8. Add preflight checks and stage smoke tests with timing and VRAM measurements.
9. Transfer current code, test real English text-only inference, then pictures and
   matching, then Indic translation, then the full UI.
10. Build deployment images only after validating the environment on the GPU.

## Important findings to retain

- The selected Runpod template is PyTorch 2.8.0. A Python 3.12 environment is the
  planned Linux target; the existing local mock environment uses Python 3.14.
- Default BLIP-2 loaders currently lack explicit precision selection. Cached
  models remain resident across pipeline stages; measure total memory together.
- 24 GB VRAM is a reasonable reduced-precision starting point; 48 GB provides
  development headroom. These are estimates, not measured requirements.
- The exact layout checkpoint is available from
  https://huggingface.co/DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet
  as yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt.
- AI4Bharat translation access is gated; accept its Hugging Face conditions and
  authenticate before downloading. Never commit tokens.
- IndicTransToolkit integration reference:
  https://github.com/VarunGumma/IndicTransToolkit
- Avoid silent truncation of source document text during migration.
- Decoded PCM should have sample width 2 before interpreting bytes as int16.
- GPU setup should reuse template CUDA Torch; a bare venv may hide it.
- Persistent storage is for model caches and code, not generated audio.
- Keep Gradio temporary-file cleanup and browser run history disabled.

Resume by reading this note and guiding just the next concrete change. The first
precision setting has not been reintroduced after the revert.
