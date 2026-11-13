# Deployment decisions and observations

This document records architecture decisions, deployment issues, and validation
performed while migrating the notebook implementation to deployed services.
Measurements describe the tested configurations and do not establish minimum
hardware requirements or production capacity.

The combined `main.py` entry point was removed in favor of `frontend:app`
and `src.api:api`.

## Decisions

- **Separate CPU frontend and GPU backend:** Gradio does not need the GPU models
  locally. Keeping it separate lets the GPU worker scale to zero independently
  of the interface. Separation is an architectural choice, not a requirement of
  Serverless. The earlier Pod served both components in one process.
- **Runpod Load Balancer endpoint:** the existing FastAPI `/process` route accepts
  direct HTTP requests. This deployment does not use a Runpod queue handler.
- **Scale to zero:** testing uses active workers set to `0` and maximum workers
  set to `1`. An active worker count of `1` keeps a worker running and billed
  while idle. Models remain in memory during a worker's lifetime; a new worker
  must load them again, even when weights are already on persistent storage.
- **Network Volume for model files:** requests succeeded after switching from
  Global Volume to Network Volume, including after the previous worker stopped
  and a new worker started. A single Network Volume constrains GPU availability
  to its data center, so storage location and available GPUs must be considered
  together.
- **Render for the frontend:** the frontend container was deployed from a Docker
  Hub image. This keeps CPU hosting separate from paid GPU inference. The Free
  service can sleep after inactivity; the frontend and GPU backend can therefore
  each have their own startup delay.
- **No application-level document/audio persistence:** the backend returns text
  and Base64 WAV in one response. Persistent storage is for reusable models and
  startup diagnostics. Gradio uses temporary playback files with cache cleanup.

### Global Volume startup failure

Repeated worker startups failed while reading Hugging Face JSON files such as
`added_tokens.json` and `preprocessor_config.json`. A forced re-download could
still produce a JSON parsing failure. The downloader exited before FastAPI
started, leaving requests without a healthy worker and producing HTTP 502 errors.

Downloads also reported rejected permission changes. Moving Transformers custom
Python modules to a local `HF_MODULES_CACHE` addressed a separate permission
failure, but did not resolve the recurring JSON-cache failure on Global Volume.
The exact filesystem or cache mechanism behind the invalid JSON was not
established. Successful Network Volume restart tests support the storage change;
they do not prove that Global Volumes are incompatible with every Hugging Face
workflow.

Startup diagnostics are retained under `/runpod-volume/omniscript-logs` when the
volume is available. Health checks alone establish application readiness, not
successful document inference; real upload and playback tests were also performed.

The Pod measurements below are historical observations, not minimum requirements
or a benchmark of the Serverless setup.

## Validation performed

- Real GPU model loading and document inference on the Pod, including English,
  Tamil translation, picture captioning, image-text matching, and audio generation.
- Direct Serverless requests returning narration and Base64 WAV audio.
- Network Volume cache reuse after a worker stopped and a new worker started.
- Local standalone frontend calling the real Serverless backend.
- Local frontend Docker container displaying text and playing returned audio.
- Deployed Render frontend displaying text and playing audio from Runpod.

Multi-user throughput, sustained load, and minimum hardware sizing have not been
established. A translated request through the deployed Render UI remains a
separate check from the earlier translation tests.

## Earlier manual Pod testing workflow

The Pod workflow was used before building and deploying the backend Docker
image. An interactive GPU machine let us install dependencies, inspect errors,
and exercise the real models step by step. Local mock tests could check HTTP
and UI behavior, but could not establish whether the large models loaded on
CUDA, generated useful narration/audio, or fitted in the available memory.

Testing used the Runpod PyTorch 2.8.0 template with an RTX PRO 4500 Blackwell
(32 GB VRAM), 62 GB advertised RAM, 12 vCPUs, Python 3.12.3, and
PyTorch 2.8.0+cu128. Persistent storage was mounted at `/workspace`.

The setup scripts now live in `scripts/runpod/pod/` to distinguish this manual
workflow from the current Serverless container startup.

### Pod environment configuration: env.sh

`env.sh` collected the Pod's environment settings in one place:

- Weights, the Hugging Face cache, and NLTK resources used `/workspace` so they
  could survive container replacement when backed by the attached volume.
- `OMNISCRIPT_MOCK=0`, `OMNISCRIPT_DEVICE=cuda:0`, and automatic dtype selection
  enabled real GPU inference instead of the local mock pipeline.
- Tesseract and the local API URL pointed to the tools and service on the Pod.
- Package downloads used `/tmp/pip-cache`. Transformers custom Python modules
  used `/tmp/omniscript-hf-modules` because copying them to the Global Volume
  encountered unsupported permission operations.

Sourcing this file exported those settings into the shell running the checks.
It contains configuration, not credentials.

### Pod environment preparation: setup.sh

`setup.sh` made preparation repeatable while working inside the prebuilt Pod:

1. Check Linux, the expected Python/PyTorch versions, and CUDA availability.
2. Source the Pod environment settings.
3. Install OCR, audio, image-processing, and build dependencies with apt.
4. Create `.runpod-venv` with `--system-site-packages` to reuse the template's
   CUDA-enabled PyTorch, then install `requirements.txt`.
5. Download NLTK sentence-tokenization resources and run `pip check`.
6. Execute a CUDA tensor operation to verify the prepared environment.
7. Run the shared `download_models.py` to prepare model files.

A successful setup established environment readiness; it did not prove that all
models or complete document inference worked. Those checks followed separately.

### Verification commands

From the repository root inside the GPU Pod, prepare the environment and run
checks using the sample image:

```bash
bash scripts/runpod/pod/setup.sh
source scripts/runpod/pod/env.sh
source .runpod-venv/bin/activate
python -m scripts.runpod.smoke_test load all
python -m scripts.runpod.smoke_test infer samples/Fig18.png --language en
python -m scripts.runpod.smoke_test infer samples/Fig18.png --language ta
```

The explicit `source` commands are needed in the interactive shell because
running setup with `bash` does not export its environment or virtual-environment
activation back to the parent shell. Substitute another document path to test
additional inputs.

`load all` loads each model in a separate process, releasing its memory before
testing the next one. It does not establish the combined pipeline memory usage.
The inference checks exercise the document pipeline and validate nonempty text
and finite, nonempty, non-silent audio. They report time and PyTorch GPU memory.
We also tested the API and the then-combined Gradio UI on the Pod, and inspected
GPU usage with `nvidia-smi` and process RAM through `/proc/<pid>/status`.

Observed peaks included 12,014 MiB (11.73 GiB) sampled total VRAM, 35% sampled
GPU utilization, and 16,388,220 KiB (15.63 GiB) process resident RAM high-water
mark. The model cache occupied about 20 GB. These measurements cover the tested
inputs, not minimum hardware requirements or concurrent-user capacity.
A same-process English test took 179.9 seconds on first use and 5.7 seconds on
second use, demonstrating model-loading overhead and reuse for that input.

### Relationship to the current deployment

The backend Dockerfile now installs dependencies and NLTK resources during the
image build. Runpod endpoint settings supply runtime environment variables, and
`start.sh` prepares cached models before launching Uvicorn. It does not execute
these Pod setup scripts. Serverless persistent paths use `/runpod-volume`
rather than the Pod's `/workspace` paths. The scripts are retained for manual
Pod testing and reference; `smoke_test.py` and `download_models.py` remain shared
under `scripts/runpod/`.
