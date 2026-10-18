# Hardware requirements and measured resource usage

## Status

This document records the Runpod testing performed on 2026-10-16. The tested
configuration runs OmniScript successfully; minimum hardware requirements have
not been established. Measurements were reported from the running Pod, rather
than reproduced locally.

## Tested environment

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA RTX PRO 4500 Blackwell, 32 GB VRAM |
| System RAM | 62 GB advertised by Runpod |
| CPU | 12 vCPUs advertised by Runpod |
| Container disk | 30 GB |
| Persistent storage | Global volume mounted at `/workspace` |
| Container image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| NVIDIA driver | 580.159.04 |
| Application | FastAPI with mounted Gradio UI at `/reader/` |
| Inference configuration | CUDA device 0; `OMNISCRIPT_DTYPE=auto` |

Testing used one Uvicorn server process and sequential UI requests. Models remain
cached in that process after first use. Concurrent-request capacity and CPU
requirements were not measured. The resolved model dtype was not recorded.

## Observed memory and GPU usage

| Measurement | Observed value | Scope |
| --- | --- | --- |
| Highest sampled total GPU memory usage | 12,014 MiB (11.73 GiB) | Picture document exercising image-text matching after earlier requests |
| Highest sampled GPU utilization | 35% | Same monitored request |
| Peak server resident RAM (`VmHWM`) | 16,388,220 KiB (15.63 GiB) | High-water mark since server startup, including loading and tested requests |
| Current server resident RAM (`VmRSS`) | 4,889,080 KiB (4.66 GiB) | After the image-text matching request |

Earlier GPU observations were 8,918 MiB after an English text request and
9,524 MiB after a translated picture request. These were readings from an
already-running server with cached models, not isolated per-stage requirements.

GPU monitoring used `nvidia-smi` at 200 ms intervals. Shorter spikes may have
been missed. Its memory reading includes GPU allocations beyond the tensors
reported by PyTorch. RAM readings came from `/proc/<server-pid>/status`; they
do not represent all Pod processes or total container memory consumption.

## Observed disk usage

| Location | Reported usage | Purpose |
| --- | --- | --- |
| `/workspace/hf-cache` | 20 G | Hugging Face model cache |
| `/workspace/models` | 131 M | Layout checkpoint |
| `/workspace/nltk_data` | 64 M | Sentence-tokenization resources |
| `/opt/OmniScript/.runpod-venv` | 1.2 G | Project environment using system site-packages |
| `/workspace/omniscript-backup.tar` | 1.2 G | Backup made before editing the Pod |
| Container writable filesystem | 1.3 G used out of 30 G | Reported by `df` at measurement time |

Directory sizes were recorded with `du -sh`; displayed values are rounded. The
virtual environment and writable container usage exclude much of the preinstalled
base image and must not be treated as the Docker image size. Disk readings were
taken before the final ITM request; caches and temporary playback files can grow.
The global volume's `df` capacity display is not a measurement of billed usage.

## Tested paths and latency

Successful tests covered English text, pictures with captions, Tamil translation,
image-text matching, and audio generation. The deployed UI displayed narration
and played audio after the Gradio mount prefix was made explicit.

In a same-process test using `sample2.png`, the first English request took
179.9 seconds, including approximately 160 seconds loading checkpoint shards.
The second request took 5.7 seconds. This demonstrates the benefit of model
caching for that input; it is not a general latency guarantee. OCR and audio
processing include CPU work, and gTTS requires network access.

## Capacity planning

- Retain the tested hardware as the baseline while validating the Docker image.
- Minimum VRAM is not established by the 11.73 GiB sampled reading. Allow room
  for larger inputs, generation buffers, allocator behavior, and missed spikes.
- Minimum system RAM is not established by the 15.63 GiB process high-water mark.
  Other services, subprocesses, and loading behavior also need memory.
- Persistent storage must accommodate the model cache plus download staging,
  future revisions, and any retained backups. Current cache usage is about 20 G.
- CPU sizing and concurrent throughput require separate measurements. Additional
  server worker processes may each load their own model copies.
- Before reducing hardware, repeat tests on the candidate configuration with
  representative large images, multiple pictures, longer text, translation, and
  the intended request concurrency. Measure startup as well as warm inference.

## Global-volume compatibility

Weights and NLTK resources are stored under `/workspace`. The repository and
virtual environment are on the container disk and must be restored or rebuilt
after a container reset.

The global filesystem rejected some permission changes during downloads; those
downloads continued. Copying Transformers custom modules failed on a permission
change, so `HF_MODULES_CACHE` is set to `/tmp/omniscript-hf-modules` on the
container disk. This keeps custom Python code separate from persistent weights.
