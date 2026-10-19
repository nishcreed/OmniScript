# OmniScript

See [hardware requirements and measured resource usage](docs/hardware-requirements.md)
for the tested Runpod configuration and capacity-planning limits.

## Local API and UI checks without models

Use the lightweight dependencies with the project's virtual environment:

```bash
python -m pip install -r requirements-local.txt
OMNISCRIPT_MOCK=1 python -m uvicorn src.api:api --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal using the same virtual environment, start the frontend:

```bash
OMNISCRIPT_API_URL=http://127.0.0.1:8000 python -m uvicorn frontend:app --host 127.0.0.1 --port 7860
```

Open http://localhost:7860/reader/, upload a PNG/JPEG, and select **Read document**.
Mock mode returns sample text and a short tone. Pitch changes its frequency and
speed changes its duration. It does not perform OCR, translation, or speech
synthesis. The API returns the same text/Base64-WAV structure as real inference,
and temporary input images are removed after processing.

Gradio uses temporary playback files; its configured cache cleanup deletes older
files periodically, and run history is disabled.

Run the API smoke checks separately:

```bash
OMNISCRIPT_MOCK=1 python -m unittest discover -s tests -v
```

For real inference, install `requirements.txt`, provide the model assets and system
dependencies, and restart the server with `OMNISCRIPT_MOCK=0`. Mock mode is read
at startup; changing the setting requires a restart.

## Frontend with a remote backend

Install `requirements-local.txt`, then run the frontend independently:

```bash
export OMNISCRIPT_API_URL="https://q3em8wv38golwg.api.runpod.ai"
# Set RUNPOD_API_KEY in this terminal for the authenticated Runpod endpoint.
python -m uvicorn frontend:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/reader/. This entry point imports only the UI and
does not load the local inference backend, so `OMNISCRIPT_MOCK` and
`OMNISCRIPT_PRELOAD` are unnecessary. The Gradio server sends requests to
`OMNISCRIPT_API_URL` with the server-side `RUNPOD_API_KEY`.
The backend entry point is `src.api:api`; the frontend entry point is `frontend:app`.
