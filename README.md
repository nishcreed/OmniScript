# OmniScript

See [hardware requirements and measured resource usage](docs/hardware-requirements.md)
for the tested Runpod configuration and capacity-planning limits.

## Local API and UI checks without models

Use the lightweight dependencies with the project's virtual environment:

```bash
python -m pip install -r requirements-local.txt
OMNISCRIPT_MOCK=1 python -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open http://localhost:8000/reader, upload a PNG/JPEG, and select **Read document**.
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
