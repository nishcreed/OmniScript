"""Verify the upload/JSON/audio contract without model dependencies."""

import base64
import importlib
import sys
import unittest
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image


class MockApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict("os.environ", {"OMNISCRIPT_MOCK": "1"}):
            cls.api_module = importlib.import_module("src.api")

    def setUp(self):
        self.buffer = BytesIO()
        Image.new("RGB", (20, 30), "white").save(self.buffer, format="PNG")
        self.client = TestClient(self.api_module.api)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def upload(self, data=None, filename="page.png", image_bytes=None):
        return self.client.post(
            "/process",
            files={"image": (filename, self.buffer.getvalue() if image_bytes is None else image_bytes)},
            data=data or {},
        )

    def test_valid_upload_returns_text_and_playable_wav_and_cleans_input(self):
        paths = []
        original = self.api_module.process_document

        def record_path(**kwargs):
            paths.append(Path(kwargs["image_path"]))
            return original(**kwargs)

        with patch.object(self.api_module, "process_document", side_effect=record_path):
            response = self.upload({"language": "ta", "speed": "2.0"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("Mock mode", payload["text"][0])
        self.assertEqual(payload["pages"][0]["text"], payload["text"])
        self.assertNotIn("image_path", payload["pages"][0])
        self.assertEqual(payload["audio"]["mime_type"], "audio/wav")
        wav_bytes = base64.b64decode(payload["audio"]["data"], validate=True)
        with wave.open(BytesIO(wav_bytes)) as audio:
            self.assertEqual(audio.getframerate(), 22050)
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getnframes(), 11025)
        self.assertFalse(paths[0].exists())

    def test_invalid_uploads(self):
        self.assertEqual(self.upload(image_bytes=b"").status_code, 400)
        self.assertEqual(self.upload(image_bytes=b"not an image").status_code, 400)
        self.assertEqual(self.upload(filename="page.txt").status_code, 400)
        self.assertEqual(self.client.post("/process").status_code, 422)

    def test_invalid_controls(self):
        for data in ({"language": "unknown"}, {"pitch": "13"}, {"speed": "0"}):
            with self.subTest(data=data):
                self.assertEqual(self.upload(data).status_code, 422)

    def test_upload_limit(self):
        with patch.object(self.api_module, "MAX_UPLOAD_BYTES", 10):
            self.assertEqual(self.upload().status_code, 413)

    def test_mock_mode_does_not_import_model_stack(self):
        self.assertNotIn("src.inference", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
