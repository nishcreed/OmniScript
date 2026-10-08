"""Check upload errors and UI response handling without models or a live server."""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image


class ReaderUiTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = importlib.import_module("src.ui")
        with patch.dict("os.environ", {"OMNISCRIPT_MOCK": "1"}):
            cls.api_module = importlib.import_module("src.api")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "document.png"
        Image.new("RGB", (20, 30), "white").save(self.path)

    async def test_heic_has_short_actionable_error(self):
        path = self.path.with_suffix(".heic")
        path.write_bytes(b"unsupported HEIC")
        with self.assertRaises(self.ui.gr.Error) as error:
            await self.ui.process_image(str(path), "en", -2, 1)
        self.assertIn("Convert HEIC", error.exception.message)
        self.assertNotIn(str(path), error.exception.message)

    async def test_corrupt_png_has_short_error(self):
        self.path.write_bytes(b"not an image")
        with self.assertRaises(self.ui.gr.Error) as error:
            await self.ui.process_image(str(self.path), "en", -2, 1)
        self.assertIn("cannot be read", error.exception.message)
        self.assertNotIn(str(self.path), error.exception.message)

    async def test_missing_image(self):
        with self.assertRaises(self.ui.gr.Error) as error:
            await self.ui.process_image(None, "en", -2, 1)
        self.assertIn("Upload a PNG or JPEG", error.exception.message)

    async def test_unexpected_backend_details_are_not_displayed(self):
        secret = "Traceback: /private/server/model.py failed with huge details"
        transport = httpx.MockTransport(
            lambda request: httpx.Response(400, json={"detail": secret})
        )
        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=transport, **kwargs)

        with patch.object(self.ui.httpx, "AsyncClient", side_effect=client_factory):
            with self.assertRaises(self.ui.gr.Error) as error:
                await self.ui.process_image(str(self.path), "en", -2, 1)
        self.assertNotIn(secret, error.exception.message)
        self.assertLess(len(error.exception.message), 100)

    async def test_unexpected_exception_is_logged_and_hidden(self):
        with patch.object(self.ui, "_process_image", side_effect=RuntimeError("private path")):
            with self.assertLogs("src.ui", level="ERROR") as logs:
                with self.assertRaises(self.ui.gr.Error) as error:
                    await self.ui.process_image(str(self.path), "en", -2, 1)
        self.assertNotIn("private path", error.exception.message)
        self.assertIn("private path", "\n".join(logs.output))

    async def test_valid_image_runs_through_mock_api_and_decodes_audio(self):
        transport = httpx.ASGITransport(app=self.api_module.api)
        original_client = httpx.AsyncClient

        def client_factory(**kwargs):
            return original_client(transport=transport, **kwargs)

        with patch.object(self.ui.httpx, "AsyncClient", side_effect=client_factory):
            text, (sample_rate, samples) = await self.ui.process_image(
                str(self.path), "ta", -2, 2
            )
        self.assertIn("Mock mode", text)
        self.assertEqual(sample_rate, 22050)
        self.assertEqual(len(samples), 11025)

    async def test_upload_component_rejects_heic_before_image_decoding(self):
        from gradio.data_classes import FileData

        demo = self.ui.build_ui()
        upload = next(block for block in demo.blocks.values() if isinstance(block, self.ui.gr.File))
        path = self.path.with_suffix(".heic")
        path.write_bytes(b"unsupported HEIC")
        with self.assertRaises(self.ui.gr.Error) as error:
            upload.preprocess(FileData(path=str(path), orig_name=path.name))
        self.assertIn("Invalid file type", error.exception.message)
        self.assertNotIn(str(path), error.exception.message)
