"""Check cached configuration repair without network access or model weights."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.runpod.download_models import validate_json_files


class ModelDownloadTests(unittest.TestCase):
    def test_valid_configuration_does_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text('{"size": 1}')
            with patch("scripts.runpod.download_models.hf_hub_download") as download:
                validate_json_files("owner/model", directory, ["config.json", "model.safetensors"])
                download.assert_not_called()

    def test_empty_configuration_is_refreshed_at_same_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory, "revision123")
            snapshot.mkdir()
            config = snapshot / "config.json"
            config.write_text("")

            def repair(**kwargs):
                config.write_text('{"size": 1}')
                return str(config)

            with patch("scripts.runpod.download_models.hf_hub_download", side_effect=repair) as download:
                validate_json_files("owner/model", str(snapshot), ["config.json"])
                download.assert_called_once_with(
                    repo_id="owner/model", filename="config.json",
                    revision="revision123", force_download=True,
                )

    def test_invalid_refreshed_configuration_stops_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "config.json")
            config.write_text("invalid")
            with patch("scripts.runpod.download_models.hf_hub_download", return_value=str(config)):
                with self.assertRaises(json.JSONDecodeError):
                    validate_json_files("owner/model", directory, ["config.json"])
