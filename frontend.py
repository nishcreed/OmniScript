"""Serve Gradio independently of the inference backend."""

import gradio as gr
from fastapi import FastAPI

from src.ui import build_ui


app = gr.mount_gradio_app(
    app=FastAPI(),
    blocks=build_ui(),
    path="/reader",
    root_path="/reader",
    show_error=False,
    max_file_size="20mb",
)
