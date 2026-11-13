"""Serve Gradio independently of the inference backend."""

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.ui import build_ui


app = FastAPI()


@app.get("/", include_in_schema=False)
def redirect_to_reader() -> RedirectResponse:
    return RedirectResponse(url="/reader/")


app = gr.mount_gradio_app(
    app=app,
    blocks=build_ui(),
    path="/reader",
    root_path="/reader",
    show_error=False,
    max_file_size="20mb",
)
