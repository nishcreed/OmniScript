from src.api import api
from src.ui import build_ui
import gradio as gr

app = gr.mount_gradio_app(
    app=api,
    blocks=build_ui(),
    path="/reader",
    run_history=False,
    max_file_size="20mb"
)