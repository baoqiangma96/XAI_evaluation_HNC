# xai_app/ui/tab_model.py
from typing import Dict, Any

import json
import gradio as gr

from xai_app.models.loader import (
    load_demo_model,
    load_torchvision_model,
    load_monai_model,
    load_transformer_model,
    load_custom_architecture,
    AVAILABLE_TORCHVISION_MODELS,
    MONAI_MODELS,
)


def _do_load_model(
    src: str,
    classmap_file,
    tv_name: str,
    monai_name: str,
    transformer_name: str,
    arch_py,
    st_file,
) -> (Dict[str, Any], str):
    if arch_py is not None:
        src = "Custom Architecture"

    if src.startswith("Demo"):
        lm = load_demo_model()
    elif src == "Torchvision Model":
        lm = load_torchvision_model(tv_name)
    elif src == "MONAI Model":
        lm = load_monai_model(monai_name)
    elif src == "Transformer Model":
        lm = load_transformer_model(transformer_name)
    elif src == "Custom Architecture":
        if arch_py is None:
            raise gr.Error("Upload a Python file defining build_model().")
        st_path = st_file.name if st_file is not None else None
        lm = load_custom_architecture(arch_py.name, st_path)
    else:
        raise gr.Error(f"Unknown model source: {src}")

    # Optional class map
    if classmap_file is not None:
        try:
            cmap_path = classmap_file.name if hasattr(classmap_file, "name") else str(classmap_file)
            with open(cmap_path, "r", encoding="utf-8") as jf:
                labels = json.load(jf)
            if isinstance(labels, list) and all(isinstance(s, str) for s in labels):
                lm.categories = labels
        except Exception as e:
            raise gr.Error(f"Failed to read class map JSON: {e}")

    model_state = {
        "kind": lm.kind,
        "device": next(lm.model.parameters()).device.type,
        "has_gradcam": lm.target_layer is not None,
        "_obj": lm,
    }

    info = {
        "kind": lm.kind,
        "device": model_state["device"],
        "has_gradcam": model_state["has_gradcam"],
        "num_classes": None if lm.categories is None else len(lm.categories),
    }
    return model_state, json.dumps(info, indent=2)


def build_model_tab():
    """
    Build the 'Model' tab and return a gr.State that stores the current model.
    """
    with gr.Tab("1) Model"):
        model_state = gr.State()

        gr.Markdown("### 🔹 Upload or select a model")

        with gr.Row():
            with gr.Column():
                arch_py_file = gr.File(label="Custom architecture (.py)", file_types=[".py"])
            with gr.Column():
                safetensor_file = gr.File(label="Optional SafeTensors weights", file_types=[".safetensors"])
            with gr.Column():
                classmap_file = gr.File(label="Optional class map JSON", file_types=[".json"])

        with gr.Accordion("Use built-in models", open=False):
            model_src = gr.Radio(
                [
                    "Demo DenseNet121 (ImageNet)",
                    "Torchvision Model",
                    "MONAI Model",
                    "Transformer Model",
                    "Custom Architecture",
                ],
                label="Model source",
                value="Demo DenseNet121 (ImageNet)",
            )

            torchvision_model_name = gr.Dropdown(
                choices=AVAILABLE_TORCHVISION_MODELS,
                value="densenet121",
                visible=False,
                label="Torchvision model",
            )
            monai_model_name = gr.Dropdown(
                choices=MONAI_MODELS,
                value="densenet121",
                visible=False,
                label="MONAI model",
            )
            transformer_model_name = gr.Textbox(
                value="google/vit-base-patch16-224",
                visible=False,
                label="Transformer model name (HuggingFace Hub)",
            )

            def toggle_model_src(src):
                return (
                    gr.update(visible=(src == "Torchvision Model")),
                    gr.update(visible=(src == "MONAI Model")),
                    gr.update(visible=(src == "Transformer Model")),
                )

            model_src.change(
                toggle_model_src,
                inputs=model_src,
                outputs=[torchvision_model_name, monai_model_name, transformer_model_name],
            )

        arch_py_file.change(lambda _: "Custom Architecture", inputs=None, outputs=model_src)

        load_btn = gr.Button("Load model", variant="primary")
        model_info = gr.Textbox(label="Model info (JSON)", lines=8)

        load_btn.click(
            _do_load_model,
            inputs=[model_src, classmap_file, torchvision_model_name,
                    monai_model_name, transformer_model_name,
                    arch_py_file, safetensor_file],
            outputs=[model_state, model_info],
        )

    return model_state
