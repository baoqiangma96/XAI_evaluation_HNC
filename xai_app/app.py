"""
Main launcher for the XAI + Evaluation App
Modular version with clean separation of UI, XAI, and evaluation.
"""

import gradio as gr
import argparse

# Import UI builders
from xai_app.ui.tab_model import build_model_tab
from xai_app.ui.tab_explain import build_explain_tab
from xai_app.ui.tab_eval import build_eval_tab


def create_app():
    """Construct the entire multi-tab Gradio interface."""
    with gr.Blocks(
        title="XAI + Evaluation App (Medical Imaging)",
        css="""
        .gradio-container [disabled],
        .gradio-container [disabled] * {
            opacity: 0.4 !important;
            pointer-events: none !important;
        }
        """
    ) as demo:

        # ============================
        #  Model Tab  — returns model_state
        # ============================
        model_state = build_model_tab()

        # ============================
        #  Explain Tab  — needs model_state
        # ============================
        explain_tab_state = build_explain_tab(model_state)

        # ============================
        #  Evaluation Tab — needs model_state + explain outputs
        # ============================
        build_eval_tab(model_state, explain_tab_state)

    return demo


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    demo = create_app()
    demo.queue()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        quiet=True
    )
