"""
Main launcher for the XAI + Evaluation App
Modular version with clean separation of UI, XAI, and evaluation.
"""

import gradio as gr

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
    demo = create_app()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        quiet=True
    )
