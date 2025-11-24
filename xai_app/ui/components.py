# xai_app/ui/components.py
import gradio as gr


def build_3d_view_controls():
    with gr.Row():
        with gr.Column():
            view_selector = gr.Radio(
                ["axial", "coronal", "sagittal"],
                value="axial",
                label="View plane"
            )
            slice_slider = gr.Slider(
                0, 128, step=1, value=48, label="Slice index"
            )
            alpha_slider = gr.Slider(
                0.0, 1.0, step=0.05, value=0.5, label="Overlay transparency"
            )
            channel_selector = gr.Dropdown(
                choices=["CT (0)", "PET (1)", "GTV (2)", "Mean (all)"],
                value="Mean (all)",
                label="Input channel"
            )
        with gr.Column():
            save_button = gr.Button("💾 Save all XAI maps (.npz)")
            download_npz = gr.File(label="Download .npz", visible=False)

    return view_selector, slice_slider, alpha_slider, channel_selector, save_button, download_npz
