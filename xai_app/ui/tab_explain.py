# xai_app/ui/tab_explain.py
from typing import Dict, Any, List

import gradio as gr

from xai_app.utils.config_loader import load_latec_config
from xai_app.xai.run_xai import run_xai_batch
from xai_app.xai.helpers import update_3d_view, save_xai_maps
from xai_app.ui.components import build_3d_view_controls


def build_explain_tab(model_state):
    """
    Build the 'Explain' tab.
    Returns a dict with keys:
      - "cache_state": gr.State holding the cache dict
      - "preds_state": gr.State holding the predictions JSON
    """

    cfg_preview = load_latec_config(modality="volume", config_name="explain")
    available_methods = list(cfg_preview.explain_method.keys())

    with gr.Tab("2) Explain"):
        with gr.Row():
            with gr.Column():
                data_files = gr.File(
                    label="Input images / volumes / .npz / .zip",
                    file_types=[".png", ".jpg", ".jpeg", ".nii", ".nii.gz", ".npz", ".npy", ".zip"],
                    type="filepath",
                    file_count="multiple",
                )

                is_nifti = gr.Checkbox(False, label="Treat as NIfTI volume", interactive=True)
                slice_idx = gr.Slider(0, 500, step=1, value=0, label="Slice index (for NIfTI)")

                with gr.Accordion("Choose XAI methods", open=True):
                    methods = gr.CheckboxGroup(
                        choices=available_methods,
                        value=["gradcam"] if "gradcam" in available_methods else [],
                        label="XAI methods (from LATEC configs)",
                    )
                    select_all_btn = gr.Button("Select all methods")
                    clear_all_btn = gr.Button("Clear methods")

                    def _select_all():
                        return gr.update(value=available_methods)

                    def _clear_all():
                        return gr.update(value=[])

                    select_all_btn.click(_select_all, outputs=methods)
                    clear_all_btn.click(_clear_all, outputs=methods)

                target_class = gr.Number(
                    value=-1, label="Target class index (set -1 to use Top-1 prediction)"
                )
                heat_alpha = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Heatmap alpha")

                run_btn = gr.Button("Run XAI (batch)", variant="primary")

            with gr.Column():
                out_gallery = gr.Gallery(
                    label="Attribution Viewer (2D / 3D overlays)", columns=3, height=600
                )
                preds_json = gr.Textbox(
                    label="Predictions (Top-5 per input)", lines=8, interactive=False
                )
                cache_state = gr.State({})
                gr.Markdown("### Interactive 3D viewer")
                view_selector, slice_slider, alpha_slider, channel_selector, save_button, download_npz = (
                    build_3d_view_controls()
                )

        # Wire batch XAI
        def _run_xai_wrapper(
            model_state_, data_files_, is_nifti_, slice_idx_, methods_, target_class_, heat_alpha_
        ):
            if not methods_:
                raise gr.Error("Select at least one XAI method.")
            imgs, cache, preds_json_ = run_xai_batch(
                model_state_,
                data_files_,
                is_nifti_,
                slice_idx_,
                methods_,
                int(target_class_),
                heat_alpha_,
            )
            return imgs, cache, preds_json_

        run_btn.click(
            _run_xai_wrapper,
            inputs=[model_state, data_files, is_nifti, slice_idx, methods, target_class, heat_alpha],
            outputs=[out_gallery, cache_state, preds_json],
        )

        # Wire viewer updates
        def _update_view(cache, methods_sel, view, sidx, alpha, channel):
            return update_3d_view(cache, methods_sel, view, int(sidx), float(alpha), channel)

        for comp in [view_selector, slice_slider, alpha_slider, channel_selector]:
            comp.change(
                _update_view,
                inputs=[cache_state, methods, view_selector, slice_slider, alpha_slider, channel_selector],
                outputs=out_gallery,
            )

        # Save NPZ
        def _save_maps(cache):
            path = save_xai_maps(cache)
            return gr.update(value=path, visible=True)

        save_button.click(_save_maps, inputs=[cache_state], outputs=[download_npz])

    return {
        "cache_state": cache_state,
        "preds_state": preds_json,
    }
