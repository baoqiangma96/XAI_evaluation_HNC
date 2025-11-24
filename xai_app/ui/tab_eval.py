# xai_app/ui/tab_eval.py
from typing import Dict, Any

import gradio as gr

from xai_app.utils.config_loader import load_latec_config
from xai_app.xai.run_eval import run_eval_multi


def build_eval_tab(model_state, explain_tab_state: Dict[str, Any]): 
    """
    Build the 'Evaluate' tab.
    Needs:
      - model_state (from build_model_tab)
      - explain_tab_state = {"cache_state": ..., "preds_state": ...} from build_explain_tab
    """
    cache_state = explain_tab_state["cache_state"]
    preds_state = explain_tab_state["preds_state"]

    cfg_eval = load_latec_config(modality="volume", config_name="eval")
    available_metrics = list(cfg_eval.eval_metric.keys())

    # ---- Group metrics by aspect (only keep those that actually exist in configs) ----
    FAITHFULNESS_CANDIDATES = [
        "faithfulnesscorrelation",
        "faithfulnessestimate",
        "monotonicitycorrelation",
        "pixelflipping",
        "regionperturbation",
        "insertion",
        "deletion",
        "irof",
        "road",
        "sufficiency",
        "infidelity"
    ]

    ROBUSTNESS_CANDIDATES = [
        "locallipschitzestimate",
        "maxsensitivity",
        "continuity",
        "relativeinputstability",
        "relativeoutputstability",
        "relativerepresentationstability"
    ]

    COMPLEXITY_CANDIDATES = [
        "sparseness",
        "complexity",
        "effectivecomplexity" 
    ]

    # Intersect with what is actually available in your eval.yaml
    FAITHFULNESS_METRICS = [m for m in FAITHFULNESS_CANDIDATES if m in available_metrics]
    ROBUSTNESS_METRICS   = [m for m in ROBUSTNESS_CANDIDATES   if m in available_metrics]
    COMPLEXITY_METRICS   = [m for m in COMPLEXITY_CANDIDATES   if m in available_metrics]

    print("[INFO] Faithfulness metrics:", FAITHFULNESS_METRICS)
    print("[INFO] Robustness metrics:", ROBUSTNESS_METRICS)
    print("[INFO] Complexity metrics:", COMPLEXITY_METRICS)


    with gr.Tab("3) Evaluate"): 
        with gr.Row():
            with gr.Column():

                with gr.Accordion("🔍 Faithfulness metrics", open=True):
                    metrics_faithfulness = gr.CheckboxGroup(
                        choices=FAITHFULNESS_METRICS,
                        value=FAITHFULNESS_METRICS[:1],  # pick first one as default if exists
                        label=None,
                    )

                with gr.Accordion("🛡 Robustness metrics", open=False):
                    metrics_robustness = gr.CheckboxGroup(
                        choices=ROBUSTNESS_METRICS,
                        value=[],
                        label=None,
                    )

                with gr.Accordion("⚙️ Complexity metrics", open=False):
                    metrics_complexity = gr.CheckboxGroup(
                        choices=COMPLEXITY_METRICS,
                        value=[],
                        label=None,
                    )

                with gr.Accordion("🧬 Plausibility metrics", open=False):
                    metrics_plausibility = gr.CheckboxGroup(
                        choices=["dice", "iou", "pointing_game", "precision_at_k", "recall_at_k", "api"],
                        value=[],
                    )
                   

                manual_target = gr.Number(
                    value=-1,
                    label="Target class (set -1 to use default in evaluation)",
                )
                gt_file = gr.File(
                            label="Ground-truth GTV masks (NPZ or NIfTI)",
                            file_count="multiple",      # allow 1 or many
                            type="filepath"             # always return file paths
                        )
  
                is_nifti_mask = gr.Checkbox(False, label="Mask is NIfTI (.nii/.nii.gz)")
                slice_idx2 = gr.Slider(0, 500, step=1, value=0, label="Slice index (if mask is NIfTI)")
                eval_btn = gr.Button("Run evaluation", variant="primary")


            with gr.Column():
                eval_table = gr.Dataframe(label="Metric summary (mean / std / median)")
                save_csv_summary = gr.File(label="Download summary CSV")
                save_csv_all = gr.File(label="Download detailed CSV")

                # === NEW ===
                ranking_table = gr.Dataframe(label="📊 Ranking summary (per Aspect)")
                save_rank_csv = gr.File(label="Download ranking CSV")

        from xai_app.xai.run_rank import compute_rankings

        def _run_eval(
            model_state_,
            cache_dict_,
            faithfulness_metrics_,
            robustness_metrics_,
            complexity_metrics_,
            plausibility_metrics_,
            manual_target_,
            preds_json_text_,
            gt_file_,
            is_nifti_mask_,
            slice_idx_,
        ):
            # ---- Merge ALL LATEC metric groups ----
            selected_latec_metrics = []
            if faithfulness_metrics_:
                selected_latec_metrics.extend(faithfulness_metrics_)
            if robustness_metrics_:
                selected_latec_metrics.extend(robustness_metrics_)
            if complexity_metrics_:
                selected_latec_metrics.extend(complexity_metrics_)

            if not selected_latec_metrics and not plausibility_metrics_:
                raise gr.Error("Please select at least one evaluation metric.")
            
            # ============================
            # 1) Run evaluation
            # ============================
            summary, csv_summary, csv_all = run_eval_multi(
                model_state_,
                cache_dict_,
                selected_latec_metrics,
                plausibility_metrics_,
                int(manual_target_),
                preds_json_text_,
                gt_file_,
                is_nifti_mask_,
                int(slice_idx_),
            )

            # ============================
            # 2) Compute rankings
            # ============================
            ranking_df = compute_rankings(csv_summary, save_csv="xai_ranking_summary.csv")

            return summary, csv_summary, csv_all, ranking_df, "xai_ranking_summary.csv"


        eval_btn.click(
            _run_eval,
            inputs=[
                model_state,
                cache_state,
                metrics_faithfulness,
                metrics_robustness,
                metrics_complexity,
                metrics_plausibility,   # NEW
                manual_target,
                preds_state,
                gt_file,
                is_nifti_mask,
                slice_idx2,
            ],
            outputs=[
                eval_table,
                save_csv_summary,
                save_csv_all,
                ranking_table,      # NEW
                save_rank_csv       # NEW
            ],
)
