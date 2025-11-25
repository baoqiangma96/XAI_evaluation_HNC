# XAI_evaluation_HNC
This is the official implementation code of paper *** "Ranking XAI Methods for Head and Neck Cancer Outcome Prediction" ***, which is sumitted to ISBI 2026.  
This study implemented a comprehensive evaluation of 13 explainable (saliency-based) AI methods using 20 metrics of faithfulness, robustness, complexity and plausibility. The user case is a head and neck outcome prediction model based on 3D DenseNet121 with the input of CT/PET/GTV. 

The framework is as below:  
<img width="3297" height="1534" alt="3" src="https://github.com/user-attachments/assets/5be01f2e-d7b1-4ad5-8e9a-07ce76f05d80" />


## 🔧 Installation

1. Create a new Conda environment
```txt
conda create -n xai_app python=3.11
conda activate xai_app
```
2. Install the required packages
```txt
pip install -r requirements.txt
```
The requirements.txt includes the correct PyTorch CUDA 11.8 wheels. If your system uses a different CUDA version (e.g., CUDA 12.x), please install the matching PyTorch version first from the official website: 👉 https://pytorch.org/get-started/locally/ 

## 📦 Part 1 — Data Download & Preprocessing

(Skip if you only want to run XAI evaluation → go to Part 3.)

### 1. Download HECKTOR 2025 Data

Go to: https://hecktor25.grand-challenge.org/data-download/

Join the challenge and download Task 1 and Task 2 training data

Unzip both datasets into the project's /Data folder:
```txt
/Data
   ├── HECKTOR2025 Task 1 Training/Task 1/
   └── HECKTOR2025 Task 2 Training/Task 2/
```
### 2. Run preprocessing

This script prepares CT/PET data, segmentation masks, NPZ arrays, and the combined clinical CSV.
```txt
python ./Data/preprocess_hecktor2025.py \
    --task1_csv "./Data/HECKTOR2025 Task 1 Training/Task 1/HECKTOR_2025_Training_Task_1.csv" \
    --task2_csv "./Data/HECKTOR 2025 Task 2 Training/Task 2/HECKTOR_2025_Training_Task_2.csv" \
    --task1_dir "./Data/HECKTOR2025 Task 1 Training/Task 1" \
    --out_csv "./Data/overlap_split.csv" \
    --out_nifti "./Data/preprocessed_nii" \
    --out_preview "./Data/preview_slices" \
    --out_npz "./Data/preprocessed_npz"
```
### 3. Output folders
```txt
/Data/preprocessed_nii/     → Preprocessed NIfTI (CT/PET)
/Data/preprocessed_npz/     → Preprocessed NPZ arrays
/Data/preview_slices/       → 2D preview images
/Data/overlap_split.csv     → Final clinical CSV
```
Dataset is now ready for model training and XAI evaluation.  

## ⚙️ Part 2 — Outcome Prediction Model Training

(Skip this section if you only want to run XAI evaluation → go to Part 3.)

After preprocessing the HECKTOR 2025 data, you can train the 3D DenseNet121 prognostic model.

### 1. Run model training
```txt
python ./ModelTraining/main.py \
    --model DenseNet121 \
    --input_modality CT PT gtv \
    --oversample True \
    --sum_channel True \
    --endpoint_path ./Data/overlap_split.csv \
    --data_path ./Data/preprocessed_nii/ \
    --result_path ./result/
```
### 2. Output

The training will generate:
```txt
/result/
   └── epoch_50_sum.safetensors     → final trained checkpoint (epoch 50)
```

This checkpoint will be used for XAI evaluation in Part 3.

### 3. WandB Logging (Optional)

This project logs training metrics using Weights & Biases.
The training script initializes WandB as:
```txt
wandb.init(project='ISBI2025', entity='mbq1137723824')
```
If you want to use your own WandB account:

Create an account at https://wandb.ai

Replace the project and entity names in the script:
```txt
wandb.init(project='YOUR_PROJECT', entity='YOUR_USERNAME')
```

Log in:
```txt
wandb login
```
If you do NOT want to use WandB:

Disable logging by running:
```txt
$env:WANDB_MODE="disabled"
python ./ModelTraining/main.py --model DenseNet121 --input_modality CT PET gtv --oversample True --sum_channel True --endpoint_path ./Data/overlap_split.csv --data_path ./Data/preprocessed_nii/ --result_path ./result/
```
### 4. Notes
```txt
--input_modality CT PET gtv → model uses 3 modalities
--oversample True → balances event/non-event cases
--sum_channel True → merges CT/PET/GTV into combined tensor using sum 
```

## 🌐 Part 3 — Run the XAI Web App

Start the interface:
```txt
python -m xai_app.app
```
Then open:

http://localhost:7860

(Use --server_port 7870 if you want a different port.)

### 🔹 Tab 1 — Model

In this tab, you load the model trained in Part 2.

You can either:

Upload your model architecture (.py) from Part 2  
Upload your trained weights (.safetensors) from Part 2

Or simply download the pre-trained example files here:
👉 model_sum.py + epoch_50_sum.safetensors: [YOUR_DOWNLOAD_LINK_HERE](https://drive.google.com/drive/folders/1ldCwm6v4vkwp9vcfjJ0G_aIbgniCg1vs?usp=sharing) 

After uploading the files, click Load model and the model info will appear.  

<img width="1711" height="891" alt="image" src="https://github.com/user-attachments/assets/1c748326-a2ea-405b-9657-79d5e4aa7597" />

### 🔹 Tab 2 — Explain

This tab generates saliency maps for your model using selected XAI methods.

What to upload

One or more input .npz files  
These come directly from Part 1 (preprocessed data) (e.g., CHUM-001_input.npz).

What you can do

Select the XAI methods you want to run  
(only checked methods will be executed)

Adjust heatmap transparency (alpha)

Run all selected methods on all uploaded .npz files (batch mode)

⚠️ Methods currently not runnable

The following methods are displayed but not fully supported in this version:

lrp, attention, attentionrollout, attentionlrp

These will cause error if selected.

What you will get

A grid of saliency heatmaps for each runnable XAI method

2D / 3D visual overlays

Model predictions (Top-1 / Top-5)

JSON summary of prediction and target class

This tab lets you visually compare different explanation methods before running full quantitative evaluation in Tab 3.

<img width="1516" height="945" alt="image" src="https://github.com/user-attachments/assets/c13d84fa-9c1b-4882-a9b0-fb27a056bb79" />

### 🔹 Tab 3 — Evaluate

This tab runs quantitative evaluation of all selected XAI methods using four categories of metrics: faithfulness, robustness, complexity, and plausibility.

What you need to upload

One or more ground-truth GTV masks (.npz only)  
These should correspond to the same patients you used in Tab 2.

Only .npz masks are supported in this version.  
NIfTI masks (.nii/.nii.gz) are not supported for evaluation.  

What you can do

The interface has four groups of metrics:

1️⃣ Faithfulness metrics

Measure how well an explanation reflects the model’s true behavior  
(e.g., insertion, deletion, pixel-flipping, infidelity…)

2️⃣ Robustness metrics

Measure stability under perturbations  
(e.g., local Lipschitz estimate, maximum sensitivity…)

3️⃣ Complexity metrics

Measure sparsity and compactness of explanations  
(e.g., sparseness, effective complexity…)

4️⃣ Plausibility metrics

Compare XAI heatmaps with GTV tumor masks (.npz)  
(e.g., Dice, IoU, Pointing Game, Precision@k, API…)

➡ Only selected metrics will be executed. And don't select regionperturbation and continuity, because they are either slow or cause error. 

What you will get

After clicking Run evaluation:

✔ Summary CSV

Aggregated results (mean / std / median) for each method × metric.

✔ Detailed CSV

Full evaluation table across all patients.

✔ Ranking CSV

Per-metric method rankings + aggregated rankings across  
faithfulness / robustness / complexity / plausibility.

All CSV files appear on the right and can be downloaded directly.

Notes

Metrics use the LATEC benchmark implementations (adapted for 3D inputs).  
Evaluation uses the XAI maps generated from Tab 2.  
Plausibility metrics require .npz GTV masks.  
If a metric is incompatible or fails, it is skipped automatically.  

<img width="1450" height="902" alt="image" src="https://github.com/user-attachments/assets/065d19ea-b6ed-4b3e-94ed-a3079ce932af" />


