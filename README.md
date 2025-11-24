# XAI_evaluation_HNC
This is the official implementation code of paper "Ranking XAI Methods for Head and Neck Cancer Outcome Prediction", which is sumitted to ISBI 2026. 

This study implemented a comprehensive evaluation of 13 explainable (saliency-based) AI methods using 20 metrics of faithfulness, robustness, complexity and plausibility. The user case is a head and neck outcome prediction model based on 3D DenseNet121 with the input of CT/PET/GTV. 

The framework is as below:
<img width="3297" height="1534" alt="3" src="https://github.com/user-attachments/assets/5be01f2e-d7b1-4ad5-8e9a-07ce76f05d80" />

!!!!!!!!!!!!!! Part 1. Data download and preprocessing (skip, directly go to Part 3 if you only care XAI evaluation part)
We used HECKTOR 2025 challenge dataset, please access the 
https://hecktor25.grand-challenge.org/data-download/ apply join the challenge and download data. We need the data from Task 1 and Task 2. Download, unzip and put them under the /Data folder.

Run python ./Data/preprocess_hecktor2025.py  --task1_csv "./Data/HECKTOR2025 Task 1 Training/Task 1/HECKTOR_2025_Training_Task_1.csv"  --task2_csv "./Data/HECKTOR2025 Task 2 Training/Task 2/HECKTOR_2025_Training_Task_2.csv" --task1_dir "./Data/HECKTOR2025 Task 1 Training/Task 1"  --out_csv ./Data/overlap_split.csv  --out_nifti  ./Data/preprocessed_nii --out_preview ./Data/preview_slices  --out_npz ./Data/preprocessed_npz

then you will get the processed imaging data (nifti, npz) and clincial data for outcome prediction model training. 


!!!!!!!!!!!!!! Part 2. Outcome prediction model training (skip, directly go to Part 3 if you only care XAI evaluation part)

python ./ModelTraining/main.py --model DenseNet121 --input_modality CT PET gtv --oversample True --sum_channel True --endpoint_path ./Data/clinical_data/overlap_split.csv --data_path ./Data/preprocessed_nii/   --result_path  ./result/ 

!!!!!!!!!!!!!! Part 3. XAI evluation in website, for better interaction and visisulization

python -m xai_app.app , then access webiste localhost:7860, you will see the User face for running and evluation XAI. 
