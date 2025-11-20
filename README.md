# XAI_evaluation_HNC
This is the official implementation code of paper "Ranking XAI Methods for Head and Neck Cancer Outcome Prediction", which is sumitted to ISBI 2026. 

This study implemented a comprehensive evaluation of 13 explainable (saliency-based) AI methods using 20 metrics of faithfulness, robustness, complexity and plausibility. The user case is a head and neck outcome prediction model based on 3D DenseNet121 with the input of CT/PET/GTV. 

The framework is as below:
<img width="3297" height="1534" alt="3" src="https://github.com/user-attachments/assets/5be01f2e-d7b1-4ad5-8e9a-07ce76f05d80" />

!!!!!!!!!!!!!! Part 1. Data download and preprocessing (skip, directly go to Part 3 if you only care XAI evaluation part)
We used HECKTOR 2025 challenge dataset, please access the 
https://hecktor25.grand-challenge.org/data-download/ apply join the challenge and download data. We need the data from Task 1 and Task 2. Download, unzip and put them under the /Data folder.
Run python ./Data/preprocessing.py, then you will get the processed imaging data and clincial data for outcome prediction model training: ./Data/npz/**.npz and ./Data/HECKTOR_2025_Training_Task_2_overlap_split.csv . 



!!!!!!!!!!!!!! Part 2. Outcome prediction model training (skip, directly go to Part 3 if you only care XAI evaluation part)

!!!!!!!!!!!!!! Part 3. XAI evluation in website, for better interaction and visisulization
