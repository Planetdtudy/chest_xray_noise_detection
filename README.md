# Noise Detection in Chest X-Ray Images

## Overview
This project trains a fusion model to classify chest X-ray images as clean or noisy.  
Noisy images are generated on-the-fly with 4 noise types: **Gaussian, Speckle, Poisson, and Salt & Pepper**.  
The model combines a **Swin Transformer CNN** for image features with **tabular noise indicators**.  

The evaluation includes:
- ROC curves & AUC
- Precision–Recall curves & Average Precision
- Comparison across all noise types

---
### Requirements
Python 3.9+ with the following packages:

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt

## Dataset

This project uses chest X-ray images from the **NIH Chest X-ray dataset** (Kaggle).  
You can download the dataset here: [https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data).

### Folder setup

After downloading, organize your images like this:
chest_xray_original/clean/ # place all downloaded clean images here
chest_xray/noisy_gaussian/ # will be generated automatically
chest_xray/noisy_speckle/ # will be generated automatically
chest_xray/noisy_poisson/ # will be generated automatically
chest_xray/noisy_salt_pepper/ # will be generated automatically


### Generating noisy images

Use the provided script `add_noise.py` (or your custom noise scripts) to generate noisy images for each noise type.
This will create the four noisy image folders automatically.
