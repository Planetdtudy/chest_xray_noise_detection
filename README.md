# Noise Detection in Chest X-Ray Images

## Overview
This project trains a SwinTransformer model leveraging ImageNet-1K pretrained weights to classify chest X-ray images as clean or noisy.  
Noisy images are generated on-the-fly with 4 noise types: **Gaussian, Speckle, Poisson, and Salt & Pepper**.  

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

Development Experience: From "Toy" to "Full"Phase 1: The Sanity Check (N=20)Before committing to a full training run, I implemented a "Toy Dataset" strategy using a subset of 20 images. The goal was to verify the FullModel architecture (Swin Transformer) and the training/validation loop synchronization.Results: The model achieved 100% accuracy within 5 epochs.Takeaway: This confirmed that the timm backbone was correctly integrated, the loss functions were calculating correctly, and the plotting logic was synchronized.Phase 2: High-Intensity Noise Classification (N=1,400)I expanded the training to the full dataset consisting of 700 clean and 700 noisy (Gaussian, $\sigma=0.25$) chest X-ray images.Model Performance: * Epoch 1: Train Acc: 98.58% | Val Acc: 100.00%Epoch 2: Train Acc: 100.00% | Val Acc: 100.00%Analysis: The Swin Transformer's hierarchical window attention is exceptionally efficient at identifying high-frequency statistical signatures. With an std of 0.25, the noise creates a distinct "digital fingerprint." The model successfully leveraged the Shifted Window mechanism to distinguish these artificial textures from natural anatomical structures almost instantaneously.

Experiment: Sensitivity AnalysisTo test the model's limits, I reduced the noise intensity from a "heavy" $\sigma=0.25$ to a "subtle" $\sigma=0.05$.Observations:Epoch 1 Training Accuracy: Dropped from 98.5% to 85.7%. This shift indicates the model is moving past simple "signature detection" and is beginning to learn more complex feature representations of image degradation.Convergence: The model still achieved 97.15% Validation Accuracy by the end of Epoch 1. This demonstrates the high sensitivity of the Shifted Window Attention mechanism; the Swin backbone is capable of detecting even low-variance statistical shifts that are nearly invisible to the human eye.💡 Technical ReflectionThe high accuracy at $\sigma=0.05$ suggests that the model is a highly capable Quality Assurance (QA) tool. However, it also suggests that for medical applications, we should aim for even lower thresholds ($\sigma=0.01$) or "Blind Noise" (where $\sigma$ is random) to ensure the model doesn't overfit to a specific noise intensity.

Training Dynamics & Sensitivity AnalysisTo move beyond a simple "sanity check," the noise intensity was lowered to $\sigma = 0.02$. This level represents subtle digital degradation that is non-trivial for standard classification models.
Epoch 1/10
  TRAIN -> Loss: 0.5560 | Acc: 70.09%
  VAL   -> Loss: 0.2229 | Acc: 91.14%
------------------------------
Epoch 2/10
  TRAIN -> Loss: 0.2147 | Acc: 92.90%
  VAL   -> Loss: 0.1074 | Acc: 95.25%
------------------------------
Epoch 3/10
  TRAIN -> Loss: 0.1268 | Acc: 94.87%
  VAL   -> Loss: 0.0821 | Acc: 95.57%