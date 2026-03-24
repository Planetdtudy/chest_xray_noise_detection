# ----------------------------
# After training
#to do
#git push origin main
#git commit -m "Version 4: Updated all files: four kinds of noise"


#loads clean + each noise-type dataset,
#computes predicted probabilities,
#calculates ROC/AUC and PR curves, and
#plots all results together for visual comparison.

# ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import numpy as np

# ----------------------------
# Parameters
# ----------------------------
image_size = 224
tabular_dim = 5
hidden_dim = 128
num_classes = 2
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
number_of_images_clean=235
# ----------------------------

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.tabular = []
        self.labels = []

        # Assuming folder structure: chest-xray/clean/ and chest-xray/noisy/
        for label, subfolder in enumerate(['clean', 'noisy']):
            folder_path = os.path.join(root_dir, subfolder)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpeg')):
                    self.images.append(os.path.join(folder_path, fname))
                    # Tabular features: zeros for clean, ones for noisy
                    self.tabular.append([0]*tabular_dim if label==0 else [1]*tabular_dim)
                    self.labels.append(label)

# ----------------------------
# Evaluation: ROC & PR curves per noise type
# ----------------------------
fusion_model.eval()
cnn.eval()

noise_types = ['noisy_gaussian', 'noisy_speckle', 'noisy_poisson', 'noisy_salt_pepper']
root_dir = 'chest_xray'

results = {}

for noise in noise_types:
    print(f"\nEvaluating noise type: {noise}")
    clean_folder = os.path.join(root_dir, 'clean')
    noisy_folder = os.path.join(root_dir, noise)

    # Build dataset for this noise type
    dataset_eval = ChestXRayDataset(root_dir)
    dataset_eval.images = []
    dataset_eval.tabular = []
    dataset_eval.labels = []

    # Clean images
    for fname in os.listdir(clean_folder):
        if fname.lower().endswith('.jpeg'):
            dataset_eval.images.append(os.path.join(clean_folder, fname))
            dataset_eval.tabular.append([0]*tabular_dim)
            dataset_eval.labels.append(0)

    # Noisy images (specific type)
    for fname in os.listdir(noisy_folder):
        if fname.lower().endswith('.jpeg'):
            dataset_eval.images.append(os.path.join(noisy_folder, fname))
            dataset_eval.tabular.append([1]*tabular_dim)
            dataset_eval.labels.append(1)

    loader_eval = DataLoader(dataset_eval, batch_size=32, shuffle=False)

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, tabular, lbls in loader_eval:
            img_emb = cnn.forward_features(imgs)
            img_emb = img_emb.mean(dim=[2,3])
            outputs = fusion_model(img_emb, tabular)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class=1 (noisy)
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute ROC, AUC, PR, AP
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    ap_score = average_precision_score(all_labels, all_probs)

    results[noise] = {'fpr': fpr, 'tpr': tpr, 'prec': prec, 'rec': rec,
                      'auc': auc_score, 'ap': ap_score}

    print(f"AUC: {auc_score:.3f},  Average Precision: {ap_score:.3f}")

# ----------------------------
# Plot all ROC and PR curves
# ----------------------------
plt.figure(figsize=(12,5))

# ROC
plt.subplot(1,2,1)
for noise, vals in results.items():
    plt.plot(vals['fpr'], vals['tpr'], label=f"{noise} (AUC={vals['auc']:.2f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves by Noise Type")
plt.legend()

# Precision–Recall
plt.subplot(1,2,2)
for noise, vals in results.items():
    plt.plot(vals['rec'], vals['prec'], label=f"{noise} (AP={vals['ap']:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves by Noise Type")
plt.legend()

plt.tight_layout()
plt.show()
