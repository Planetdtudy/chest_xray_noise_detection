import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
# ----------------------------
# Parameters
# ----------------------------
image_size = 224
batch_size = 32
num_epochs = 10
learning_rate = 3e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=3),  # Swin needs 3 channels
    transforms.ToTensor(),
    # Optional: Normalize using ImageNet stats for better pretrained performance
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Dataset (FIXED INDENTATION)
# ----------------------------
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []

        # class 0: clean
        clean_folder = os.path.join(root_dir, "clean")
        if os.path.exists(clean_folder):
            for f in os.listdir(clean_folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(clean_folder, f))
                    self.labels.append(0)

        # class 1: noisy_gaussian
        noise_folder = os.path.join(root_dir, "noisy_gaussian")
        if os.path.exists(noise_folder):
            for f in os.listdir(noise_folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(noise_folder, f))
                    self.labels.append(1)

    def __len__(self):
        return len(self.images)

    # Indented inside the class now!
    def __getitem__(self, idx):
        # Open as is (Grayscale)
        img = Image.open(self.images[idx]) 
        img = transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# ----------------------------
# Model Definition
# ----------------------------
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0 # Global pool features
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# ----------------------------
# Setup
# ----------------------------
dataset = ChestXRayDataset("chest_xray")

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# REMOVE THESE TWO LINES ONCE YOU ARE READY FOR FULL TRAINING
#train_ds = torch.utils.data.Subset(train_ds, range(min(20, len(train_ds))))
#val_ds   = torch.utils.data.Subset(val_ds, range(min(20, len(val_ds))))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

model = FullModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
# ----------------------------
# The Training & Validation Loop
# ----------------------------
for epoch in range(num_epochs):
    # --- 1. Reset everything for the new Epoch ---
    model.train()
    running_loss, running_correct = 0.0, 0
    running_val_loss, val_correct = 0.0, 0

    # --- 2. Training Loop (Finish ALL batches first) ---
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs) 
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_correct += (torch.argmax(outputs, dim=1) == lbls).sum().item()
    
    # --- 3. Validation Loop (Wait until Training is DONE) ---
    model.eval()
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs) 
            lossv = criterion(outputs, lbls)
            
            running_val_loss += lossv.item() * imgs.size(0)
            val_correct += (torch.argmax(outputs, dim=1) == lbls).sum().item()

    # --- 4. Calculations (At the very end of the Epoch) ---
    epoch_loss = running_loss / len(train_ds)
    epoch_val_loss = running_val_loss / len(val_ds)
    # Calculate Accuracies
    train_acc = running_correct / len(train_ds)
    val_acc = val_correct / len(val_ds)
    
    train_losses.append(epoch_loss)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  TRAIN -> Loss: {epoch_loss:.4f} | Acc: {train_acc*100:.2f}%")
    print(f"  VAL   -> Loss: {epoch_val_loss:.4f} | Acc: {val_acc*100:.2f}%")
    print("-" * 30)
#Ploting the training and validation loss curves
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
# ----------------------------

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np

model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        # Get the probability for the "Noisy" class (index 1)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(lbls.numpy())

# Calculate ROC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
ap = average_precision_score(all_labels, all_probs)

# Epoch 1/10
#   TRAIN -> Loss: 0.7123 | Acc: 35.00%
#   VAL   -> Loss: 0.6449 | Acc: 65.00%
# ------------------------------
# Epoch 2/10
#   TRAIN -> Loss: 0.6095 | Acc: 80.00%
#   VAL   -> Loss: 0.5975 | Acc: 55.00%
# ------------------------------
# Epoch 3/10
#   TRAIN -> Loss: 0.5387 | Acc: 70.00%
#   VAL   -> Loss: 0.5375 | Acc: 55.00%
# ------------------------------
# Epoch 4/10
#   TRAIN -> Loss: 0.4512 | Acc: 85.00%
#   VAL   -> Loss: 0.4600 | Acc: 85.00%
# ------------------------------
# Epoch 5/10
#   TRAIN -> Loss: 0.3848 | Acc: 95.00%
#   VAL   -> Loss: 0.3817 | Acc: 100.00%
# ------------------------------
# Epoch 6/10
#   TRAIN -> Loss: 0.3132 | Acc: 100.00%
#   VAL   -> Loss: 0.3049 | Acc: 100.00%
# ------------------------------
# Epoch 7/10
#   TRAIN -> Loss: 0.2524 | Acc: 100.00%
#   VAL   -> Loss: 0.2329 | Acc: 100.00%
# ------------------------------
# Epoch 8/10
#   TRAIN -> Loss: 0.1975 | Acc: 100.00%
#   VAL   -> Loss: 0.1697 | Acc: 100.00%
# ------------------------------
# Epoch 9/10
#   TRAIN -> Loss: 0.1451 | Acc: 100.00%
#   VAL   -> Loss: 0.1221 | Acc: 100.00%
# ------------------------------
# Epoch 10/10
#   TRAIN -> Loss: 0.1115 | Acc: 100.00%
#   VAL   -> Loss: 0.0877 | Acc: 100.00%
# ------------------------------
#20 images:Epoch 1-3 (The Guessing Phase): The accuracy is jumping around ($35\%$ to $80\%$). The model is basically "feeling around in the dark," trying to figure out which features belong to the noise and which belong to the lungs.Epoch 4-5 (The "Aha!" Moment): Notice how the loss takes a big dip and accuracy jumps to $85-100\%$. The Swin Transformer has found the pattern.Epoch 6-10 (Refinement): The accuracy stays at $100\%$, but the Loss continues to drop (from $0.31$ down to $0.08$). This means the model isn't just getting the answers right; it's getting more confident in its answers.