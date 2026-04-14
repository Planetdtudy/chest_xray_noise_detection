# ----------------------------
# Imports
# ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
image_size = 224
batch_size = 32
num_epochs = 10
learning_rate = 1e-3

# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# ----------------------------
# Dataset
# ----------------------------
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir):

        self.images = []
        self.labels = []

        # class 0: clean
        clean_folder = os.path.join(root_dir, "clean")
        for f in os.listdir(clean_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.images.append(os.path.join(clean_folder, f))
                self.labels.append(0)

        # class 1: all noisy types
        noisy_folders = [
            "noisy_gaussian",
            "noisy_poisson",
            "noisy_salt_pepper",
            "noisy_speckle"
        ]

        for folder in noisy_folders:
            path = os.path.join(root_dir, folder)
            if not os.path.exists(path):
                continue

            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(path, f))
                    self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# ----------------------------
# Load dataset
# ----------------------------
dataset = ChestXRayDataset("chest_xray")

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# ----------------------------
# CNN (Swin backbone)
# ----------------------------
cnn = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
cnn.head = nn.Identity()
cnn.eval()  # freeze CNN

# ----------------------------
# Classifier head
# ----------------------------
class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# get embedding size
with torch.no_grad():
    sample_imgs, _ = next(iter(train_loader))
    sample_emb = cnn.forward_features(sample_imgs)
    sample_emb = sample_emb.mean(dim=[2,3])

model = Classifier(sample_emb.shape[1], num_classes=2)

# ----------------------------
# Training setup
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn.to(device)
model.to(device)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    running_correct = 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = cnn.forward_features(imgs)
            emb = emb.mean(dim=[2,3])

        outputs = model(emb)
        loss = criterion(outputs, lbls)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_correct += (preds == lbls).sum().item()

    epoch_loss = running_loss / len(train_ds)
    epoch_acc = running_correct / len(train_ds)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.2f}%")

# ----------------------------
# Validation loop
# ----------------------------
model.eval()
val_loss = 0
val_correct = 0

with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        emb = cnn.forward_features(imgs)
        emb = emb.mean(dim=[2,3])

        outputs = model(emb)
        loss = criterion(outputs, lbls)

        preds = torch.argmax(outputs, dim=1)

        val_loss += loss.item() * imgs.size(0)
        val_correct += (preds == lbls).sum().item()

val_loss /= len(val_ds)
val_acc = val_correct / len(val_ds)

print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

# ----------------------------
# Visualization
# ----------------------------
model.eval()

with torch.no_grad():
    imgs, lbls = next(iter(val_loader))
    imgs = imgs.to(device)

    emb = cnn.forward_features(imgs)
    emb = emb.mean(dim=[2,3])

    outputs = model(emb)
    preds = torch.argmax(outputs, dim=1)

    imgs = imgs.cpu()

    fig, axes = plt.subplots(1, min(8, imgs.size(0)), figsize=(16,4))
    for i in range(min(8, imgs.size(0))):
        ax = axes[i]
        ax.imshow(imgs[i].permute(1,2,0))
        ax.set_title(f"L:{lbls[i].item()} P:{preds[i].item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()