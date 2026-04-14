#source /path/to/your/.venv/bin/activate
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# ----------------------------
# Parameters
# ----------------------------
image_size = 224
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
tabular_dim = 5 

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

def extract_features(img_tensor):
    """
    img_tensor: (C, H, W)
    returns: 1D feature vector
    """

    img = img_tensor.numpy()

    gray = img.mean(axis=0)

    mean = gray.mean()
    std = gray.std()
    minv = gray.min()
    maxv = gray.max()

    # simple edge/noise proxy (Laplacian)
    laplacian_var = np.var(
        gray[1:-1, 1:-1] -
        (gray[:-2, 1:-1] + gray[2:, 1:-1] +
         gray[1:-1, :-2] + gray[1:-1, 2:]) / 4
    )

    return torch.tensor([mean, std, minv, maxv, laplacian_var], dtype=torch.float)


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []

        clean_folder = os.path.join(root_dir, "clean")
        for f in os.listdir(clean_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.images.append(os.path.join(clean_folder, f))
                self.labels.append(0)

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

        tabular = extract_features(img)

        return img, tabular, label
    
 #----swin tiny without head, pretrained=False since we only want the architecture and not the pretrained weights   
cnn = timm.create_model(
'swin_tiny_patch4_window7_224',
pretrained=False
)
cnn.head = nn.Identity()
cnn.eval()
# Freeze CNN parameters
for param in cnn.parameters():
    param.requires_grad = False 

class FusionModel(nn.Module):
    def __init__(self, image_dim, tabular_dim=5, hidden_dim=128, num_classes=2):
            super().__init__()

            self.fc = nn.Sequential(
                nn.Linear(image_dim + tabular_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )

    def forward(self, img_emb, tabular):
        x = torch.cat([img_emb, tabular], dim=1)
        return self.fc(x)
            

dataset = ChestXRayDataset("chest_xray")

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

with torch.no_grad():
    sample_imgs, sample_tab, _ = next(iter(train_loader))
    emb = cnn.forward_features(sample_imgs)
    emb = emb.mean(dim=[2,3])

image_dim = emb.shape[1]

model = FusionModel(image_dim=image_dim, tabular_dim=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn.to(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()

    total_loss = 0
    correct = 0

    for imgs, tab, lbls in train_loader:
        imgs, tab, lbls = imgs.to(device), tab.to(device), lbls.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = cnn.forward_features(imgs.to(device))
            emb = emb.mean(dim=[2,3])

        outputs = model(emb, tab)

        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * imgs.size(0)
        correct += (preds == lbls).sum().item()

    acc = correct / len(train_ds)
    print(f"Epoch {epoch+1}: Loss {total_loss/len(train_ds):.4f}, Acc {acc*100:.2f}%")



model.eval()

val_correct = 0

with torch.no_grad():
    for imgs, tab, lbls in val_loader:
        imgs, tab, lbls = imgs.to(device), tab.to(device), lbls.to(device)

        emb = cnn.forward_features(imgs)
        emb = emb.mean(dim=[2,3])

        outputs = model(emb, tab)
        preds = torch.argmax(outputs, dim=1)

        val_correct += (preds == lbls).sum().item()

print("Val Acc:", val_correct / len(val_ds))
