#source /path/to/your/.venv/bin/activate
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
total_images = 500
ratio_clean = 0.3   # 30% clean, 70% noisy
image_size = 224
tabular_dim = 5
hidden_dim = 128
num_classes = 2
batch_size = 32
num_epochs = 10
learning_rate = 1e-3

batch_size_clean = int(total_images * ratio_clean)
batch_size_noisy = total_images - batch_size_clean

# ----------------------------
# Synthetic dataset
# ----------------------------
clean_imgs = torch.ones(batch_size_clean, 3, image_size, image_size) * 0.5
noisy_imgs = torch.randn(batch_size_noisy, 3, image_size, image_size)

img_batch = torch.cat([clean_imgs, noisy_imgs], dim=0)  # [2000, 3, 224, 224]

# Tabular features: zeros for clean, ones for noisy
tabular_clean = torch.zeros(batch_size_clean, tabular_dim)
tabular_noisy = torch.ones(batch_size_noisy, tabular_dim)
tabular_batch = torch.cat([tabular_clean, tabular_noisy], dim=0)

# Labels: 0 = clean, 1 = noisy
labels = torch.cat([torch.zeros(batch_size_clean, dtype=torch.long),
                    torch.ones(batch_size_noisy, dtype=torch.long)], dim=0)

# Shuffle dataset
perm = torch.randperm(total_images)
img_batch = img_batch[perm]
tabular_batch = tabular_batch[perm]
labels = labels[perm]

# Create DataLoader
dataset = TensorDataset(img_batch, tabular_batch, labels)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# Swin Tiny CNN
# ----------------------------
cnn = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
cnn.head = nn.Identity()  # remove classifier
cnn.eval()  # freeze CNN for now

# ----------------------------
# Fusion MLP
# ----------------------------
class FusionModel(nn.Module):
    def __init__(self, image_embed_dim, tabular_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_embed_dim + tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, img_emb, tabular):
        x = torch.cat([img_emb, tabular], dim=1)
        return self.fc(x)

# Get embedding size from CNN
with torch.no_grad():
    sample_emb = cnn.forward_features(img_batch[:2])
    sample_emb = sample_emb.mean(dim=[2,3])  # [B, C]

fusion_model = FusionModel(image_embed_dim=sample_emb.shape[1],
                           tabular_dim=tabular_dim,
                           hidden_dim=hidden_dim,
                           num_classes=num_classes)

# ----------------------------
# Training setup
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=learning_rate)

fusion_model.train()

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for imgs, tabular, lbls in loader:
        optimizer.zero_grad()
        
        # Step 1: CNN embeddings
        with torch.no_grad():  # freeze CNN
            img_emb = cnn.forward_features(imgs)
            img_emb = img_emb.mean(dim=[2,3])  # [B, C]
        
        # Step 2: Fusion + forward
        outputs = fusion_model(img_emb, tabular)
        
        # Step 3: Loss
        loss = criterion(outputs, lbls)
        
        # Step 4: Backward
        loss.backward()
        optimizer.step()
        
        # Accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == lbls).float().mean()
        
        running_loss += loss.item() * imgs.size(0)
        running_acc += acc.item() * imgs.size(0)
    
    epoch_loss = running_loss / total_images
    epoch_acc = running_acc / total_images
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.2f}%")

# ----------------------------
# Plot a few predictions
# ----------------------------
fusion_model.eval()
with torch.no_grad():
    sample_imgs, sample_tabular, sample_labels = next(iter(loader))
    img_emb = cnn.forward_features(sample_imgs)
    img_emb = img_emb.mean(dim=[2,3])
    outputs = fusion_model(img_emb, sample_tabular)
    preds = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(1, min(8, sample_imgs.shape[0]), figsize=(16,4))
    for i in range(min(8, sample_imgs.shape[0])):
        ax = axes[i] if sample_imgs.shape[0] > 1 else axes
        ax.imshow(sample_imgs[i].mean(dim=0), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"L:{sample_labels[i].item()} P:{preds[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

