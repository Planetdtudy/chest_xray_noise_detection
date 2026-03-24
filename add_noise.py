import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
#obsolete version
# ----------------------------
# Function to add Gaussian noise
# ----------------------------
def add_noise(img, noise_level=0.1):
    """
    Adds Gaussian noise to a PyTorch image tensor.
    """
    noisy_img = img + noise_level * torch.randn_like(img)
    return torch.clamp(noisy_img, 0.0, 1.0)

# ----------------------------
# Dataset
# ----------------------------
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, tabular_dim=5, noise_level=0.1, transform=None):
        """
        root_dir: folder containing 'clean/' subfolder
        tabular_dim: number of tabular features
        noise_level: standard deviation of Gaussian noise for noisy class
        transform: torchvision transforms to apply to images
        """
        self.images = []
        self.tabular = []
        self.labels = []
        self.noise_level = noise_level
        self.tabular_dim = tabular_dim
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        # Add clean images (label 0)
        clean_folder = os.path.join(root_dir, 'clean')
        for fname in os.listdir(clean_folder):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                self.images.append(os.path.join(clean_folder, fname))
                self.tabular.append([0]*tabular_dim)
                self.labels.append(0)

        # Add noisy images (label 1) by reusing clean images
        for fname in os.listdir(clean_folder):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                self.images.append(os.path.join(clean_folder, fname))
                self.tabular.append([1]*tabular_dim)
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Apply noise if it's in the noisy class
        if label == 1:
            img = add_noise(img, noise_level=self.noise_level)

        tabular = torch.tensor(self.tabular[idx], dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return img, tabular, label

#####

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Load image and convert to tensor
img_path = "chest-xray/clean/img1.png"
img = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
img_tensor = transform(img)

# Add noise
noisy_img = add_noise(img_tensor, noise_level=0.1)

# Visualize
fig, axes = plt.subplots(1,2, figsize=(8,4))
axes[0].imshow(img_tensor.permute(1,2,0))
axes[0].set_title("Clean")
axes[0].axis('off')
axes[1].imshow(noisy_img.permute(1,2,0))
axes[1].set_title("Noisy")
axes[1].axis('off')
plt.show()
