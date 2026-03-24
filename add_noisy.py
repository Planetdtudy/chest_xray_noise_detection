import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
#newer version
# ----------------------------
# Paths
# ----------------------------
clean_dir = "chest_xray/clean"
base_noisy_dirs = {
    'gaussian': "chest_xray/noisy_gaussian",
    'speckle': "chest_xray/noisy_speckle",
    'poisson': "chest_xray/noisy_poisson",
    'salt_pepper': "chest_xray/noisy_salt_pepper"
}

# Create all noisy folders
for path in base_noisy_dirs.values():
    os.makedirs(path, exist_ok=True)
# ----------------------------
# Parameters
# ----------------------------
target_size = (512, 512)
noise_std = 0.05  # Adjust for noise strength (0.02–0.1 typical)
seed = 42  # optional: set to None for nondeterministic noise

# ----------------------------
# Helper functions
# ----------------------------
def center_crop_or_resize(image, size=(512, 512)):
    """If image >= size: center-crop. If smaller: resize with interpolation."""
    h, w = image.shape
    new_h, new_w = size
    if h >= new_h and w >= new_w:
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        return image[start_h:start_h+new_h, start_w:start_w+new_w]
    else:
        # image is smaller in at least one dimension -> resize to target
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def add_noise(image, std=0.05, scale=1.0, noise_type='gaussian'):
    """
    Add various noise types to the image.
    noise_type: 'gaussian', 'speckle', 'poisson', or 'salt_pepper'
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = image + scale * noise

    elif noise_type == 'speckle':
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = image + image * noise  # multiplicative noise

    elif noise_type == 'poisson':
        vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
        noisy = np.random.poisson(image * vals) / float(vals)

    elif noise_type == 'salt_pepper':
        noisy = image.copy()
        prob = 0.01  # adjust density
        rnd = np.random.rand(*image.shape)
        noisy[rnd < prob / 2] = 0
        noisy[rnd > 1 - prob / 2] = 1

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return np.clip(noisy, 0, 1)



# ----------------------------
# Processing
# ----------------------------
if seed is not None:
    np.random.seed(seed)
    random.seed(seed)

if not os.path.isdir(clean_dir):
    raise FileNotFoundError(f"Clean directory not found: {clean_dir}")

clean_images = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(('.jpeg'))])
if len(clean_images) == 0:
    raise RuntimeError(f"No image files found in {clean_dir}")

noise_types = ['gaussian', 'speckle', 'poisson', 'salt_pepper']
count_dict = {}

for ntype in noise_types:
    ntype_dir = base_noisy_dirs[ntype]
    count = 0
    print(f"Generating {ntype} noise images...")

    for filename in clean_images:
        img_path = os.path.join(clean_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: failed to read {img_path}, skipping.")
            continue

        img = img.astype(np.float32) / 255.0
        noisy = add_noise(img, std=noise_std, scale=1.0, noise_type=ntype)
        noisy_uint8 = (noisy * 255.0).round().astype(np.uint8)
        save_path = os.path.join(ntype_dir, filename)
        cv2.imwrite(save_path, noisy_uint8)
        count += 1

    count_dict[ntype] = count
    print(f" → {count} images saved in '{ntype_dir}'")

# ----------------------------
# Summary Plot
# ----------------------------
plt.figure(figsize=(8, 5))
plt.bar(count_dict.keys(), count_dict.values(), color='skyblue')
plt.xlabel("Noise Type")
plt.ylabel("Images Generated")
plt.title("Noisy Image Generation Summary")
plt.tight_layout()
plt.show()

print("\n All noisy sets generated successfully!")
for ntype, count in count_dict.items():
    print(f"{ntype:<12}: {count} images")
# ----------------------------
# Plot two random pairs (if enough images)
# ----------------------------
n_plot = 2
available = [f for f in clean_images if os.path.exists(os.path.join(base_noisy_dirs[1], f))]
if len(available) == 0:
    raise RuntimeError("No noisy images found to plot. Did processing fail?")

sample_count = min(n_plot, len(available))
samples = random.sample(available, sample_count)

plt.figure(figsize=(10, 6))
for i, fname in enumerate(samples):
    clean_path = os.path.join(clean_dir, fname)
    noisy_path = os.path.join(base_noisy_dirs[1], fname)

    clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

    plt.subplot(sample_count, 2, i*2 + 1)
    plt.imshow(clean, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Clean: {fname}')
    plt.axis('off')

    plt.subplot(sample_count, 2, i*2 + 2)
    plt.imshow(noisy, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Noisy: {fname}')
    plt.axis('off')

plt.tight_layout()
plt.show()
