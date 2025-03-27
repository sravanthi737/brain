import torch
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandAffined, ToTensord, Resized
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import nibabel as nib
import os

# Define the data path
base_path = "/content/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_289"
data_dict = {
    "image": [
        os.path.join(base_path, "BraTS20_Training_289_t1.nii"),
        os.path.join(base_path, "BraTS20_Training_289_t1ce.nii"),
        os.path.join(base_path, "BraTS20_Training_289_t2.nii"),
        os.path.join(base_path, "BraTS20_Training_289_flair.nii")
    ],
    "label": os.path.join(base_path, "BraTS20_Training_289_seg.nii")
}

# Data Pipeline
train_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False),  # Load .nii files
    EnsureChannelFirstd(keys=["image", "label"]),  # Add channel dimension
    ScaleIntensityd(keys=["image"]),  # Normalize intensity
    RandAffined(keys=["image", "label"], prob=0.5, rotate_range=0.1, scale_range=0.1),  # Augmentation
    Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),  # Resize for memory efficiency
    ToTensord(keys=["image", "label"])
])

# Create dataset (single sample for demo)
train_ds = Dataset([data_dict], transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

# Model Development
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=4,  # 4 input modalities (T1, T1ce, T2, FLAIR)
    out_channels=4,  # Background, edema, enhancing tumor, non-enhancing tumor
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training (fixed indentation)
def train_model(epochs=2):  # Reduced epochs for speed
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

train_model()

# Inference
model.eval()
test_input = train_ds[0]["image"].unsqueeze(0).to(device)  # Use the same sample for inference
with torch.no_grad():
    pred = sliding_window_inference(test_input, (128, 128, 128), 1, model, overlap=0.5)
    pred = torch.argmax(pred, dim=1).cpu().numpy()[0]  # Predicted segmentation

# Load ground truth for comparison
ground_truth = train_ds[0]["label"].numpy()[0]  # First channel of label

# Statistical Analysis
def analyze_segmentation(pred, ground_truth):
    # Volume calculation (voxel count per class)
    pred_volumes = [np.sum(pred == i) for i in range(4)]
    gt_volumes = [np.sum(ground_truth == i) for i in range(4)]

    # T-test between predicted and ground truth volumes
    t_stat, p_val = stats.ttest_ind(pred_volumes, gt_volumes)
    print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

    return pred_volumes

volumes = analyze_segmentation(pred, ground_truth)

# Visualization
plt.figure(figsize=(15, 5))
slice_idx = 64  # Middle slice
plt.subplot(1, 3, 1)
plt.imshow(test_input[0, 1, :, :, slice_idx].cpu().numpy(), cmap="gray")  # T1ce modality
plt.title("Input MRI (T1ce, Slice 64)")
plt.subplot(1, 3, 2)
plt.imshow(ground_truth[:, :, slice_idx])
plt.title("Ground Truth")
plt.subplot(1, 3, 3)
plt.imshow(pred[:, :, slice_idx])
plt.title("Predicted Segmentation")
plt.show()

# Heatmap of abnormality (enhancing tumor, class 2 in BraTS)
abnormality_map = (pred == 2).astype(float)  # Highlight enhancing tumor
plt.figure()
plt.imshow(abnormality_map[:, :, slice_idx], cmap="hot")
plt.title("Enhancing Tumor Heatmap")
plt.show()

# Volume Plot
sns.barplot(x=["Background", "Edema", "Enhancing Tumor", "Non-enhancing Tumor"], y=volumes)
plt.title("Segmented Region Volumes")
plt.ylabel("Voxel Count")
plt.show()
