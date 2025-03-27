# brain
import os
import torch
import numpy as np
import monai
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    RandFlipd, RandAffined, EnsureTyped, ToTensord
)
from monai.data import CacheDataset, DataLoader, Dataset
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import compute_dice
from monai.utils import set_determinism

# Set deterministic behavior
set_determinism(seed=42)

# Paths to images & masks
DATASET_PATH = r"C:\Users\user\Desktop\PROJECTS\EMTENSOR-AI PROJECT\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\ct_scans"
LABEL_PATH = r"C:\Users\user\Desktop\PROJECTS\EMTENSOR-AI PROJECT\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1\masks"

def get_file_list():
    """Load image-mask file pairs into a dictionary for MONAI."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

    files = []
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            mask_filename = filename.replace(".nii", "_mask.nii").replace(".nii.gz", "_mask.nii.gz")
            mask_path = os.path.join(LABEL_PATH, mask_filename)
            if os.path.exists(mask_path):
                files.append({
                    "image": os.path.join(DATASET_PATH, filename),
                    "label": mask_path
                })

    if not files:
        raise ValueError("No valid image-mask pairs found in the dataset directory.")

    return files

def preprocess_data():
    """Apply transformations to both images and segmentation masks."""
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), mode="bilinear"),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])
    
    dataset = CacheDataset(data=get_file_list(), transform=train_transforms)
    return DataLoader(dataset, batch_size=2, shuffle=True)

def get_model(device):
    """Define 3D UNet for multi-class brain segmentation."""
    model = UNet(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=5,  # Five classes (gray matter, white matter, ventricles, hippocampus, tumors)
        channels=(16, 32, 64, 128), 
        strides=(2, 2, 2),
        num_res_units=2
    ).to(device)
    return model

def train_model():
    """Train the UNet model."""
    dataloader = preprocess_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)  # Multi-class Dice + CrossEntropy Loss

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model

def inference(model, image_path):
    """Perform inference on a new brain scan."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image"])
    ])
    
    data = [{"image": image_path}]
    dataset = Dataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            output = model(image)
            return output

def visualize_segmentation(image, mask):
    """Overlay predicted segmentation masks onto the original image."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[0, :, :, image.shape[-1]//2], cmap="gray")  # Show central slice
    plt.title("CT Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask[0, :, :, mask.shape[-1]//2], cmap="jet", alpha=0.5)
    plt.title("Segmentation Overlay")
    plt.show()

def evaluate_model(model, test_dataloader):
    """Compute mean Dice score for validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            image, label = batch["image"].to(device), batch["label"].to(device)
            pred = model(image)
            dice = compute_dice(pred, label, include_background=False).cpu().numpy()
            dice_scores.append(dice)
    
    avg_dice = np.mean(dice_scores)
    print(f"Mean Dice Score: {avg_dice}")

def compare_volumes(group1, group2):
    """Perform hypothesis testing on segmented brain volumes."""
    t_stat, p_value = ttest_ind(group1, group2)
    print(f"T-test results: t={t_stat}, p={p_value}")
    if p_value < 0.05:
        print("Significant difference detected!")

if __name__ == "__main__":
    trained_model = train_model()
    print("Model training complete!")
    
    # Example inference
    test_image_path = "path/to/sample/image.nii"
    pred_mask = inference(trained_model, test_image_path)
    
    # Visualization
    visualize_segmentation(test_image_path, pred_mask)
    
    # Model Evaluation
    test_loader = preprocess_data()  # Load test dataset
    evaluate_model(trained_model, test_loader)
