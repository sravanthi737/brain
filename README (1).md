## Brain Tumor Segmentation and Statistical Analysis

This project focuses on brain tumor segmentation using MRI images. It utilizes a 3D U-Net model trained on the BraTS 2020 dataset for multi-class segmentation and performs statistical analysis on tumor volumes. The key outputs include segmentation maps, heatmaps, and volume comparisons.

## Project Overview
1)This project uses MRI scans from the BraTS 2020 dataset to:

2)Preprocess MRI images (T1, T1ce, T2, FLAIR).

3)Train a 3D U-Net model for multi-class segmentation (background, edema, enhancing tumor, non-enhancing tumor).

4)Conduct statistical analysis of the segmented tumor volumes.

5)Visualize the segmentation output and generate heatmaps for abnormality detection.

## Dataset
The dataset used is from the BraTS 2020 Challenge. A sample data path setup is as follows:
/content/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_289
## Technologies Used
*Programming Language: Python

*Frameworks & Libraries:

*PyTorch

*MONAI (Medical Open Network for AI)

*NumPy, Matplotlib, Seaborn

*SciPy, Nibabel

## Code Pipeline
1. Data Preprocessing
Loads the BraTS MRI scans in .nii format using LoadImaged.

Augments the images with random affine transformations.

Resizes the images to (128, 128, 128) for memory efficiency.

2. Model Development
3D U-Net Architecture:

Takes 4 input modalities (T1, T1ce, T2, FLAIR).

Outputs a 4-class segmentation map.

Loss Function: Dice Loss

Optimizer: Adam

3. Inference
Performs sliding window inference on MRI scans.

Generates predicted segmentation maps for background, edema, and tumor regions.

4. Statistical Analysis
Calculates voxel volumes for each segmented class (background, edema, enhancing tumor, non-enhancing tumor).

Conducts a T-test to compare predicted and ground truth tumor volumes.

5. Visualization
Displays input MRI slices, ground truth labels, and predicted segmentation.

Generates a heatmap for enhancing tumor regions.

Plots segmented region volumes as a bar chart.

## Installation
Follow these steps to set up the project environment:
Clone the Repository
git clone https://github.com/your_username/brain-segmentation  
cd brain-segmentation  
Create Virtual Environment (Recommended)
python -m venv venv  
source venv/bin/activate  # Linux/Mac  
venv\Scripts\activate    # Windows  

## Install Required Dependencies
pip install -r requirements.txt  
Usage
Run the Python script for training, inference, and analysis:
python brain_segmentation_analysis.py  

## Visualization Outputs
Segmentation Results:
Displays input MRI slices, ground truth, and predicted segmentation side by side.
### Example Visuals:

- **Segmented Region Volumes:**  
  ![Segmented Region Volumes](https://github.com/vinod40004/AI-Brain-Segmentation/blob/main/Segemented%20Region%20Volumes.jpeg)

- **Tumor Heat Map:**  
  ![Tumor Heat Map](https://github.com/vinod40004/AI-Brain-Segmentation/blob/main/Tumour%20Heat%20Map.jpeg)

- **Ground Truth:**  
  ![Ground Truth](https://github.com/vinod40004/AI-Brain-Segmentation/blob/main/ground%20truth.jpeg)

- **Input MRI:**  
  ![Input MRI Image](https://github.com/vinod40004/AI-Brain-Segmentation/blob/main/input%20mri.jpeg)

- **Predicted Segmentation:**  
  ![Predicted Segmentation](https://github.com/vinod40004/AI-Brain-Segmentation/blob/main/predicted%20Segmentation.jpeg)



Heatmap:
Visualizes the enhancing tumor regions.

Volume Analysis Plot:
Compares segmented region volumes (background, edema, tumors).

## Future Work
Implement advanced tumor classification models.

Fine-tune hyperparameters for improved segmentation.

Explore additional statistical tests for tumor analysis.

## License
This project is licensed under the MIT License.

## Contributors
Avireddy Vinod – Developer and Researcher

Feel free to contribute by submitting issues or pull requests!

### Contact  
For any questions or feedback, reach out at:  

- **Email**: [2100040004ece@gmail.com](mailto:2100040004ece@gmail.com)  
- **LinkedIn**: [Vinod Avireddy](https://www.linkedin.com/in/vinod-avireddy-552912226/)  





