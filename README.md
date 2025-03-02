# Facial Landmark Detection with Deep Learning

This project demonstrates training two deep learning models to detect 68 facial landmarks on images using the iBug 300W dataset. The models predict (x,y) coordinates for facial features like eyes, nose, and mouth.

## Process Overview

### 1. Dataset Preparation
- **Data Structure**: Uses XML annotations containing:
  - Image paths
  - Face bounding boxes (crop parameters)
  - 68 facial landmark coordinates
- **Parsing**: Extracts crop coordinates and landmarks from XML files
- **TF Dataset**: Creates TensorFlow Dataset with:
  - Image paths
  - Crop parameters
  - Landmark coordinates

### 2. Preprocessing Pipeline (`process_data` function)
1. **Image Loading**: 
   - Read JPEG files
   - Convert to float32 tensors (0-1 range)

2. **Cropping**:
   - Crop face region using bounding box
   - Ensure valid crop dimensions

3. **Resizing**: 
   - Standardize to 224x224 pixels

4. **Data Augmentation**:
   - Random brightness (+-30%)
   - Random contrast (70-130% range)
   - Random hue (+-10%)
   - Model 2 adds: Random rotation (+-10°)

5. **Landmark Processing**:
   - Adjust coordinates for cropping
   - Model 2: Rotate landmarks with image

### 3. Model Architecture
**Base Model**: MobileNetV2 (pretrained on ImageNet)
- Input: 224x224x3 images
- Weights: Frozen during initial training

**Head**:
1. Global Average Pooling
2. Dense (512 units, ReLU)
3. Dropout (20%)
4. Output Layer (136 units : 68x2 landmarks)

### 4. Training
- **Loss**: Mean Squared Error (MSE)
- **Metric**: Mean Absolute Error (MAE)
- **Optimizer**: Adam (LR=1e-4 for Model 2)
- **Regularization**:
  - Early Stopping (10 epoch patience)
  - Model Checkpoints
- **Training/Validation Split**: 90/10

### 5. Evaluation
- Visual comparison of predicted vs actual landmarks
- Loss/MAE curves analysis
- Test on validation set images

## Key Differences Between Models

**Model 1 (Basic):**
- **Rotation Augmentation:** No rotation
- **Landmark Rotation:** Static landmarks
- **Learning Rate:** Default Adam (1e-3)
- **Augmentation Scope:** Basic color changes
- **Input Normalization:** Normalized to [-1, 1]
- **Robustness:** Moderate performance

**Model 2 (Enhanced):**
- **Rotation Augmentation:** +-10° rotation
- **Landmark Rotation:** Rotated with image
- **Learning Rate:** Custom set to 1e-4
- **Augmentation Scope:** color and geometric transformations
- **Input Normalization:** Normalized to [-1, 1] with rotation adjustments
- **Robustness:** Higher robustness for rotated faces

## How to Use
1. **Dataset Setup**:
   - Download iBug 300W dataset

2. **Run script**:
   - face_landmarks.ipynb
   
## Requirements
- TensorFlow 2.x
- Matplotlib
- Numpy
- tensorflow_addons (for Model 2)
