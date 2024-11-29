# Image Classification Using SVM

## Project Overview

This repository contains the implementation of an image classification system using a Support Vector Machine (SVM). The goal of the project is to classify images of cats and dogs stored in two folders (`train` and `test1`). The classifier is trained on a labeled dataset and used to predict the category of unseen images from the test folder.

## Index

- [Project Overview](#project-overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Steps to Run](#steps-to-run)
- [Dependencies](#dependencies)
- [Results](#results)
- [Dataset](#dataset)
- [License](#license)

## Features

### Data Preprocessing
- Image resizing to a consistent size (64x64 pixels)
- Flattening images to 1D arrays for model input
- Scaling the pixel values using StandardScaler
- Dimensionality reduction using PCA to improve classifier performance

### Classification
- Support Vector Machine (SVM) classifier with a linear kernel
- Binary classification: "cat" or "dog"

### Predictions
- Classify all images in the `test1` folder
- Store results in a CSV file with image number and predicted category
- Option to predict a specific image from the test folder

## File Structure

```
project_root/
│   image_classifications.csv
│   task3.py
│
├───test1
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└───train
    ├───cat
    │   ├── cat.0.jpg
    │   ├── cat.1.jpg
    │   └── ...
    └───dog
        ├── dog.0.jpg
        ├── dog.1.jpg
        └── ...

```

## Prerequisites

### System Requirements
- Python 3.7+
- Minimum 4GB RAM
- Sufficient disk space for dataset

### Dependencies
Install the required libraries using pip:

```bash
pip install opencv-python numpy scikit-learn pandas
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-classification-svm.git
cd image-classification-svm
```

2. Download the dataset from [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/competitions/dogs-vs-cats/data)

3. Organize your dataset:
   - Place training images in the `train/` directory
   - Place test images in the `test1/` directory

## Usage

### Training the Model
```bash
python task3.py
```
This script will:
- Preprocess images
- Train the SVM classifier
- Save the trained model
- Classify images in the test folder
- Generate `image_classifications.csv`

### Predicting a Specific Image
After running the script, you can predict a specific image by entering its number when prompted.

## Results

The `image_classifications.csv` will contain:
- `image`: Image number
- `category`: Predicted category ("cat" or "dog")

## Performance Metrics
- Accuracy: To be determined based on your specific dataset
- Recommended: Add confusion matrix and classification report

## Troubleshooting
- Ensure all dependencies are installed
- Check image file formats (jpg recommended)
- Verify image paths in the script

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License - see LICENSE file for details

## Acknowledgments
- Dataset: Kaggle Dogs vs Cats Competition
- Libraries: OpenCV, NumPy, Scikit-learn, Pandas
