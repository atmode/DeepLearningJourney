---
title: "Brain Tumor Classification using a Convolutional Neural Network (CNN)"
description: "This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify brain MRI images. The model is trained to distinguish between four categories: glioma tumor, meningioma tumor, pituitary tumor, and no tumor. The code is designed to run in a Google Colab environment, leveraging Google Drive for dataset storage."
---

## Brain Tumor Classification using a Convolutional Neural Network (CNN)

This project uses a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras...

## Project Workflow

1.  **Data Loading & Preprocessing**: Loads grayscale images from Google Drive, normalizes pixel values to a `[0, 1]` range, and splits the data into training, validation, and testing sets.
2.  **Data Visualization**: Displays sample images from the dataset and plots the class distribution to check for imbalances.
3.  **Model Building**: Defines a sequential CNN model with multiple convolutional, max-pooling, and dense layers. A dropout layer is included to reduce overfitting.
4.  **Training**: Trains the model using the training data and validates it on a separate validation set. Callbacks like `EarlyStopping` and `ReduceLROnPlateau` are used to optimize the training process.
5.  **Evaluation**: Assesses the model's performance on the unseen test dataset using metrics like accuracy, precision, and recall. A classification report and a confusion matrix are generated.
6.  **Prediction**: Includes a function to classify a single brain MRI image and display the prediction alongside the actual label.

---

## How to Run

### 1\. Prerequisites

- A Google account with access to Google Drive and Google Colab.
- The dataset uploaded to your Google Drive.

### 2\. Dataset Setup

Your dataset should be organized in your Google Drive with the following folder structure:

```
/MyDrive/
└── BrainTumor_TestTrain/
    └── test-train/
        ├── Training/
        │   ├── glioma_tumor/
        │   ├── meningioma_tumor/
        │   ├── no_tumor/
        │   └── pituitary_tumor/
        └── Testing/
            ├── glioma_tumor/
            ├── meningioma_tumor/
            ├── no_tumor/
            └── pituitary_tumor/
```

### 3\. Execution Steps

1.  Open the script as a notebook in **Google Colab**.
2.  Run the first code cell to install `keras-tuner` and mount your Google Drive. You'll be prompted to authorize access.
3.  Run the remaining cells in order to load the data, train the model, evaluate its performance, and test predictions on sample images.

---

## Dependencies

The main libraries used in this project are:

- `TensorFlow`
- `Keras-Tuner`
- `Scikit-learn`
- `NumPy`
- `Pandas`
- `Matplotlib`
- `Seaborn`

The script handles the installation of `keras-tuner`. The other libraries are pre-installed in the Google Colab environment.
