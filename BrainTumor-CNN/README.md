---

# CNN for Brain Tumor Classification

This project uses a Convolutional Neural Network (CNN) in TensorFlow to classify MRI scans into four distinct categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**.

The script is designed for Google Colab and automates the entire machine learning pipeline.

---

### Key Features

- **Data Handling**: Loads and preprocesses images directly from a specified Google Drive folder.
- **CNN Architecture**: Implements a sequential CNN model with multiple convolutional, max-pooling, and dense layers.
- **Efficient Training**: Uses `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` to adjust the learning rate dynamically.
- **Performance Evaluation**: Generates a detailed **classification report** and a visual **confusion matrix** to assess model accuracy, precision, and recall.
- **Prediction**: Includes a function to test the trained model on new, individual images.

---

### How to Use

1.  **Set up the Dataset**: Ensure your image dataset is located in your Google Drive at the path `/gdrive/MyDrive/BrainTumor_TestTrain/`.
2.  **Open in Colab**: Load and open the script file in a Google Colab notebook.
3.  **Run the Notebook**: Execute all the cells from top to bottom to train and evaluate the model.

---

### Core Technologies

- **TensorFlow & Keras** for building and training the neural network.
- **Scikit-learn** for model evaluation metrics.
- **NumPy & Pandas** for data manipulation.
- **Matplotlib & Seaborn** for data visualization.
