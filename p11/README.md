---
# Neural Network Regression Model

A Python script that builds, trains, and evaluates a TensorFlow/Keras neural network for regression tasks. It includes data preprocessing, training with early stopping, and performance visualization.
---

## Getting Started

### 1\. Requirements

Install the necessary Python libraries:

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

### 2\. Configure Paths

Before running, update the `INPUT_FILE` and `OUTPUT_FILE` variables in the script to point to your dataset.

```python
INPUT_FILE = "/path/to/your/input_data.in"
OUTPUT_FILE = "/path/to/your/output_data.out"
```

### 3\. Run the Script

Execute the script from your terminal:

```bash
python main.py
```

---

## Configuration

Key hyperparameters like **learning rate**, **batch size**, **epochs**, and **dropout rate** can be easily modified at the top of the script.

---

## Output

The script will print a complete performance report to the console, including **R²**, **MSE**, **MAE**, and **RMSE** for both training and test sets. It will also generate plots showing:

- Training and validation loss over epochs.
- A scatter plot of predicted vs. actual values.
