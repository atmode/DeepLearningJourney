---

# PCA Dimensionality Reduction Script

This Python script performs **Principal Component Analysis (PCA)** to reduce the dimensionality of a dataset from a text file. It uses `scikit-learn` and `numpy` to transform the input data into a lower-dimensional representation.

---

## Description

The script reads a matrix of vectors and three parameters (`n`, `m`, `x`) from an input file. It then applies PCA to reduce the feature space from `m` dimensions down to `min(n, m, x)` dimensions and saves the result to an output file.

- **`n`**: Number of vectors (rows).
- **`m`**: Number of features (columns).
- **`x`**: Desired number of principal components.

---

## How to Use

### 1\. Prerequisites

Make sure you have Python and the following libraries installed:

```bash
pip install numpy scikit-learn
```

### 2\. Input File Format

Create an input file (e.g., `TC00.in`).

- The **first line** must contain three space-separated integers: `n m x`.
- The **next `n` lines** must each contain `m` space-separated numbers representing a vector.

**Example `TC00.in`:**

```
4 3 2
1.0 2.0 3.0
4.5 5.1 6.2
7.8 8.0 9.3
2.1 3.5 4.9
```

### 3\. Run the Script

Execute the script from your terminal. Make sure the script and the input file are in the same directory.

```bash
python main.py
```

### 4\. Output

The script will generate an output file named `TC00-out.txt` containing the transformed data with reduced dimensions. Each row corresponds to a transformed input vector.
