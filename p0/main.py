import numpy as np
from sklearn.decomposition import PCA

def main(n: int, m: int, x: int, vectors: list[list[float]]):
    input_matrix = np.array(vectors)
    
    num_components = min(x, n, m)
    
    pca_transformer = PCA(n_components=num_components)
    transformed_matrix = pca_transformer.fit_transform(input_matrix)
    
    assert transformed_matrix.shape == (n, num_components), f"shape {transformed_matrix.shape} ≠ {(n, num_components)}"
    return transformed_matrix.tolist()

def test_pca_from_file():
    
    with open('TC00.in', 'r') as f:
        lines = f.readlines()
    
    # Parse first line to get n, m, x
    n, m, x = map(int, lines[0].strip().split())
    
    # Parse the vectors
    vectors = []
    for i in range(1, n + 1):
        vector = list(map(float, lines[i].strip().split()))
        vectors.append(vector)
    
    print(f"Input: n={n}, m={m}, x={x}")
    print(f"Vectors shape: {n}x{m}")
    
    # Run PCA
    result = main(n, m, x, vectors)
    
    with open('TC00-out.txt', 'w') as f:
        for row in result:
            row_str = ' '.join(str(val) for val in row)
            f.write(row_str + '\n')
    
    print(f"Output written to TC01-out.txt")
    print(f"Output shape: {len(result)}x{len(result[0]) if result else 0}")

if __name__ == "__main__":
    test_pca_from_file()
