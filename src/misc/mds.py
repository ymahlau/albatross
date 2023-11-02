import numpy as np


def mds(
        distances: np.ndarray,  # square symmetric distance array of size nxn
) -> np.ndarray:  # mds coordinates of size nx2
    a = - 0.5 * np.square(distances)
    row_avg = np.mean(a, axis=0)
    col_avg = np.mean(a, axis=1)
    full_avg = np.mean(a)
    b = a - row_avg[np.newaxis, :] - col_avg[:, np.newaxis] + full_avg
    eigenvalues, eigenvectors = np.linalg.eig(b)
    # find two largest eigenvalues and corresponding vectors
    largest_idx = np.argmax(eigenvalues)
    largest_value = eigenvalues[largest_idx]
    largest_vector = eigenvectors[:, largest_idx]
    eigenvalues[largest_idx] = -1
    second_idx = np.argmax(eigenvalues)
    second_value = eigenvalues[second_idx]
    second_vector = eigenvectors[:, second_idx]
    # sanity check
    if largest_value < 0 or second_value < 0:
        raise Exception(f"Eigenvalue smaller zero, maybe distance matrix is not correct?")
    # normalize vectors
    largest_normalized = largest_vector * np.sqrt(largest_value)
    second_normalized = second_vector * np.sqrt(second_value)
    result = np.stack([largest_normalized, second_normalized], axis=-1)
    return result
