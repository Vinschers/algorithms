from vicentin.utils import min, mean, SVD, sum, shape, matmul, transpose


def PCA(X, k=-1):
    """
    Performs Principal Component Analysis (PCA) on the dataset X using Singular Value Decomposition (SVD).

    PCA is a popular dimensionality reduction technique that transforms the original data into a new coordinate system,
    where the first axis captures the greatest variance, the second axis the second greatest, and so on. This function
    centers the data, computes its SVD, and selects the top k principal components based on the eigenvalues of the
    covariance matrix.

    Time Complexity:
        - O(N * d^2), where N is the number of samples and d is the number of features (assuming efficient SVD computation).

    Space Complexity:
        - O(d^2), primarily for storing the covariance matrix approximation and singular vectors.

    Args:
        X (ndarray | Tensor): A 2D array or Tensor of shape (N, d), where N is the number of samples and d is the number of features.
        k (int, optional): The number of principal components to retain. If k is negative, the function uses
                           min(N, d) components. Otherwise, k is bounded by min(N, d). Defaults to -1.

    Returns:
        tuple:
            - Y (ndarray | Tensor): A 2D array of shape (N, k) containing the projection of X onto the top k principal components.
            - components (ndarray | Tensor): A 2D array of shape (k, d) where each row is a principal component (eigenvector).
            - eigenvalues (ndarray | Tensor): A 1D array of length k containing the eigenvalues of the covariance matrix corresponding
                                              to the selected components.
            - var_explained (ndarray | Tensor): A 1D array of length k representing the proportion of total variance explained by each
                                                principal component.

    Raises:
        AssertionError: If X is not a 2D array.
    """
    assert len(shape(X)) == 2, "X must be a 2D array."

    N = shape(X)[0]
    d = shape(X)[1]

    if k < 0:
        k = min(N, d)
    else:
        k = min(k, min(N, d))

    # Center the data
    Xc = X - mean(X, axis=0)

    # Perform Singular Value Decomposition
    _, S, Vt = SVD(Xc, full_matrices=False)

    # Compute eigenvalues
    D = (S**2) / (N - 1)

    # Compute eigenvectors
    U = Vt.T

    # Select top-k components
    L = U[:, :k]
    Y = matmul(Xc, L)  # Projection of data

    # Calculate the explained variance
    var_explained = D[:k] / sum(D)

    return Y, transpose(L), D[:k], var_explained
