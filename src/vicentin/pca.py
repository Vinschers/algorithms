from vicentin.utils import mean, SVD


def PCA(X):
    """
    Performs Principal Component Analysis (PCA) on a given dataset.

    PCA is a dimensionality reduction technique that transforms data into a set of orthogonal
    components that maximize variance. This implementation computes the principal components
    using Singular Value Decomposition (SVD) on the covariance matrix.

    Time Complexity:
        - O(N * M^2), where N is the number of samples and M is the number of features,
          assuming SVD is computed efficiently.

    Space Complexity:
        - O(M^2), for storing the covariance matrix and singular vectors.

    Args:
        X (numpy.ndarray): A 2D array of shape (N, M), where N is the number of samples
                           and M is the number of features.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - `D`: A 1D array containing the eigenvalues of the covariance matrix,
                   representing the variance explained by each principal component.
            - `Ut`: A 2D array where each row is a principal component (eigenvector),
                    sorted in descending order of eigenvalues.

    Raises:
        AssertionError: If the input array `X` is not a 2D array.
    """
    assert len(X.shape) == 2, "X must be a 2D array."

    N = X.shape[0]

    Xc = X - mean(X, axis=0)
    _, S, Vt = SVD(Xc, full_matrices=False)

    D = (S**2) / (N - 1)

    return D, Vt
