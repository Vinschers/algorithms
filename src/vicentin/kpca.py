from vicentin.utils import exp, sqrt, eigh, argsort, sum, expand_dims, where, matmul, shape, index_select, cast, transpose


def rbf(sigma):
    """
    Constructs a radial basis function (RBF) kernel with a specified bandwidth sigma.

    The RBF kernel is defined as:
        K(xi, xj) = exp( -||xi - xj||^2 / (2 * sigma^2) ),
    where ||xi - xj||^2 represents the squared Euclidean distance between feature vectors x_i and x_j.

    Args:
        sigma (float): The bandwidth (standard deviation) parameter controlling the width of the Gaussian kernel.

    Returns:
        function:
            A function f_rbf(x_i, x_j) that computes the RBF kernel value between input vectors x_i and x_j.
    """

    def f_rbf(xi, xj):
        return exp(-sum((xi - xj) ** 2, axis=-1) / (2 * sigma**2))

    return f_rbf


def KPCA(X, kernel, k=-1):
    """
    Performs Kernel Principal Component Analysis (KPCA) on the dataset X using a specified kernel.

    KPCA is a nonlinear dimensionality reduction technique that first computes a kernel matrix
    capturing the similarity between data points, centers the kernel matrix, and then performs
    Singular Value Decomposition (SVD) to extract the principal components in a high-dimensional feature space.

    The function assumes the data has d features. The kernel function should accept two inputs
    (e.g., pairs of feature vectors) and return their similarity.

    Args:
        X (ndarray): A 2D array of shape (N, d), where N is the number of samples and d is the number of features.
        kernel (function): A kernel function that computes the similarity between pairs of samples.
        k (int, optional): The number of principal components to retain. If k is -1, all components (N) are used.
                           Defaults to -1.

    Returns:
        tuple:
            - Y (ndarray): A 2D array of shape (N, k) representing the projection of X in the kernel PCA space.
            - components (ndarray): A 2D array of shape (k, N) where each row is a principal component.
            - eigenvalues (ndarray): A 1D array of length k containing the eigenvalues of the centered kernel matrix.
            - K (ndarray): The centered kernel matrix of shape (N, N).
    """
    assert len(shape(X)) == 2, "X must be a 2D array."

    N = shape(X)[0]

    if k == -1:
        k = N

    Xi = expand_dims(X, axis=1)
    Xj = expand_dims(X, axis=0)

    K = kernel(Xi, Xj)

    # Centering K
    K_mean_row = sum(K, axis=0) / N  # Mean of each row
    K_mean_col = sum(K, axis=1, keepdims=True) / N  # Mean of each row
    K_mean_total = sum(K) / (N**2)  # Overall mean of all elements
    Kc = K - K_mean_row - K_mean_col + K_mean_total

    D, U = eigh(Kc)

    idx = argsort(D, direction="DESCENDING")
    D = index_select(D, idx, axis=0)
    U = index_select(U, idx, axis=1)

    D = where(D <= 0, cast(1e-18, D.dtype), D)  # Make non-positive eigenvalues positive and close to 0
    D = D[:k]

    U = U / sqrt((N - 1) * D[None, :])  # Scailing eigenvectors

    L = U[:, :k]  # Each column is an eigenvector
    Y = matmul(Kc, L)  # Projection of data

    return Y, transpose(L), D, K


def KPCA_project(t, X, L, K, kernel):
    """
    Projects new data points onto the kernel PCA space using the precomputed kernel matrix and principal components.

    Given new data t, this function computes the kernel values between t and the original dataset X,
    centers these kernel values in the same manner as the training kernel matrix K, and projects the result
    onto the KPCA components (contained in L).

    Args:
        t (ndarray | Tensor): A 2D array of new data points of shape (n_new, d), where d is the number of features.
        X (ndarray | Tensor): The original dataset used to compute the kernel PCA, with shape (N, d).
        L (ndarray | Tensor): The principal component matrix obtained from KPCA (typically with shape (N, k) or (k, N)
                                 depending on implementation), used for projection.
        K (ndarray | Tensor): The original kernel matrix computed from X of shape (N, N).
        kernel (function): A kernel function that computes the similarity between data points.

    Returns:
        ndarray | tf.Tensor:
            A 2D array representing the projection of the new data t onto the kernel PCA components.
    """

    N = shape(X)[0]

    Xi = expand_dims(X, axis=1)

    k = kernel(t, Xi)

    # Centering kc
    kc = k - sum(k, axis=1, keepdims=True) / N - sum(K, axis=0) / N + sum(K) / (N**2)

    return matmul(kc, L)
