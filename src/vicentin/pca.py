from vicentin.utils import mean, SVD


def PCA(X):
    assert len(X.shape) == 2, "X must be a 2D array."

    N = X.shape[0]

    Xc = X - mean(X, axis=0)
    _, D, Ut = SVD(Xc.T @ Xc, full_matrices=False)

    D /= N - 1

    return D, Ut
