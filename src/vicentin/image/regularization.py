import numpy as np
from vicentin.image.differentiation import grad


def tychonov(img):
    dx, dy = grad(img)
    return np.sum(dx**2 + dy**2)


def total_variation(img, epsilon=1e-2):
    dx, dy = grad(img)
    return np.sum(np.sqrt(epsilon**2 + dx**2 + dy**2))
