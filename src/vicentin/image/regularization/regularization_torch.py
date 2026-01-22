import torch
from vicentin.image.differentiation import grad


def tychonov(img):
    dx, dy = grad(img)
    return torch.sum(dx**2 + dy**2)


def total_variation(img, epsilon=1e-2):
    dx, dy = grad(img)
    return torch.sum(torch.sqrt(epsilon**2 + dx**2 + dy**2))
