from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, layer: int | nn.Module = -1):
        self.model = model

        if isinstance(layer, int):
            self.layer = self.get_conv_layer(layer)
        else:
            self.layer = layer

        self.A = None
        self.alpha = None

    def get_conv_layer(self, n):
        """
        retrieves the n-th convolutional layer from the model.

        model: torch.nn.Module
            the neural network model.
        n: int
            the index of the convolutional layer to retrieve.
        """
        conv_layers = []

        # iterate through all modules and collect only convolutional layers
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                conv_layers.append(layer)

        return conv_layers[n]

    def forward_hook(self, module, input, output):
        """
        hook to capture feature maps during the forward pass.

        module: torch.nn.Module
            the convolutional layer being hooked.
        input: torch.Tensor
            the input tensor to the layer.
        output: torch.Tensor
            the output activations (feature maps).
        """
        self.A = output

    def backward_hook(self, module, grad_input, grad_output):
        """
        hook to capture gradients during the backward pass.

        module: torch.nn.Module
            the convolutional layer being hooked.
        grad_input: tuple
            gradients with respect to the input.
        grad_output: tuple
            gradients with respect to the output.
        """
        self.alpha = grad_output[0].mean(dim=(-1, -2), keepdim=True)

    def _grad_cam(
        self,
        img: torch.Tensor,
        class_idx: Optional[int | list | tuple | torch.Tensor] = None,
    ):
        if img.dim() == 3:
            img = img.unsqueeze(0)

        B, _, H, W = img.shape
        device = img.device

        output = self.model(img)

        if class_idx is None:
            target_classes = output.argmax(dim=1)
        elif isinstance(class_idx, int):
            target_classes = torch.tensor([class_idx] * B, device=device)
        elif isinstance(class_idx, (list, tuple)):
            if len(class_idx) != B:
                raise ValueError(
                    f"Length of class_idx list ({len(class_idx)}) must match batch size ({B})"
                )
            target_classes = torch.tensor(class_idx, device=device)
        else:
            target_classes = class_idx

        one_hot = torch.zeros_like(output)
        if isinstance(class_idx, int):
            one_hot[:, class_idx] = 1
        else:
            one_hot.scatter_(1, target_classes.view(-1, 1), 1)

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=False)

        if self.A is None or self.alpha is None:
            raise RuntimeError("A or alpha is None.")

        A = self.A.detach()
        alpha = self.alpha.detach()

        gcam = torch.sum(A * alpha, dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, (H, W), mode="bilinear", align_corners=False)

        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0] + 1e-8
        gcam = gcam.view(B, 1, H, W)

        return gcam

    def __call__(
        self,
        img: torch.Tensor,
        class_idx: Optional[int | list | tuple | torch.Tensor] = None,
    ):
        forward_handle = self.layer.register_forward_hook(self.forward_hook)
        backward_handle = self.layer.register_full_backward_hook(
            self.backward_hook
        )

        try:
            self.model.eval()
            gcam = self._grad_cam(img, class_idx)
        finally:
            forward_handle.remove()
            backward_handle.remove()

        return gcam
