import torch
import torch.nn as nn


class SmartFlatten(nn.Flatten):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__(start_dim, end_dim)
        self.captured_shape = None

    def forward(self, input):
        if self.captured_shape is None:
            self.captured_shape = input.shape[1:]
        return super().forward(input)

    def mirror(self):
        return SmartUnflatten(self)


class SmartUnflatten(nn.Module):
    def __init__(self, partner: SmartFlatten):
        super().__init__()
        self.partner = partner
        self.register_buffer("fixed_shape", torch.tensor([], dtype=torch.long))

    def forward(self, x):
        if self.fixed_shape.numel() > 0:
            return x.view(x.size(0), *self.fixed_shape.tolist())

        if self.partner.captured_shape is not None:
            shape = self.partner.captured_shape
            self.fixed_shape = torch.tensor(shape, device=x.device)
            return x.view(x.size(0), *shape)

        raise RuntimeError(
            "SmartUnflatten shape not initialized! Run a dummy input through the Encoder first."
        )
