import torch

def shift_right(x: torch.Tensor):

    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    return x