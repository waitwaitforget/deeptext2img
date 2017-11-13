import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *size):
        super(Reshape, self).__init__()

        self.size = size

    def forward(self, x):
        return x.view_(self.size)
