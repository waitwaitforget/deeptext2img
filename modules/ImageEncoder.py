import torch.nn as nn


class ImageEncoder(nn.Module):
    '''
    Image encoder: input must be feature vectors
    '''
    def __init__(self, isz, hsz, noop):
        super(ImageEncoder, self).__init__()

        self.isz = isz
        self.hsz = hsz
        self.noop = noop
        if not self.noop:
            self.net = nn.Linear(isz, hsz)

    def forward(self, x):
        if self.noop:
            return x
        else:
            return self.net(x)
