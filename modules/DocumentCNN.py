from Reshape import Reshape
import torch.nn as nn


class DocumentCNN(nn.Module):

    def __init__(self, alphasize, emb_dim, dropout, cnn_dim, avg):
        super(DocumentCNN, self).__init__()

        net = nn.Sequential()

        def TemporalLayer(i, ic, oc, ks, ps):
            net.add_module('tempconv {}'.format(i), nn.Conv1d(ic, oc, kernel_size=ks))
            #net.add_module('thres {}'.format(i), nn.Threshold(1e-9))
            net.add_module('dropout {}'.format(i), nn.Dropout(dropout))
            net.add_module('temppool {}'.format(i), nn.MaxPool1d(kernel_size=ps, stride=ps))
        # 336 * 256
        TemporalLayer(1, alphasize, 256, 7, 3)

        TemporalLayer(2, 256, 128, 7, 3)
        # 110 * 128
        TemporalLayer(3, 128, 64, 3, 3)
        # 36 * 64
        net.add_module('reshape', Reshape(2304))
        # 2304
        net.add_module('lin1', nn.Linear(2304, 256))
        net.add_module('lindrop', nn.Dropout(0.7))
        net.add_module('lin2', nn.Linear(256, emb_dim))
        self.net = net

    def forward(self, x):
        return self.net(x)