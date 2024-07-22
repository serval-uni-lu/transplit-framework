import torch
import torch.nn as nn
import torch.nn.functional as F


class SVS(nn.Module):
    def __init__(self, c_in, d_model, c_out, period, n_filters):
        super(SVS, self).__init__()

        self.c_in = c_in
        self.d_model = d_model
        self.c_out = c_out
        self.period = period
        self.n_filters = n_filters

        self.conv_layer = nn.Conv1d(c_in, n_filters, kernel_size=period, stride=period, bias=False)
        self.conv_layer.to("cuda")

        self.embedding = nn.Linear(n_filters, d_model)
        self.activation = nn.ReLU()


    def encode(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layer(x)
        x = x.transpose(1, 2)
        x = self.embedding(x)
        return self.activation(x)


    def forward(self, x):
        return self.encode(x)


    def decode(self, x):
        # x.shape = (batch_size, y_seq_len // period, n_filters)
        # conv_layer.weight.shape = (n_filters, c_in, period)
        W = self.conv_layer.weight
        W = W.transpose(1, 2)[..., :self.c_out]
        x = torch.einsum('btf,fpc->btpc', x, W)
        x = x.flatten(start_dim=1, end_dim=2)
        return x
