import torch
import torch.nn as nn

from core.gcn.rgcn import RGCNModel


class RGCNTModel(nn.Module):
    """
    A RGCN Model + a Temperature parameter for the Soft Nearest Neighbor Loss
    """

    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_hidden_layers=1, temperature=1):
        super(RGCNTModel, self).__init__()
        self.rgcn = RGCNModel(num_nodes, h_dim, out_dim, num_rels, num_bases, num_hidden_layers)
        self.inv_temperature = nn.Parameter(torch.Tensor([1 / temperature]))

    def get_temperature(self):
        return 1 / self.inv_temperature

    def forward(self, g):
        return self.rgcn.forward(g)
