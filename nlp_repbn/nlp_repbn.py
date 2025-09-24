import torch
import torch.nn as nn

class RepBN_NLP(nn.Module):
    """
    RepBN_NLP applies BatchNorm1d along the embedding/features dimension for NLP tasks.
    It also introduces a learnable alpha parameter for residual scaling.
    Input: Tensor of shape [batch_size, seq_len, embed_dim]
    """
    def __init__(self, embed_dim):
        super(RepBN_NLP, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        x = x.transpose(1, 2)             # [batch_size, embed_dim, seq_len]
        x_bn = self.bn(x)                 # [batch_size, embed_dim, seq_len]
        x = x_bn + self.alpha * x         # Residual scaling
        x = x.transpose(1, 2)             # [batch_size, seq_len, embed_dim]
        return x