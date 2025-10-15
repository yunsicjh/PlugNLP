import torch
from torch import nn


class DynamicRepBNForNLP(nn.Module):
    """用于NLP任务的动态RepBN

    Args:
        hidden_dim (int): 隐藏层维度，通常是word embedding或transformer的输出维度
        eps (float): BatchNorm的eps参数，默认1e-5
        momentum (float): BatchNorm的momentum参数，默认0.1

    Input shape:
        - (batch_size, seq_len, hidden_dim)
    Output shape:
        - (batch_size, seq_len, hidden_dim)
    """

    def __init__(self, hidden_dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = nn.Parameter(torch.ones(1))
        # 对hidden_dim维度做归一化
        self.bn = nn.BatchNorm1d(hidden_dim, eps=eps, momentum=momentum)
        self.scale_factor = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, hidden_dim)
        """
        # 保存原始形状
        batch_size, seq_len, hidden_dim = x.shape

        # 转换维度以适应BatchNorm1d
        # (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
        x_reshaped = x.reshape(-1, hidden_dim)

        # 计算方差 - 在batch和sequence维度上
        var = torch.var(x, dim=(0, 1), keepdim=True)

        # 计算动态权重
        bn_weight = torch.sigmoid(self.scale_factor * var)

        # 应用BatchNorm
        x_bn = self.bn(x_reshaped)

        # 动态加权
        output = bn_weight * x_bn + (1 - bn_weight) * self.alpha * x_reshaped

        # 恢复原始形状
        output = output.reshape(batch_size, seq_len, hidden_dim)

        return output


# 测试代码
if __name__ == "__main__":
    # NLP任务的典型参数
    batch_size = 32  # 批大小
    seq_len = 128  # 序列长度
    hidden_dim = 768  # 隐藏层维度(如BERT-base)

    # 创建模型
    model = DynamicRepBNForNLP(hidden_dim=hidden_dim)
    print(model)
    print("适配NLP任务的DynamicRepBN")

    # 生成随机输入张量 (batch_size, seq_len, hidden_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    print("Input shape:", x.shape)

    # 前向传播
    output = model(x)
    print("Output shape:", output.shape)

    # 验证输出维度不变
    assert output.shape == x.shape, "输出维度应与输入维度相同"
