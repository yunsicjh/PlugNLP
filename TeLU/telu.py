import torch


def TeLU(input):
    return input * torch.tanh(torch.exp(input))
