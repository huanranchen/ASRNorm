import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        '''

        :param dim: C of N,C,H,D
        '''
        super(ASRNorm, self).__init__()
        self.eps = eps

        self.standard_encoder = nn.Linear(dim, dim // 2)
        self.rescale_encoder = nn.Linear(dim, dim // 16)
        self.standard_mean_decoder = nn.Linear(dim // 2, dim)
        self.standard_var_decoder = nn.Linear(dim // 2, dim)
        self.rescale_mean_decoder = nn.Linear(dim // 16, dim)
        self.rescale_var_decoder = nn.Linear(dim // 16, dim)

        self.lambda_1 = nn.Parameter(torch.zeros(dim))
        self.lambda_2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        '''

        :param x: N,C,H,D
        :return:
        '''
        lambda_1 = torch.sigmoid(self.lambda_1)
        lambda_2 = torch.sigmoid(self.lambda_2)
        N, C, H, D = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        mean = x.mean(0)
        var = x.std(0)
        standard_encoded = F.relu(self.standard_encoder(x))
        asr_mean = self.standard_mean_decoder(standard_encoded)
        asr_var = F.relu(self.standard_var_decoder(standard_encoded))
        mean = lambda_1 * asr_mean + (1 - lambda_1) * mean
        var = lambda_2 * asr_var + (1 - lambda_2) * var
        x = (x - mean) / (var + self.eps)

        rescale_encoded = F.relu(self.rescale_encoder(x))
        asr_mean = torch.tanh(self.rescale_mean_decoder(rescale_encoded))
        asr_var = torch.sigmoid(self.rescale_var_decoder(rescale_encoded))

        x = x * asr_mean + asr_var

        x = x.reshape(N, H, D, C)
        x = x.permute(0, 3, 1, 2)
        return x
