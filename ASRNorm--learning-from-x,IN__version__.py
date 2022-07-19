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

        self.standard_encoder = nn.Linear(dim, dim // 16)
        self.rescale_encoder = nn.Linear(dim, dim // 16)
        self.standard_mean_decoder = nn.Linear(dim // 16, dim)
        self.standard_var_decoder = nn.Linear(dim // 16, dim)
        self.rescale_mean_decoder = nn.Linear(dim // 16, dim)
        self.rescale_var_decoder = nn.Linear(dim // 16, dim)

        self.lambda_1 = nn.Parameter(torch.zeros(dim) - 10)
        self.lambda_2 = nn.Parameter(torch.zeros(dim) - 10)

        self.bias_1 = nn.Parameter(torch.zeros(dim))
        self.bias_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        '''

        :param x: N,C,H,D
        :return:
        '''
        lambda_1 = torch.sigmoid(self.lambda_1)
        lambda_2 = torch.sigmoid(self.lambda_2)
        N, C, H, D = x.shape
        x = x.reshape(N, C, H * D)
        real_mean = x.mean(2)
        real_var = x.std(2)
        asr_mean = self.standard_mean_decoder(F.relu(self.standard_encoder(x.permute(0, 2, 1)))).permute(0, 2, 1)
        asr_var = F.relu(self.standard_var_decoder(F.relu(self.standard_encoder(x.permute(0, 2, 1))))).permute(0, 2, 1)
        mean = lambda_1 * asr_mean + (1 - lambda_1) * real_mean.unsqueeze(2)
        var = lambda_2 * asr_var + (1 - lambda_2) * real_var.unqueeze(2)

        x = (x - mean) / (var + self.eps)

        asr_mean = torch.tanh(
            self.rescale_mean_decoder(F.relu(self.rescale_encoder(x.permute(0, 2, 1))))) + self.bias_1
        asr_var = torch.sigmoid(
            self.rescale_var_decoder(F.relu(self.rescale_encoder(x.permute(0, 2, 1))))) + self.bias_2
        x = x * asr_var.permute(0, 2, 1) + asr_mean.permute(0, 2, 1)
        x = x.reshape(N, C, H, D)

        return x
