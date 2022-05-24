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

        self.lambda_1 = nn.Parameter(torch.zeros(dim) - 5)
        self.lambda_2 = nn.Parameter(torch.zeros(dim) - 5)

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
        asr_mean = self.standard_mean_decoder(F.relu(self.standard_encoder(real_mean))).squeeze()
        asr_var = F.relu(self.standard_var_decoder(F.relu(self.standard_encoder(real_var)))).squeeze()
        mean = lambda_1 * asr_mean + (1 - lambda_1) * real_mean
        var = lambda_2 * asr_var + (1 - lambda_2) * real_var

        x = (x - mean.unsqueeze(2)) / (var.unsqueeze(2) + self.eps)
        if torch.sum(x > 1e6) > 0:
            print('-' * 100)
            print(x.std(2))
            print('warning! x is so big')

        asr_mean = torch.tanh(
            self.rescale_mean_decoder(F.relu(self.rescale_encoder(real_mean)))).squeeze() + self.bias_1
        asr_var = torch.sigmoid(
            self.rescale_var_decoder(F.relu(self.rescale_encoder(real_var)))).squeeze() + self.bias_2
        x = x * asr_var.unsqueeze(2) + asr_mean.unsqueeze(2)
        x = x.reshape(N, C, H, D)

        return x
