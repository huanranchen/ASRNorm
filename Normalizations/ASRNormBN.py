import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRNormBN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        '''

        :param dim: C of N,C,H,D
        '''
        super(ASRNormBN, self).__init__()
        self.eps = eps

        self.standard_encoder = nn.Linear(dim, dim // 16)
        self.rescale_encoder = nn.Linear(dim, dim // 16)
        self.standard_mean_decoder = nn.Linear(dim // 16, dim)
        self.standard_var_decoder = nn.Linear(dim // 16, dim)
        self.rescale_mean_decoder = nn.Linear(dim // 16, dim)
        self.rescale_var_decoder = nn.Linear(dim // 16, dim)

        self.lambda_1 = nn.Parameter(torch.zeros(dim)-5)
        self.lambda_2 = nn.Parameter(torch.zeros(dim)-5)

        self.bias_1 = nn.Parameter(torch.zeros(dim))
        # training image net in one hour suggest to initialize as 0
        self.bias_2 = nn.Parameter(torch.zeros(dim))

    def init(self):
        pass

    def forward(self, x):
        '''

        :param x: N,C,H,D
        :return:
        '''
        lambda_1 = torch.sigmoid(self.lambda_1)
        lambda_2 = torch.sigmoid(self.lambda_2)
        N, C, H, D = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        real_mean = x.mean(0)
        real_var = x.std(0)
        asr_mean = self.standard_mean_decoder(F.relu(self.standard_encoder(real_mean.view(1, -1)))).squeeze()
        asr_var = F.relu(self.standard_var_decoder(F.relu(self.standard_encoder(real_var.view(1, -1))))).squeeze()
        mean = lambda_1 * asr_mean + (1 - lambda_1) * real_mean
        var = lambda_2 * asr_var + (1 - lambda_2) * real_var

        x = (x - mean) / (var + self.eps)

        asr_mean = torch.tanh(self.rescale_mean_decoder(
            F.relu(self.rescale_encoder(real_mean.view(1, -1))))).squeeze() + self.bias_1
        asr_var = torch.sigmoid(
            self.rescale_var_decoder(
                F.relu(self.rescale_encoder(real_var.view(1, -1))))).squeeze() + self.bias_2
        x = x * asr_var + asr_mean
        x = x.reshape(N, H, D, C)
        x = x.permute(0, 3, 1, 2)

        return x
