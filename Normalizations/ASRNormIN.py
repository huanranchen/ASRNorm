import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ASRNormIN(nn.Module):
    def __init__(self, dim, eps=1e-6, init_beta=None, init_gamma=None):
        '''

        :param dim: C of N,C,H,D
        '''
        super(ASRNormIN, self).__init__()

        self.eps = eps
        self.num_channels = dim
        self.stan_mid_channel = self.num_channels // 2
        self.rsc_mid_channel = self.num_channels // 16

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.standard_encoder = nn.Linear(dim, self.stan_mid_channel)  # 16
        self.rescale_encoder = nn.Linear(dim, self.rsc_mid_channel)

        # standardization
        self.standard_mean_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.stan_mid_channel, dim)
        )

        self.standard_std_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.stan_mid_channel, dim),
            self.relu
        )

        # Rescaling
        self.rescale_beta_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.rsc_mid_channel, dim),
            self.tanh
        )

        self.rescale_gamma_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.rsc_mid_channel, dim),
            self.sigmoid
        )

        self.lambda_mu = nn.Parameter(torch.empty(1))
        self.lambda_sigma = nn.Parameter(torch.empty(1))

        # 如果是PACS多源泛化，那么我们会含有这俩个参数反之不含有
        # 与此同时，这个时候上面的 bias 都是 pretrained model 里面的固定值
        # if pacs:
        self.lambda_beta = nn.Parameter(torch.empty(1))
        self.lambda_gamma = nn.Parameter(torch.empty(1))

        # else:
        self.bias_beta = nn.Parameter(torch.empty(dim))
        self.bias_gamma = nn.Parameter(torch.empty(dim))


        # init lambda and bias
        with torch.no_grad():
            init.constant_(self.lambda_mu, self.sigmoid(torch.tensor(-3)))
            init.constant_(self.lambda_sigma, self.sigmoid(torch.tensor(-3)))
            init.constant_(self.lambda_beta, self.sigmoid(torch.tensor(-5)))
            init.constant_(self.lambda_gamma, self.sigmoid(torch.tensor(-5)))

            if init_beta is None:
                init.constant_(self.bias_beta, 0.)
            else:
                self.bias_beta.copy_(init_beta)
            if init_gamma is None:
                init.constant_(self.bias_gamma, 1.)
            else:
                self.bias_gamma.copy_(init_gamma)

    def forward(self, x):
        '''

        :param x: N,C,H,W
        :return:
        '''

        N, C, H, W = x.size()
        x_mean = torch.mean(x, dim=(2, 3))
        x_std = torch.sqrt(torch.var(x, dim=(2, 3)) + self.eps)

        # standardization
        x_standard_mean = self.standard_mean_decoder(self.standard_encoder(x_mean))
        x_standard_std = self.standard_std_decoder(self.standard_encoder(x_std))

        lambda_sigma = self.sigmoid(self.lambda_sigma)
        lambda_mu = self.sigmoid(self.lambda_mu)

        mean = lambda_mu * x_standard_mean + (1 - lambda_mu) * x_mean
        std = lambda_sigma * x_standard_std + (1 - lambda_sigma) * x_std

        mean = mean.reshape((N, C, 1, 1))
        std = std.reshape((N, C, 1, 1))

        x = (x - mean) / std

        # rescaling
        x_rescaling_beta = self.rescale_beta_decoder(self.rescale_encoder(x_mean))
        x_rescaling_gamma = self.rescale_gamma_decoder(self.rescale_encoder(x_std))

        # if self.pacs:
        beta = self.lambda_beta * x_rescaling_beta + self.bias_beta
        gamma = self.lambda_gamma * x_rescaling_gamma + self.bias_gamma
        # else:
        # beta = x_rescaling_beta + self.bias_beta
        # gamma = x_rescaling_gamma + self.bias_gamma

        beta = beta.reshape((N, C, 1, 1))
        gamma = gamma.reshape((N, C, 1, 1))

        x = x * gamma + beta

        return x


def set_module(model, name, norm_layer):
    """Replace module of model by name with multi-level path"""
    path = name.split('.')
    cur = model
    for p in path[:-1]:
        cur = getattr(cur, p)
    setattr(cur, path[-1], norm_layer)


# 先试试行不行，如果行，再进行修改
def build_ASRNormIN(model:nn.Module, init=False):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            set_module(model, name, ASRNormIN(module.num_features, init_beta=(module.bias if init else None), init_gamma=(module.weight if init else None)))
