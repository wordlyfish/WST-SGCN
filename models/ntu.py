import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision

from torchstat import stat
from ptflops import get_model_complexity_info
from thop import profile

from calflops import calculate_flops

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.1):
        super(TCNBlock, self).__init__()

        pad = (kernel_size - 1) // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                    kernel_size=(kernel_size, 1),
                    padding=(pad, 0),
                    stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
            )
    def forward(self, x):
        x = self.temporal_conv(x)
        return x

class DeepTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.1, dilation=1):
        super(DeepTCNBlock, self).__init__()
        padding = ((kernel_size - 1) // 2 * dilation, 0)


        self.conv1_depthwise = nn.Conv2d(in_channels, in_channels,
                                         kernel_size=(kernel_size, 1),
                                         padding=padding,
                                         stride=(stride, 1),
                                         groups=in_channels,
                                         dilation=(dilation, 1))
        self.conv1_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)


        self.conv2_depthwise = nn.Conv2d(out_channels, out_channels,
                                         kernel_size=(kernel_size, 1),
                                         padding=padding,
                                         groups=out_channels,
                                         dilation=(dilation, 1))
        self.conv2_pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout)

        self.residual = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))

    def forward(self, x):
        res = self.residual(x)


        out = self.conv1_depthwise(x)
        out = self.conv1_pointwise(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)


        out = self.conv2_depthwise(out)
        out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = out + res
        return out

class WaveletGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix,
                 wavelet_type='mexican', scales=4, learnable_scales=True, idx=None):
        super(WaveletGraphConv, self).__init__()
        self.scales = scales
        self.wavelet_type = wavelet_type
        self.idx = idx


        self.adj_matrix = adj_matrix

        if learnable_scales:
            self.log_scale = nn.Parameter(torch.log(torch.linspace(0.1, 2.0, scales)))
        else:
            self.register_buffer('log_scale', torch.log(torch.linspace(0.1, 2.0, scales)))


        self.weights = nn.Parameter(torch.randn(scales+1, in_channels, out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

        nn.init.kaiming_normal_(self.weights, mode='fan_out')

    @property
    def scale_params(self):

        return torch.exp(self.log_scale)
    def _build_normalized_laplacian(self, adj):


        adj = 0.5 * (adj + adj.transpose(0, 1))


        adj = adj + torch.eye(adj.size(0), device=adj.device) * 0.1


        D_values = adj.sum(1)

        D_values = D_values + 1e-5


        D = torch.diag(D_values)


        L = D - adj


        try:
            lambda_max = torch.linalg.eigvalsh(L).max()

            lambda_max = max(lambda_max.item(), 1.0)
        except:

            lambda_max = D_values.max().item() * 2


        L_norm = 2 * L / lambda_max - torch.eye(adj.size(0), device=adj.device)

        return L_norm

    def _wavelet_kernel(self, s):


        if self.wavelet_type == 'mexican':

            a = 2.0 / (torch.sqrt(torch.tensor(3.0, device=s.device)) * torch.pi**0.25)
            return a * (1 - s**2) * torch.exp(-s**2 / 2)

        elif self.wavelet_type == 'morlet':

            w0 = 5.0
            return torch.exp(-s**2 / 2) * torch.cos(w0 * s)

        elif self.wavelet_type == 'meyer':

            v = torch.clamp(s, 0.0, 1.0)
            meyer = torch.where(
                v < 1/3,
                3*v - 1.5*v**3,
                torch.where(
                    v < 2/3,
                    0.5*(1 - v**2)**1.5,
                    torch.zeros_like(v)
                )
            )
            return meyer * torch.pi**0.5

        elif self.wavelet_type == 'gaussian':

            return -s * torch.exp(-s**2 / 2) / torch.sqrt(torch.tensor(2*torch.pi))

        elif self.wavelet_type == 'ricker':

            return (1 - s**2) * torch.exp(-s**2 / 2)

        elif self.wavelet_type == 'Shannon':

            return torch.sinc(s) * torch.cos(3*torch.pi*s/2)

        elif self.wavelet_type == 'ham':

            return torch.where((s >= 0.5) & (s <= 1.5), 1.0, 0.0)

        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

    def forward(self, x):
        N, C, T, V = x.size()


        L = self._build_normalized_laplacian(self.adj_matrix)

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
        except Exception as e:

            print(f"Eigendecomposition failed, using the fallback approximation: {e}")

            eigenvectors = torch.eye(V, device=x.device)

            eigenvalues = torch.diag(L)


        wavelet_bases = []
        for s in self.scale_params:
            psi = self._wavelet_kernel(s * eigenvalues)
            wavelet = eigenvectors @ torch.diag(psi) @ eigenvectors.T
            wavelet_bases.append(wavelet)


        wavelet_bases.append(torch.eye(V, device=x.device))


        out_wavelet = torch.zeros(N, self.weights.shape[-1], T, V, device=x.device)
        for i, wavelet in enumerate(wavelet_bases):

            support = torch.einsum('nctv,vw->nctw', x, wavelet)

            out_wavelet += torch.einsum('nctv,co->notv', support, self.weights[i])


        out = out_wavelet

        out = self.bn(out)
        return self.relu(out)

class WaveletBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix,
                 wavelet_params, temporal_kernel=3, stride=1,idx=None):
        super(WaveletBlock, self).__init__()


        self.wavelet_conv = WaveletGraphConv(
            in_channels, out_channels, adj_matrix,
            wavelet_type=wavelet_params['type'],
            scales=wavelet_params['scales'],
            learnable_scales=wavelet_params['learnable'],
            idx=idx
        )


        temporal_dropout = wavelet_params.get('temporal_dropout', 0.1)


        self.temporal_conv = DeepTCNBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel,
            stride=stride,
            dropout=temporal_dropout

        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        res = self.res_conv(x)
        x = self.wavelet_conv(x)
        x = self.temporal_conv(x)

        return x + res

def get_ntu_adjacency(num_joints):
    adj = torch.eye(num_joints)

    connections = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
        (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0),
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
        (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)]

    for (i, j) in connections:
        if i < num_joints and j < num_joints:
            adj[i, j] = 1
            adj[j, i] = 1

    return adj

class WGCN(nn.Module):
    def __init__(self, num_classes=60, num_point=25, num_frame=32, num_person=2, num_channel=3, dropout=0.5,
                 adj_matrix=None, wavelet_params=None, config=None):
        super(WGCN, self).__init__()


        init_adj = get_ntu_adjacency(num_point)
        init_adj = init_adj + torch.randn_like(init_adj) * 0.01


        self.adj_raw = nn.Parameter(torch.logit(init_adj.clamp(0.1, 0.9)))

        self.num_classes = num_classes
        self.num_point = num_point


        if wavelet_params is None:
            wavelet_params = {
                'type': 'morlet',  # mexican, morlet, meyer, gaussian, ricker, Shannon, ham
                'scales': 4,
                'learnable': True
            }

        if config is None:
            config = [
            [64, 64, 9], [64, 64, 9],
            [64, 64, 9], [64, 128, 3],
            [128, 128, 3], [128, 256, 3],
            [256, 256, 3], [256, 256, 3]
            ]


        self.input_map = nn.Sequential(
            nn.Conv2d(3, config[0][0], 1),
            nn.BatchNorm2d(config[0][0]),
            nn.LeakyReLU(0.1)
        )

        self.blocks = nn.ModuleList()
        in_channels = config[0][0]

        for idx, (out_channels, _, temporal_kernel) in enumerate(config):
            self.blocks.append(
                WaveletBlock(
                    in_channels, out_channels, init_adj,
                    wavelet_params=wavelet_params,
                    temporal_kernel=temporal_kernel,
                    stride=2 if idx % 2 == 0 else 1,
                    idx=idx
                )
            )
            in_channels = out_channels


        self.global_pool = nn.AdaptiveAvgPool2d(1)


        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def get_normalized_adjacency(self):


        adj = torch.sigmoid(self.adj_raw)


        adj = 0.5 * (adj + adj.transpose(0, 1))


        adj = adj + torch.eye(self.num_point, device=adj.device) * 0.1


        D_values = adj.sum(1)

        D_values = D_values + 1e-5


        D_inv_sqrt = torch.diag(torch.pow(D_values, -0.5))


        norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt

        return norm_adj

    def forward(self, x, targets=None):
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)
        x = self.input_map(x)


        current_adj = self.get_normalized_adjacency()

        mid_features = None

        for i, block in enumerate(self.blocks):
            block.wavelet_conv.adj_matrix = current_adj
            x = block(x)


        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = x.view(N, M, -1).mean(1)

        logits = self.fc(x)

        return logits


if __name__ == '__main__':


    config = [
                [96, 96, 3], [96, 96, 3],
            [192, 192, 3], [192, 192, 3],
            [192, 192, 3], [192, 192, 3],
            [192, 192, 3], [192, 192, 3],
            ]

    wavelet_config = {
        'type': 'morlet',  # mexican, morlet, meyer, gaussian, ricker, Shannon, ham
        'scales': 4,
        'learnable': True
    }
    model = WGCN(
        num_classes=60,
        num_point=25,
        config=config,
        dropout=0,
        wavelet_params=wavelet_config
    )
    data = torch.randn(1, 3, 128, 25, 2)
    output = model(data)
    print(output.shape)


    input = torch.randn(1, 3, 128, 25, 2)
    thop_flops, thop_params = profile(model, (input,))
    print('thop flops: %.2f M, params: %.2f M' % (thop_flops / 1e6, thop_params / 1e6))


    input_shape = (1, 3, 128, 25, 2)
    flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=True)
    print(f"calflops Model FLOPs:{flops} MACs:{macs} Params:{params}")

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
