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

#from calflops import calculate_flops

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

        # Layer 1
        self.conv1_depthwise = nn.Conv2d(in_channels, in_channels, 
                                         kernel_size=(kernel_size, 1), 
                                         padding=padding, 
                                         stride=(stride, 1), # Stride on the first conv of the block
                                         groups=in_channels,
                                         dilation=(dilation, 1))
        self.conv1_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2
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

        # Layer 1
        out = self.conv1_depthwise(x)
        out = self.conv1_pointwise(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Layer 2
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

        # 存储邻接矩阵但不注册为buffer
        self.adj_matrix = adj_matrix

        if learnable_scales:
            self.log_scale = nn.Parameter(torch.log(torch.linspace(0.1, 2.0, scales)))
        else:
            self.register_buffer('log_scale', torch.log(torch.linspace(0.1, 2.0, scales)))
        
        # 可学习权重
        self.weights = nn.Parameter(torch.randn(scales+1, in_channels, out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

        nn.init.kaiming_normal_(self.weights, mode='fan_out')

    @property
    def scale_params(self):
        # 动态计算，确保与当前设备一致
        return torch.exp(self.log_scale)  # 自动继承 log_scale 的设备
    def _build_normalized_laplacian(self, adj):
        """构建归一化图拉普拉斯矩阵，增加数值稳定性"""
        # 确保邻接矩阵是对称的
        adj = 0.5 * (adj + adj.transpose(0, 1))
        
        # 添加自连接并确保数值稳定性
        adj = adj + torch.eye(adj.size(0), device=adj.device) * 0.1
        
        # 度矩阵的对角线值
        D_values = adj.sum(1)
        # 添加小的扰动以确保可逆
        D_values = D_values + 1e-5
        
        # 构建度矩阵
        D = torch.diag(D_values)
        
        # 拉普拉斯矩阵
        L = D - adj
        
        # 计算最大特征值，添加稳定性处理
        try:
            lambda_max = torch.linalg.eigvalsh(L).max()
            # 确保lambda_max不会太小
            lambda_max = max(lambda_max.item(), 1.0)
        except:
            # 如果特征值计算失败，使用一个合理的默认值
            lambda_max = D_values.max().item() * 2
        
        # 归一化拉普拉斯矩阵
        L_norm = 2 * L / lambda_max - torch.eye(adj.size(0), device=adj.device)
        
        return L_norm

    def _wavelet_kernel(self, s):
        """扩展的小波核函数集合"""
        # s: 缩放后的特征值张量，形状为 [scales, V] 或 [V]
        
        if self.wavelet_type == 'mexican':
            # 修正后的墨西哥帽小波 (与pywt一致)
            a = 2.0 / (torch.sqrt(torch.tensor(3.0, device=s.device)) * torch.pi**0.25)
            return a * (1 - s**2) * torch.exp(-s**2 / 2)

        elif self.wavelet_type == 'morlet':
            # 简化版Morlet小波（实数版本，省略复数部分）
            w0 = 5.0  # 中心频率参数
            return torch.exp(-s**2 / 2) * torch.cos(w0 * s)

        elif self.wavelet_type == 'meyer':
            # Meyer小波（分段函数实现）
            v = torch.clamp(s, 0.0, 1.0)  # 确保在[0,1]区间处理
            meyer = torch.where(
                v < 1/3,
                3*v - 1.5*v**3,
                torch.where(
                    v < 2/3,
                    0.5*(1 - v**2)**1.5,
                    torch.zeros_like(v)
                )
            )
            return meyer * torch.pi**0.5  # 归一化系数

        elif self.wavelet_type == 'gaussian':
            # 高斯一阶导数（类似Mexican hat的变体）
            return -s * torch.exp(-s**2 / 2) / torch.sqrt(torch.tensor(2*torch.pi))

        elif self.wavelet_type == 'ricker':
            # Ricker小波（与Mexican hat等价）
            return (1 - s**2) * torch.exp(-s**2 / 2)

        elif self.wavelet_type == 'Shannon':
            # Shannon小波（理想带通滤波器）
            return torch.sinc(s) * torch.cos(3*torch.pi*s/2)

        elif self.wavelet_type == 'ham':
            # 自定义的Ham小波（矩形窗调整版）
            return torch.where((s >= 0.5) & (s <= 1.5), 1.0, 0.0)
        
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

    def forward(self, x):
        N, C, T, V = x.size()

        # 使用当前的邻接矩阵计算拉普拉斯矩阵
        L = self._build_normalized_laplacian(self.adj_matrix)
        # 动态计算特征分解，添加错误处理
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
        except Exception as e:
            # 如果特征分解失败，使用近似方法
            print(f"特征分解失败，使用近似方法: {e}")
            # 使用随机正交矩阵作为近似特征向量
            eigenvectors = torch.eye(V, device=x.device)
            # 使用对角线元素作为近似特征值
            eigenvalues = torch.diag(L)
        
        # 小波变换
        wavelet_bases = []
        for s in self.scale_params:
            psi = self._wavelet_kernel(s * eigenvalues)
            wavelet = eigenvectors @ torch.diag(psi) @ eigenvectors.T
            wavelet_bases.append(wavelet)
        
        # 包括低频分量
        wavelet_bases.append(torch.eye(V, device=x.device))  # 恒等变换
        
        # 小波卷积 (计算高频和低频分量)
        out_wavelet = torch.zeros(N, self.weights.shape[-1], T, V, device=x.device)
        for i, wavelet in enumerate(wavelet_bases):
            # 空域卷积
            support = torch.einsum('nctv,vw->nctw', x, wavelet)
            # 特征变换
            out_wavelet += torch.einsum('nctv,co->notv', support, self.weights[i])

        # 语义引导特征学习（仅在前三层且标签可用时）
        out = out_wavelet # 默认输出是小波特征
        
        out = self.bn(out)
        return self.relu(out)

class WaveletBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix, 
                 wavelet_params, temporal_kernel=3, stride=1,idx=None):
        super(WaveletBlock, self).__init__()
        
        # 小波图卷积
        self.wavelet_conv = WaveletGraphConv(
            in_channels, out_channels, adj_matrix, 
            wavelet_type=wavelet_params['type'],
            scales=wavelet_params['scales'],
            learnable_scales=wavelet_params['learnable'],
            idx=idx
        )

        # 时间卷积
        temporal_dropout = wavelet_params.get('temporal_dropout', 0.1) # Get dropout from params, default 0.1

        # 直接使用 DeepTCNBlock 作为时间卷积模块
        self.temporal_conv = DeepTCNBlock(
            in_channels=out_channels, # 输入通道为GCN的输出通道
            out_channels=out_channels, # 输出通道保持不变
            kernel_size=temporal_kernel, # 使用配置中指定的temporal_kernel
            stride=stride, # 使用WaveletBlock的stride进行时间下采样
            dropout=temporal_dropout
            # dilation is default 1 in DeepTCNBlock
        )
        # 残差连接
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
        if i < num_joints and j < num_joints:  # 确保索引有效
            adj[i, j] = 1
            adj[j, i] = 1
    
    return adj

class WGCN(nn.Module):
    def __init__(self, num_classes=60, num_point=25, num_frame=32, num_person=2, num_channel=3, dropout=0.5,
                 adj_matrix=None, wavelet_params=None, config=None):
        super(WGCN, self).__init__()

        # 初始化邻接矩阵，添加小的随机扰动以打破对称性
        init_adj = get_ntu_adjacency(num_point)
        init_adj = init_adj + torch.randn_like(init_adj) * 0.01
        
        # 创建可学习的邻接矩阵参数，使用sigmoid确保值在[0,1]范围内
        self.adj_raw = nn.Parameter(torch.logit(init_adj.clamp(0.1, 0.9)))

        self.num_classes = num_classes
        self.num_point = num_point

        # 默认参数
        if wavelet_params is None:
            wavelet_params = {
                'type': 'morlet',
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
        
        # 网络结构
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
        
        # 分类层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        # self.fc = nn.Sequential(
        #     nn.Linear(in_channels, 512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, num_classes)
        # )``
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def get_normalized_adjacency(self):
        """获取归一化的邻接矩阵，确保数值稳定性"""
        # 使用sigmoid确保邻接矩阵元素在[0,1]范围内
        adj = torch.sigmoid(self.adj_raw)
        
        # 确保邻接矩阵是对称的
        adj = 0.5 * (adj + adj.transpose(0, 1))
        
        # 添加自连接
        adj = adj + torch.eye(self.num_point, device=adj.device) * 0.1
        
        # 对称归一化 - 使用更稳定的方式计算
        D_values = adj.sum(1)
        # 添加小的扰动以确保可逆
        D_values = D_values + 1e-5
        
        # 计算D^(-1/2)
        D_inv_sqrt = torch.diag(torch.pow(D_values, -0.5))
        
        # 计算归一化邻接矩阵
        norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return norm_adj

    def forward(self, x, targets=None):
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)
        x = self.input_map(x)

        # 获取当前的归一化邻接矩阵
        current_adj = self.get_normalized_adjacency()

        mid_features = None
        
        for i, block in enumerate(self.blocks):
            block.wavelet_conv.adj_matrix = current_adj
            x = block(x)
        
        # final_features = self.global_pool(x).squeeze(-1).squeeze(-1)
        # final_features = final_features.view(N, M, -1).mean(1)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = x.view(N, M, -1).mean(1)
        # 分类结果
        logits = self.fc(x)
        
        return logits


if __name__ == '__main__':
    # Example usage
    adj = torch.eye(25)  # Identity matrix as dummy adjacency
    
    config = [
                [96, 96, 3], [96, 96, 3],
            [192, 192, 3], [192, 192, 3],
            [192, 192, 3], [192, 192, 3],
            [192, 192, 3], [192, 192, 3],
            ]

#     config = [ [64, 64, 9], [64, 64, 9],
#             [64, 128, 9], [128, 128, 3],
#             [128, 256, 3], [256, 256, 3],
#             [256, 256, 3], [256, 256, 3],
#   ]
              
    wavelet_config = {
        'type': 'morlet',  # 可选'mexican'或'ham'
        't_type': 'morlet',
        'scales': 4,
        't_scales': 8,
        'learnable': True
    }
    model = WGCN(
        num_classes=60,
        num_point=25,
        config=config
    )
    data = torch.randn(1, 3, 128, 25, 2)  # (N, C, T, V, M)
    output = model(data)
    print(output.shape)  # Should be (2, 60)

    # # ptflops
    # flops, params = get_model_complexity_info(model, (3, 128, 25, 2), as_strings=True, print_per_layer_stat=True)
    # print('ptflops flops: ', flops, 'params: ', params)

    # thop
    input = torch.randn(1, 3, 128, 25, 2)
    thop_flops, thop_params = profile(model, (input,))
    print('thop flops: %.2f M, params: %.2f M' % (thop_flops / 1e6, thop_params / 1e6))

    #calflops
    input_shape = (1, 3, 128, 25, 2)
    flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=True)
    print(f"calflops Model FLOPs:{flops} MACs:{macs} Params:{params}")
    
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))

    # # =========
    # flops_counter = TorchFLOPsByFX(model)
    # # flops_counter.graph_model.graph.print_tabular()
    # inputx = torch.randn(1, 3, 128, 25, 2)
    # flops_counter.propagate(inputx)
    # flops_counter.print_result_table()
    # flops_1 = flops_counter.print_total_flops(show=False)
    # print(f"torch_flops: {flops_1} FLOPs")
    # print("=" * 80)

