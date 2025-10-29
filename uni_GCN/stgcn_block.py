"""
Spatio-Temporal Graph Convolutional Network blocks.

This module implements the core ST-GCN building blocks including:
- GCN_unit: Spatial graph convolution
- STGCN_block: Combined spatial-temporal convolution with residual connections
- STGCNChain: Sequential chains of STGCN blocks
"""

import torch
import torch.nn as nn
import numpy as np


class GCN_unit(nn.Module):
    """Graph Convolutional Unit for spatial modeling

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of graph convolution kernel
        A (torch.Tensor): Adjacency matrix
        adaptive (bool): Whether to use adaptive adjacency matrix
        t_kernel_size (int): Temporal kernel size
        t_stride (int): Temporal stride
        t_padding (int): Temporal padding
        t_dilation (int): Temporal dilation
        bias (bool): Whether to use bias in convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        assert A.size(0) == self.kernel_size

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

        self.adaptive = adaptive
        if self.adaptive:
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, T, V)
            len_x: Sequence length (unused, for compatibility)

        Returns:
            torch.Tensor: Output tensor after graph convolution
        """
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()

        y = self.bn(x)
        y = self.relu(y)
        return y


class STGCN_block(nn.Module):
    """Spatio-Temporal Graph Convolutional block

    Combines spatial graph convolution with temporal convolution and residual connection.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (tuple): (temporal_size, spatial_size)
        A (torch.Tensor): Adjacency matrix
        adaptive (bool): Whether to use adaptive adjacency matrix
        stride (int): Temporal stride
        dropout (float): Dropout probability
        residual (bool): Whether to use residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)

        # Spatial graph convolution
        self.gcn = GCN_unit(
            in_channels,
            out_channels,
            kernel_size[1],
            A,
            adaptive=adaptive,
        )

        # Temporal convolution
        if kernel_size[0] > 1:
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size[0], 1),
                    (stride, 1),
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )
        else:
            self.tcn = nn.Identity()

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, T, V)
            len_x: Sequence length (unused, for compatibility)

        Returns:
            torch.Tensor: Output tensor after ST-GCN block
        """
        res = self.residual(x)
        x = self.gcn(x, len_x)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNChain(nn.Sequential):
    """Sequential chain of STGCN blocks

    Args:
        in_dim (int): Input dimension
        block_args (list): List of [channel, depth] pairs for each stage
        kernel_size (tuple): Kernel size for ST-GCN blocks
        A (torch.Tensor): Adjacency matrix
        adaptive (bool): Whether to use adaptive adjacency matrix
    """

    def __init__(self, in_dim, block_args, kernel_size, A, adaptive):
        super(STGCNChain, self).__init__()

        last_dim = in_dim
        for i, [channel, depth] in enumerate(block_args):
            for j in range(depth):
                self.add_module(
                    f'layer{i}_{j}',
                    STGCN_block(last_dim, channel, kernel_size, A.clone(), adaptive)
                )
                last_dim = channel


def get_stgcn_chain(in_dim, level, kernel_size, A, adaptive):
    """Create ST-GCN chain with predefined architectures

    Args:
        in_dim (int): Input dimension
        level (str): Architecture level ('spatial' or 'temporal')
        kernel_size (tuple): Kernel size for ST-GCN blocks
        A (torch.Tensor): Adjacency matrix
        adaptive (bool): Whether to use adaptive adjacency matrix

    Returns:
        tuple: (STGCNChain module, output dimension)
    """
    if level == 'spatial':
        # Spatial processing: 3-stage channel expansion
        block_args = [[64, 1], [128, 1], [256, 1]]
    elif level == 'temporal':
        # Temporal processing: 3-layer temporal modeling
        block_args = [[256, 3]]
    else:
        raise NotImplementedError(f"Unsupported level: {level}")

    return STGCNChain(in_dim, block_args, kernel_size, A, adaptive), block_args[-1][0]
