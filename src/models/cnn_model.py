import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T) -> reshape to (B * N, C, T)
        B, C, N, T = x.shape
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * N, C, T)
        out = self.conv(x_flat)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = out.view(B, N, -1, T).permute(0, 2, 1, 3).contiguous()
        return out


class AdaptiveGraphConv(nn.Module):
    """
    Adaptive Graph Convolution layer (Graph WaveNet style).
    A_adp = Softmax(ReLU(E1 @ E2.T))
    """
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, adj_adp: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T)
        # adj_adp: (N, N)
        B, C, N, T = x.shape
        x_perm = x.permute(0, 3, 1, 2).contiguous().view(B * T * C, N)
        diffused = torch.matmul(x_perm, adj_adp.t()).view(B, T, C, N).permute(0, 2, 3, 1).contiguous()
        return self.linear(diffused)


class DilatedGraphResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 2, dilation: int = 1, dropout: float = 0.1, use_adaptive_graph: bool = True):
        super().__init__()
        self.use_adaptive_graph = use_adaptive_graph
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        if use_adaptive_graph:
            self.gconv = AdaptiveGraphConv(channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, adj_adp: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, N, T)
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        gated = filter_out * gate_out
        gated = self.dropout(gated)

        # Spatial adaptive graph convolution (optional)
        if self.use_adaptive_graph and adj_adp is not None:
            gconv_out = self.gconv(gated, adj_adp)
            fused = gated + gconv_out
        else:
            fused = gated

        # Skip connection to output head
        skip = self.skip_conv(fused)

        # Residual connection
        proj = self.proj(fused)
        out = self.norm(x + proj)
        return out, skip


class Dilated1DCNN(nn.Module):
    """
    Adaptive Graph-Enhanced Dilated Causal Convolutional Network.
    Integrates WaveNet causal dilated residual blocks with learnable adaptive graph diffusion,
    multi-scale skip aggregation, and residual velocity forecasting.
    """
    def __init__(
        self,
        num_nodes: int = 207,
        in_features: int = 2,
        in_steps: int = 12,
        out_steps: int = 12,
        channels: int = 64,
        node_dim: int = 16,
        kernel_size: int = 2,
        dilations: List[int] = [1, 2, 4, 8],
        dropout: float = 0.1,
        use_adaptive_graph: bool = True,
        use_residual_head: bool = True,
        **kwargs
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.channels = channels
        self.use_adaptive_graph = use_adaptive_graph
        self.use_residual_head = use_residual_head

        # Adaptive Node Embeddings for Dynamic Spatial Graph Learning
        if use_adaptive_graph:
            self.node_vec1 = nn.Parameter(torch.randn(num_nodes, node_dim))
            self.node_vec2 = nn.Parameter(torch.randn(num_nodes, node_dim))

        self.input_proj = nn.Conv2d(in_features, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            DilatedGraphResidualBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
                use_adaptive_graph=use_adaptive_graph
            )
            for d in dilations
        ])

        # WaveNet Multi-Scale Skip Aggregation Head
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, out_steps, kernel_size=(1, in_steps))
        )

        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, in_steps, num_nodes, in_features)
        B, T, N, F_dim = x.shape

        # Last observed speed reading: (B, N)
        last_speed = x[:, -1, :, 0]

        # Construct dynamic adaptive adjacency matrix: (N, N)
        if self.use_adaptive_graph:
            adj_adp = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2.t())), dim=-1)
        else:
            adj_adp = None

        # Reshape to (B, in_features, num_nodes, in_steps)
        h = x.permute(0, 3, 2, 1).contiguous()
        h = self.input_proj(h) # (B, channels, N, T)

        # Multi-scale skip aggregation across all dilation stages
        skip_total = 0
        for block in self.blocks:
            h, skip = block(h, adj_adp)
            skip_total = skip_total + skip

        # Final projection head: (B, out_steps, num_nodes, 1) -> (B, out_steps, num_nodes)
        delta = self.head(skip_total).squeeze(-1)

        # Residual forecasting: Base velocity + Predicted delta
        if self.use_residual_head:
            out = last_speed.unsqueeze(1) * self.residual_scale + delta # (B, out_steps, N)
        else:
            out = delta
        return out
