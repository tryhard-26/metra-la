import math
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Union


def calculate_scaled_laplacian(adj_matrix: np.ndarray, lambda_max: float = 2.0) -> np.ndarray:
    adj = sp.coo_matrix(adj_matrix)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    laplacian = sp.coo_matrix(laplacian)
    
    if lambda_max is None:
        lambda_max, _ = sp.linalg.eigsh(laplacian, 1, which="LM")
        lambda_max = float(lambda_max[0])
    
    scaled_laplacian = (2.0 / lambda_max) * laplacian - sp.eye(adj.shape[0])
    return scaled_laplacian.tocoo().astype(np.float32).toarray()


def calculate_chebyshev_polynomials(laplacian: np.ndarray, k: int = 3) -> List[torch.Tensor]:
    n = laplacian.shape[0]
    cheb_polynomials = [np.identity(n, dtype=np.float32)]
    if k >= 2:
        cheb_polynomials.append(laplacian.astype(np.float32))
    for i in range(2, k):
        cheb = 2.0 * laplacian.dot(cheb_polynomials[i - 1]) - cheb_polynomials[i - 2]
        cheb_polynomials.append(cheb.astype(np.float32))
    return [torch.tensor(p, dtype=torch.float32) for p in cheb_polynomials]


class ChebGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 3):
        super().__init__()
        self.k = k
        self.weights = nn.Parameter(torch.empty(k, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, cheb_polys: List[torch.Tensor]) -> torch.Tensor:
        # x: (B, T, N, C_in)
        B, T, N, C_in = x.shape
        x_reshaped = x.view(B * T, N, C_in)

        outputs = []
        for i in range(self.k):
            poly = cheb_polys[i].to(x.device) # (N, N)
            rhs = torch.matmul(poly, x_reshaped)
            out_i = torch.matmul(rhs, self.weights[i])
            outputs.append(out_i)

        out = torch.stack(outputs, dim=0).sum(dim=0) + self.bias
        return out.view(B, T, N, -1)


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj_adp: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C_in)
        # adj_adp: (N, N)
        B, T, N, C_in = x.shape
        x_reshaped = x.view(B * T, N, C_in)
        diffused = torch.matmul(adj_adp, x_reshaped) # (B*T, N, C_in)
        out = self.linear(diffused)
        return out.view(B, T, N, -1)


class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=(kernel_size, 1),
            padding=(self.pad, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, N)
        conv_out = self.conv(x)
        p, q = torch.chunk(conv_out, 2, dim=1)
        return p * torch.sigmoid(q)


class STConvBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, k: int = 3, dropout: float = 0.1, use_adaptive_graph: bool = True):
        super().__init__()
        self.use_adaptive_graph = use_adaptive_graph
        self.tconv1 = TemporalGatedConv(in_channels, hidden_channels, kernel_size=3)
        self.gconv = ChebGraphConv(hidden_channels, hidden_channels, k=k)
        if use_adaptive_graph:
            self.adp_gconv = AdaptiveGraphConv(hidden_channels, hidden_channels)
        self.tconv2 = TemporalGatedConv(hidden_channels, out_channels, kernel_size=3)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cheb_polys: List[torch.Tensor], adj_adp: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        res = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 1. Temporal Conv 1
        h = x.permute(0, 3, 1, 2)
        h = self.tconv1(h)
        h = h.permute(0, 2, 3, 1)

        # 2. Dual Spatial Graph Conv (Physical ChebNet + optional Adaptive Dynamic Graph)
        g_phys = self.gconv(h, cheb_polys)
        if self.use_adaptive_graph and adj_adp is not None:
            g_adp = self.adp_gconv(h, adj_adp)
            h = F.relu(g_phys + g_adp)
        else:
            h = F.relu(g_phys)
        h = self.dropout(h)

        # 3. Temporal Conv 2
        h = h.permute(0, 3, 1, 2)
        h = self.tconv2(h)
        h = h.permute(0, 2, 3, 1)

        # Residual + Norm
        out = self.norm(h + res)
        return out


class SpatioTemporalGNN(nn.Module):
    """
    Dual-Graph Spatio-Temporal Graph Neural Network (STGCN + Adaptive Graph Diffusion).
    Fuses Chebyshev spectral graph convolutions on physical highway graph with dynamic adaptive
    graph learning, temporal GLU gated convolutions, and residual velocity forecasting.
    """
    def __init__(
        self,
        num_nodes: int = 207,
        in_features: int = 2,
        in_steps: int = 12,
        out_steps: int = 12,
        hidden_channels: int = 64,
        node_dim: int = 16,
        k: int = 3,
        num_blocks: int = 2,
        dropout: float = 0.1,
        adj_matrix: Optional[np.ndarray] = None,
        use_adaptive_graph: bool = True,
        use_residual_head: bool = True,
        **kwargs
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.k = k
        self.use_adaptive_graph = use_adaptive_graph
        self.use_residual_head = use_residual_head

        # Load or calculate Chebyshev polynomials
        if adj_matrix is None:
            adj_path = Path(__file__).resolve().parent.parent.parent / "data" / "adj_METR-LA.pkl"
            if adj_path.exists():
                with open(adj_path, "rb") as f:
                    u = pickle._Unpickler(f)
                    u.encoding = "latin1"
                    adj_data = u.load()
                adj_matrix = adj_data[2]
            else:
                adj_matrix = np.eye(num_nodes, dtype=np.float32)

        lap = calculate_scaled_laplacian(adj_matrix)
        cheb_polys = calculate_chebyshev_polynomials(lap, k=k)
        for i, p in enumerate(cheb_polys):
            self.register_buffer(f"cheb_poly_{i}", p)

        # Adaptive Node Embeddings for Dynamic Spatial Interactions
        if use_adaptive_graph:
            self.node_vec1 = nn.Parameter(torch.randn(num_nodes, node_dim))
            self.node_vec2 = nn.Parameter(torch.randn(num_nodes, node_dim))

        self.input_proj = nn.Linear(in_features, hidden_channels)

        self.blocks = nn.ModuleList([
            STConvBlock(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                k=k,
                dropout=dropout,
                use_adaptive_graph=use_adaptive_graph
            )
            for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(in_steps * hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, out_steps)
        )

        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def get_cheb_polys(self) -> List[torch.Tensor]:
        return [getattr(self, f"cheb_poly_{i}") for i in range(self.k)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, in_steps, num_nodes, in_features)
        B, T, N, in_feat = x.shape

        # Base velocity from last observed step: (B, N)
        last_speed = x[:, -1, :, 0]

        # Construct adaptive adjacency matrix: (N, N)
        if self.use_adaptive_graph:
            adj_adp = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2.t())), dim=-1)
        else:
            adj_adp = None

        h = self.input_proj(x) # (B, T, N, hidden_channels)
        cheb_polys = self.get_cheb_polys()

        for block in self.blocks:
            h = block(h, cheb_polys, adj_adp)

        # Flatten time and feature dimension: (B, N, T * hidden_channels)
        h = h.permute(0, 2, 1, 3).contiguous().view(B, N, T * h.size(-1))

        # Project to velocity delta: (B, N, out_steps)
        delta = self.head(h)

        # Residual forecasting: Base velocity + Predicted delta
        if self.use_residual_head:
            out = last_speed.unsqueeze(-1) * self.residual_scale + delta # (B, N, out_steps)
        else:
            out = delta

        # Reshape to (B, out_steps, num_nodes)
        out = out.permute(0, 2, 1).contiguous()
        return out
