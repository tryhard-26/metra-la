import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., Seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class SpatialAttentionDiffusion(nn.Module):
    """
    Fast Vectorized Spatial Attention Diffusion Layer across 207 highway sensors.
    Learns dynamic spatial correlation between highway nodes via low-rank cross-attention.
    """
    def __init__(self, num_nodes: int, d_model: int, node_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.node_vec2 = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.linear_s = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        # Dynamic spatial attention matrix: (N, N)
        adj_spatial = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2.t())), dim=-1)
        
        # Spatial diffusion: (B*T, N, D)
        x_reshaped = x.view(B * T, N, D)
        diffused = torch.matmul(adj_spatial, x_reshaped) # (B*T, N, D)
        diffused = self.linear_s(diffused).view(B, T, N, D)
        
        out = self.norm(x + self.dropout(diffused))
        return out


class SpatialTemporalTransformerBlock(nn.Module):
    def __init__(self, num_nodes: int, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1, use_spatial_diffusion: bool = True):
        super().__init__()
        self.use_spatial_diffusion = use_spatial_diffusion
        # 1. Temporal Multi-Head Attention (over T_in = 12)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(d_model)
        self.dropout_t = nn.Dropout(dropout)

        # 2. Fast Spatial Attention Diffusion (over N = 207)
        if use_spatial_diffusion:
            self.spatial_diffusion = SpatialAttentionDiffusion(num_nodes=num_nodes, d_model=d_model, dropout=dropout)

        # 3. Position-Wise Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D)
        B, T, N, D = x.shape

        # Temporal attention over T
        h_t = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        norm_t = self.norm_t(h_t)
        attn_out, _ = self.temporal_attn(norm_t, norm_t, norm_t)
        h_t = h_t + self.dropout_t(attn_out)
        x = h_t.view(B, N, T, D).permute(0, 2, 1, 3).contiguous() # (B, T, N, D)

        # Fast Spatial Attention Diffusion across highway sensors (optional)
        if self.use_spatial_diffusion:
            x = self.spatial_diffusion(x)

        # FFN
        x = self.norm_ffn(x + self.ffn(x))
        return x


class TemporalTransformer(nn.Module):
    """
    Spatial-Temporal Dual Attention Transformer for Traffic Forecasting.
    Combines temporal multi-head attention, fast spatial attention diffusion across road sensors,
    learnable spatial node embeddings, and residual velocity forecasting.
    """
    def __init__(
        self,
        num_nodes: int = 207,
        in_features: int = 2,
        in_steps: int = 12,
        out_steps: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        use_spatial_diffusion: bool = True,
        use_pos_encoding: bool = True,
        use_residual_head: bool = True,
        **kwargs
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.d_model = d_model
        self.use_spatial_diffusion = use_spatial_diffusion
        self.use_pos_encoding = use_pos_encoding
        self.use_residual_head = use_residual_head

        self.input_proj = nn.Linear(in_features, d_model)
        
        # Spatial Node Embeddings
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, d_model))
        nn.init.xavier_uniform_(self.node_embedding)

        # Continuous Temporal Positional Encoding
        if use_pos_encoding:
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=in_steps + 10)

        # Stacked Spatial-Temporal Attention Blocks
        self.blocks = nn.ModuleList([
            SpatialTemporalTransformerBlock(
                num_nodes=num_nodes,
                d_model=d_model,
                nhead=nhead,
                d_ff=d_ff,
                dropout=dropout,
                use_spatial_diffusion=use_spatial_diffusion
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Forecasting Projection Head
        self.head = nn.Sequential(
            nn.Linear(in_steps * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, out_steps)
        )

        # Direct Highway Residual Velocity Skip
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, T_in, N, F)
        B, T, N, in_feat = x.shape

        # Last observed speed as base velocity: (B, N) -> Feature 0
        last_speed = x[:, -1, :, 0]

        # Project input features: (B, T, N, d_model)
        h = self.input_proj(x)

        # Add learnable spatial node embeddings
        h = h + self.node_embedding.view(1, 1, N, self.d_model)

        # Add temporal positional encodings
        if self.use_pos_encoding:
            h_t = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
            h_t = self.pos_encoder(h_t)
            h = h_t.view(B, N, T, self.d_model).permute(0, 2, 1, 3).contiguous() # (B, T, N, d_model)

        # Pass through Spatial-Temporal Transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # Reshape to (B, N, T * d_model)
        h = h.permute(0, 2, 1, 3).contiguous().view(B, N, T * self.d_model)

        # Predict multi-step velocity changes: (B, N, out_steps)
        delta = self.head(h)

        # Residual forecasting: Base velocity + Predicted delta
        if self.use_residual_head:
            out = last_speed.unsqueeze(-1) * self.residual_scale + delta
        else:
            out = delta

        # Output shape: (B, out_steps, num_nodes)
        out = out.permute(0, 2, 1).contiguous()
        return out
