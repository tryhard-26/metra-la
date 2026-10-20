#!/usr/bin/env python3
"""
scripts/ablation_study.py
Systematic component ablation study suite across 4 deep spatio-temporal architectures
on the METR-LA traffic forecasting benchmark.

Evaluates isolated architectural contributions:
1. Temporal Transformer: Spatial Attention Diffusion, Positional Encodings, Residual Head.
2. Spatio-Temporal GNN: Chebyshev polynomial filter order K in {1, 2, 3}, Adaptive Graph Diffusion.
3. Dilated 1D-CNN: Causal dilation schedules d in {[1,1,1,1], [1,2,4], [1,2,4,8]}, Adaptive Graph.
4. Multi-Layer LSTM: Layer depth (1 vs 2 layers), Spatial Node Embeddings, Layer Normalization.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tabulate import tabulate
from tqdm import tqdm

from src.dataset import build_traffic_dataloaders, StandardScaler
from src.metrics import masked_mae_torch, evaluate_horizon_metrics
from src.models import TemporalTransformer, Dilated1DCNN, MultiLayerLSTM, SpatioTemporalGNN

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ablation_study")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    return logger


def select_device(preferred: str, logger: logging.Logger) -> torch.device:
    pref = preferred.lower().strip()
    if pref == "cuda" and torch.cuda.is_available():
        logger.info(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    elif pref == "mps" and torch.backends.mps.is_available():
        logger.info("Using device: mps (Apple Silicon Metal Performance Shaders)")
        return torch.device("mps")
    elif pref == "auto":
        if torch.cuda.is_available():
            logger.info(f"Auto-detected device: cuda ({torch.cuda.get_device_name(0)})")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Auto-detected device: mps (Apple Silicon)")
            return torch.device("mps")
    logger.info("Using device: cpu")
    return torch.device("cpu")


def find_latest_checkpoint(pattern: str) -> Optional[Path]:
    output_dir = ROOT_DIR / "output"
    runs = sorted(list(output_dir.glob(pattern)), reverse=True)
    for run in runs:
        ckpt = run / "best_model.pt"
        if ckpt.exists():
            return ckpt
    return None


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    logger: logging.Logger,
    variant_name: str
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        preds = model(x_batch)

        preds_np = preds.detach().cpu().numpy()
        preds_rescaled = scaler.inverse_transform(preds_np)
        y_rescaled = scaler.inverse_transform(y_batch.numpy())

        all_preds.append(preds_rescaled)
        all_targets.append(y_rescaled)

    preds_all = np.concatenate(all_preds, axis=0)
    targets_all = np.concatenate(all_targets, axis=0)

    results = evaluate_horizon_metrics(preds_all, targets_all, horizons=[3, 6, 12])
    logger.info(
        f"Evaluated {variant_name:<38} -> MAE 15m: {results['horizon_3_15min']['mae']:.2f} | "
        f"30m: {results['horizon_6_30min']['mae']:.2f} | 60m: {results['horizon_12_60min']['mae']:.2f} | "
        f"Overall: {results['overall']['mae']:.2f}"
    )
    return results


def train_ablation_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    epochs: int,
    lr: float,
    logger: logging.Logger,
    name: str,
    max_train_batches: Optional[int] = None
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    logger.info(f"Training ablation variant '{name}' for {epochs} epochs...")
    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for b_idx, (x_batch, y_batch) in enumerate(train_loader):
            if max_train_batches and b_idx >= max_train_batches:
                break
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)

            preds_rescaled = scaler.inverse_transform(preds)
            y_rescaled = scaler.inverse_transform(y_batch)

            loss = masked_mae_torch(preds_rescaled, y_rescaled, null_val=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1

        train_loss = total_loss / max(count, 1)

        # Validation
        model.eval()
        val_loss_tot = 0.0
        val_count = 0
        with torch.no_grad():
            for b_idx, (x_batch, y_batch) in enumerate(val_loader):
                if max_train_batches and b_idx >= (max_train_batches // 4):
                    break
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                preds_rescaled = scaler.inverse_transform(preds)
                y_rescaled = scaler.inverse_transform(y_batch)
                v_loss = masked_mae_torch(preds_rescaled, y_rescaled, null_val=0.0)
                val_loss_tot += v_loss.item()
                val_count += 1

        val_loss = val_loss_tot / max(val_count, 1)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(f"  [{name}] Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    return model


def plot_ablation_suite(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Temporal Transformer Component Ablations (Degradation % vs Full)
    ax_a = axes[0, 0]
    tf_df = df[df["model_family"] == "Temporal Transformer"].copy()
    base_mae = tf_df[tf_df["is_baseline"]]["overall_mae"].values[0]
    tf_df["pct_degradation"] = ((tf_df["overall_mae"] - base_mae) / base_mae) * 100.0

    colors_tf = ["#2b5c8f" if b else "#d95f02" for b in tf_df["is_baseline"]]
    bars_a = ax_a.barh(tf_df["ablation_variant"], tf_df["pct_degradation"], color=colors_tf, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax_a.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax_a.set_xlabel("Error Degradation Relative to Baseline (+% MAE)")
    ax_a.set_title("A: Temporal Transformer Component Ablation", fontweight="bold")
    for bar in bars_a:
        w = bar.get_width()
        val_str = f"+{w:.2f}%" if w > 0.01 else "Baseline (0%)"
        ax_a.annotate(val_str, (max(w, 0) + 0.3, bar.get_y() + bar.get_height() / 2),
                      va="center", ha="left", fontsize=9, fontweight="bold")
    ax_a.set_xlim(-0.5, max(tf_df["pct_degradation"].max() + 3.0, 5.0))

    # Panel B: Spatio-Temporal GNN Chebyshev Polynomial Order K (Multi-Horizon Progression)
    ax_b = axes[0, 1]
    gnn_df = df[df["model_family"] == "Spatio-Temporal GNN"].copy()
    horizons = ["15m", "30m", "60m"]
    x_indices = np.arange(len(gnn_df))
    bar_width = 0.25

    colors_h = ["#6baed6", "#3182bd", "#08519c"]
    for idx, (h_key, h_col) in enumerate([("h15_mae", "15m"), ("h30_mae", "30m"), ("h60_mae", "60m")]):
        ax_b.bar(x_indices + (idx - 1) * bar_width, gnn_df[h_key], width=bar_width,
                 label=f"{h_col} Horizon", color=colors_h[idx], edgecolor="black", linewidth=0.7)

    ax_b.set_xticks(x_indices)
    ax_b.set_xticklabels(gnn_df["ablation_variant"], rotation=15, ha="right", fontsize=9)
    ax_b.set_ylabel("Masked MAE (mph)")
    ax_b.set_title("B: GNN Spectral ChebNet Order K & Graph Ablation", fontweight="bold")
    ax_b.legend(loc="upper left", framealpha=0.9)
    ax_b.set_ylim(2.0, max(gnn_df["h60_mae"].max() + 0.8, 4.5))

    # Panel C: Dilated 1D-CNN Receptive Field & Dilation Schedule Comparison
    ax_c = axes[1, 0]
    cnn_df = df[df["model_family"] == "Dilated 1D-CNN"].copy()
    x_c = np.arange(len(cnn_df))
    ax_c.bar(x_c - 0.15, cnn_df["overall_mae"], width=0.3, label="Overall MAE", color="#fc8d59", edgecolor="black", linewidth=0.8)
    ax_c.bar(x_c + 0.15, cnn_df["overall_rmse"], width=0.3, label="Overall RMSE", color="#b30000", edgecolor="black", linewidth=0.8)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(cnn_df["ablation_variant"], rotation=15, ha="right", fontsize=9)
    ax_c.set_ylabel("Error Metric (mph)")
    ax_c.set_title("C: Dilated 1D-CNN Dilation Schedule & Adaptive Graph", fontweight="bold")
    ax_c.legend(loc="upper left", framealpha=0.9)

    for i, r in cnn_df.reset_index(drop=True).iterrows():
        ax_c.annotate(f"{r['overall_mae']:.2f}", (i - 0.15, r['overall_mae'] + 0.1), ha="center", fontsize=9)
        ax_c.annotate(f"{r['overall_rmse']:.2f}", (i + 0.15, r['overall_rmse'] + 0.1), ha="center", fontsize=9)

    # Panel D: Architectural Feature Sensitivity Matrix (Across all 4 paradigms)
    ax_d = axes[1, 1]
    palette = {"Temporal Transformer": "#2b5c8f", "Spatio-Temporal GNN": "#238b45",
               "Dilated 1D-CNN": "#d95f02", "Multi-Layer LSTM": "#7570b3"}

    summary_records = []
    for model_name, sub in df.groupby("model_family"):
        base_val = sub[sub["is_baseline"]]["overall_mae"].values[0]
        max_degrade = ((sub["overall_mae"].max() - base_val) / base_val) * 100.0
        min_degrade = ((sub["overall_mae"].min() - base_val) / base_val) * 100.0
        avg_mae = sub["overall_mae"].mean()
        summary_records.append({
            "model_family": model_name,
            "baseline_mae": base_val,
            "max_degradation_pct": max_degrade,
            "avg_ablation_mae": avg_mae
        })
    sum_df = pd.DataFrame(summary_records)

    x_d = np.arange(len(sum_df))
    bars_d = ax_d.bar(x_d, sum_df["max_degradation_pct"],
                     color=[palette.get(m, "#333333") for m in sum_df["model_family"]],
                     edgecolor="black", linewidth=0.8, alpha=0.85)

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(sum_df["model_family"], rotation=15, ha="right", fontsize=9)
    ax_d.set_ylabel("Maximum Component Sensitivity (+% Degradation)")
    ax_d.set_title("D: Component Removal Sensitivity by Paradigm", fontweight="bold")
    for bar, r in zip(bars_d, sum_df.itertuples()):
        ax_d.annotate(f"+{r.max_degradation_pct:.1f}%",
                      (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3),
                      ha="center", fontsize=10, fontweight="bold")
    ax_d.set_ylim(0, max(sum_df["max_degradation_pct"].max() + 3.0, 10.0))

    fig.suptitle("Comprehensive Architectural Ablation Study (METR-LA Highway Network)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_ablation_study(args: argparse.Namespace) -> None:
    logger = setup_logger()
    device = select_device(args.device, logger)

    logger.info("============================================================")
    logger.info("STARTING SPATIO-TEMPORAL ARCHITECTURAL ABLATION STUDY")
    logger.info("============================================================")

    data_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader, scaler, sensor_ids = build_traffic_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        in_steps=12,
        out_steps=12,
        num_workers=0
    )

    ablation_records = []

    # -------------------------------------------------------------
    # 1. Temporal Transformer Ablation (Adriteyo Das)
    # -------------------------------------------------------------
    logger.info("--- Evaluating Temporal Transformer Ablation Family ---")
    tf_ckpt = find_latest_checkpoint("run-temporal_transformer-*")
    tf_base = TemporalTransformer(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        d_model=64, nhead=4, num_layers=3, d_ff=128, dropout=0.1,
        use_spatial_diffusion=True, use_pos_encoding=True, use_residual_head=True
    ).to(device)

    if tf_ckpt and tf_ckpt.exists():
        logger.info(f"Loading base Transformer checkpoint from {tf_ckpt}")
        ckpt = torch.load(tf_ckpt, map_location=device)
        tf_base.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.info("Training base Transformer...")
        tf_base = train_ablation_model(tf_base, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "Transformer Base", args.max_batches)

    res_tf_base = evaluate_model(tf_base, test_loader, scaler, device, logger, "Transformer: Full Base")
    ablation_records.append({
        "model_family": "Temporal Transformer",
        "author": "Adriteyo Das (230953244)",
        "ablation_variant": "Full Model (Base)",
        "is_baseline": True,
        "h15_mae": res_tf_base["horizon_3_15min"]["mae"],
        "h30_mae": res_tf_base["horizon_6_30min"]["mae"],
        "h60_mae": res_tf_base["horizon_12_60min"]["mae"],
        "overall_mae": res_tf_base["overall"]["mae"],
        "overall_rmse": res_tf_base["overall"]["rmse"],
        "overall_mape": res_tf_base["overall"]["mape"],
        "overall_r2": res_tf_base["overall"]["r2"],
        "delta_mae": 0.0
    })

    # Variant A1: w/o Spatial Diffusion
    tf_no_spatial = TemporalTransformer(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        d_model=64, nhead=4, num_layers=3, d_ff=128, dropout=0.1,
        use_spatial_diffusion=False, use_pos_encoding=True, use_residual_head=True
    )
    tf_no_spatial = train_ablation_model(tf_no_spatial, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "Transformer w/o Spatial Diff", args.max_batches)
    res_tf_no_spatial = evaluate_model(tf_no_spatial, test_loader, scaler, device, logger, "Transformer: w/o Spatial Diffusion")
    ablation_records.append({
        "model_family": "Temporal Transformer",
        "author": "Adriteyo Das (230953244)",
        "ablation_variant": "w/o Spatial Diffusion",
        "is_baseline": False,
        "h15_mae": res_tf_no_spatial["horizon_3_15min"]["mae"],
        "h30_mae": res_tf_no_spatial["horizon_6_30min"]["mae"],
        "h60_mae": res_tf_no_spatial["horizon_12_60min"]["mae"],
        "overall_mae": res_tf_no_spatial["overall"]["mae"],
        "overall_rmse": res_tf_no_spatial["overall"]["rmse"],
        "overall_mape": res_tf_no_spatial["overall"]["mape"],
        "overall_r2": res_tf_no_spatial["overall"]["r2"],
        "delta_mae": res_tf_no_spatial["overall"]["mae"] - res_tf_base["overall"]["mae"]
    })

    # Variant A2: w/o Positional Encoding
    tf_no_pe = TemporalTransformer(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        d_model=64, nhead=4, num_layers=3, d_ff=128, dropout=0.1,
        use_spatial_diffusion=True, use_pos_encoding=False, use_residual_head=True
    )
    tf_no_pe = train_ablation_model(tf_no_pe, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "Transformer w/o PE", args.max_batches)
    res_tf_no_pe = evaluate_model(tf_no_pe, test_loader, scaler, device, logger, "Transformer: w/o Positional Encoding")
    ablation_records.append({
        "model_family": "Temporal Transformer",
        "author": "Adriteyo Das (230953244)",
        "ablation_variant": "w/o Positional Encoding",
        "is_baseline": False,
        "h15_mae": res_tf_no_pe["horizon_3_15min"]["mae"],
        "h30_mae": res_tf_no_pe["horizon_6_30min"]["mae"],
        "h60_mae": res_tf_no_pe["horizon_12_60min"]["mae"],
        "overall_mae": res_tf_no_pe["overall"]["mae"],
        "overall_rmse": res_tf_no_pe["overall"]["rmse"],
        "overall_mape": res_tf_no_pe["overall"]["mape"],
        "overall_r2": res_tf_no_pe["overall"]["r2"],
        "delta_mae": res_tf_no_pe["overall"]["mae"] - res_tf_base["overall"]["mae"]
    })

    # Variant A3: w/o Residual Velocity Head
    tf_no_res = TemporalTransformer(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        d_model=64, nhead=4, num_layers=3, d_ff=128, dropout=0.1,
        use_spatial_diffusion=True, use_pos_encoding=True, use_residual_head=False
    )
    tf_no_res = train_ablation_model(tf_no_res, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "Transformer w/o Residual Skip", args.max_batches)
    res_tf_no_res = evaluate_model(tf_no_res, test_loader, scaler, device, logger, "Transformer: w/o Residual Skip Head")
    ablation_records.append({
        "model_family": "Temporal Transformer",
        "author": "Adriteyo Das (230953244)",
        "ablation_variant": "w/o Residual Skip Head",
        "is_baseline": False,
        "h15_mae": res_tf_no_res["horizon_3_15min"]["mae"],
        "h30_mae": res_tf_no_res["horizon_6_30min"]["mae"],
        "h60_mae": res_tf_no_res["horizon_12_60min"]["mae"],
        "overall_mae": res_tf_no_res["overall"]["mae"],
        "overall_rmse": res_tf_no_res["overall"]["rmse"],
        "overall_mape": res_tf_no_res["overall"]["mape"],
        "overall_r2": res_tf_no_res["overall"]["r2"],
        "delta_mae": res_tf_no_res["overall"]["mae"] - res_tf_base["overall"]["mae"]
    })

    # -------------------------------------------------------------
    # 2. Spatio-Temporal GNN Ablation (Varanasi Naga Akhil)
    # -------------------------------------------------------------
    logger.info("--- Evaluating Spatio-Temporal GNN Ablation Family ---")
    gnn_ckpt = find_latest_checkpoint("run-gnn-*")
    gnn_num_blocks = 3
    if gnn_ckpt:
        cfg_p = gnn_ckpt.parent / "run_config.json"
        if cfg_p.exists():
            with open(cfg_p) as f:
                gnn_num_blocks = json.load(f).get("num_layers", 3)

    gnn_base = SpatioTemporalGNN(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        hidden_channels=64, k=3, num_blocks=gnn_num_blocks, dropout=0.1,
        use_adaptive_graph=True, use_residual_head=True
    ).to(device)

    if gnn_ckpt and gnn_ckpt.exists():
        logger.info(f"Loading base GNN checkpoint from {gnn_ckpt}")
        ckpt = torch.load(gnn_ckpt, map_location=device)
        gnn_base.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.info("Training base GNN...")
        gnn_base = train_ablation_model(gnn_base, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "GNN Base K=3", args.max_batches)

    res_gnn_base = evaluate_model(gnn_base, test_loader, scaler, device, logger, "STGCN: Full Base (K=3)")
    ablation_records.append({
        "model_family": "Spatio-Temporal GNN",
        "author": "Varanasi Naga Akhil (230953496)",
        "ablation_variant": "ChebNet K=3 (Base)",
        "is_baseline": True,
        "h15_mae": res_gnn_base["horizon_3_15min"]["mae"],
        "h30_mae": res_gnn_base["horizon_6_30min"]["mae"],
        "h60_mae": res_gnn_base["horizon_12_60min"]["mae"],
        "overall_mae": res_gnn_base["overall"]["mae"],
        "overall_rmse": res_gnn_base["overall"]["rmse"],
        "overall_mape": res_gnn_base["overall"]["mape"],
        "overall_r2": res_gnn_base["overall"]["r2"],
        "delta_mae": 0.0
    })

    # Variant B1: ChebNet K=1 (1-hop GCN)
    gnn_k1 = SpatioTemporalGNN(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        hidden_channels=64, k=1, num_blocks=gnn_num_blocks, dropout=0.1,
        use_adaptive_graph=True, use_residual_head=True
    )
    gnn_k1 = train_ablation_model(gnn_k1, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "GNN K=1", args.max_batches)
    res_gnn_k1 = evaluate_model(gnn_k1, test_loader, scaler, device, logger, "STGCN: ChebNet K=1")
    ablation_records.append({
        "model_family": "Spatio-Temporal GNN",
        "author": "Varanasi Naga Akhil (230953496)",
        "ablation_variant": "ChebNet K=1 (1-Hop)",
        "is_baseline": False,
        "h15_mae": res_gnn_k1["horizon_3_15min"]["mae"],
        "h30_mae": res_gnn_k1["horizon_6_30min"]["mae"],
        "h60_mae": res_gnn_k1["horizon_12_60min"]["mae"],
        "overall_mae": res_gnn_k1["overall"]["mae"],
        "overall_rmse": res_gnn_k1["overall"]["rmse"],
        "overall_mape": res_gnn_k1["overall"]["mape"],
        "overall_r2": res_gnn_k1["overall"]["r2"],
        "delta_mae": res_gnn_k1["overall"]["mae"] - res_gnn_base["overall"]["mae"]
    })

    # Variant B2: ChebNet K=2 (2-hop)
    gnn_k2 = SpatioTemporalGNN(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        hidden_channels=64, k=2, num_blocks=gnn_num_blocks, dropout=0.1,
        use_adaptive_graph=True, use_residual_head=True
    )
    gnn_k2 = train_ablation_model(gnn_k2, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "GNN K=2", args.max_batches)
    res_gnn_k2 = evaluate_model(gnn_k2, test_loader, scaler, device, logger, "STGCN: ChebNet K=2")
    ablation_records.append({
        "model_family": "Spatio-Temporal GNN",
        "author": "Varanasi Naga Akhil (230953496)",
        "ablation_variant": "ChebNet K=2 (2-Hop)",
        "is_baseline": False,
        "h15_mae": res_gnn_k2["horizon_3_15min"]["mae"],
        "h30_mae": res_gnn_k2["horizon_6_30min"]["mae"],
        "h60_mae": res_gnn_k2["horizon_12_60min"]["mae"],
        "overall_mae": res_gnn_k2["overall"]["mae"],
        "overall_rmse": res_gnn_k2["overall"]["rmse"],
        "overall_mape": res_gnn_k2["overall"]["mape"],
        "overall_r2": res_gnn_k2["overall"]["r2"],
        "delta_mae": res_gnn_k2["overall"]["mae"] - res_gnn_base["overall"]["mae"]
    })

    # Variant B3: w/o Adaptive Dynamic Graph (Static Physical Distance Only)
    gnn_no_adp = SpatioTemporalGNN(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        hidden_channels=64, k=3, num_blocks=gnn_num_blocks, dropout=0.1,
        use_adaptive_graph=False, use_residual_head=True
    )
    gnn_no_adp = train_ablation_model(gnn_no_adp, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "GNN Static Physical Only", args.max_batches)
    res_gnn_no_adp = evaluate_model(gnn_no_adp, test_loader, scaler, device, logger, "STGCN: w/o Adaptive Dynamic Graph")
    ablation_records.append({
        "model_family": "Spatio-Temporal GNN",
        "author": "Varanasi Naga Akhil (230953496)",
        "ablation_variant": "Static Graph Only",
        "is_baseline": False,
        "h15_mae": res_gnn_no_adp["horizon_3_15min"]["mae"],
        "h30_mae": res_gnn_no_adp["horizon_6_30min"]["mae"],
        "h60_mae": res_gnn_no_adp["horizon_12_60min"]["mae"],
        "overall_mae": res_gnn_no_adp["overall"]["mae"],
        "overall_rmse": res_gnn_no_adp["overall"]["rmse"],
        "overall_mape": res_gnn_no_adp["overall"]["mape"],
        "overall_r2": res_gnn_no_adp["overall"]["r2"],
        "delta_mae": res_gnn_no_adp["overall"]["mae"] - res_gnn_base["overall"]["mae"]
    })

    # -------------------------------------------------------------
    # 3. Dilated 1D-CNN Ablation (Jayesh Agarwal)
    # -------------------------------------------------------------
    logger.info("--- Evaluating Dilated 1D-CNN Ablation Family ---")
    cnn_ckpt = find_latest_checkpoint("run-cnn-*")
    cnn_base = Dilated1DCNN(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        channels=64, dilations=[1, 2, 4, 8], dropout=0.1,
        use_adaptive_graph=True, use_residual_head=True
    ).to(device)

    if cnn_ckpt and cnn_ckpt.exists():
        logger.info(f"Loading base CNN checkpoint from {cnn_ckpt}")
        ckpt = torch.load(cnn_ckpt, map_location=device)
        cnn_base.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.info("Training base CNN...")
        cnn_base = train_ablation_model(cnn_base, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "CNN Base", args.max_batches)

    res_cnn_base = evaluate_model(cnn_base, test_loader, scaler, device, logger, "CNN: Full Base (d=[1,2,4,8])")
    ablation_records.append({
        "model_family": "Dilated 1D-CNN",
        "author": "Jayesh Agarwal (230953348)",
        "ablation_variant": "Dilations [1,2,4,8] (Base)",
        "is_baseline": True,
        "h15_mae": res_cnn_base["horizon_3_15min"]["mae"],
        "h30_mae": res_cnn_base["horizon_6_30min"]["mae"],
        "h60_mae": res_cnn_base["horizon_12_60min"]["mae"],
        "overall_mae": res_cnn_base["overall"]["mae"],
        "overall_rmse": res_cnn_base["overall"]["rmse"],
        "overall_mape": res_cnn_base["overall"]["mape"],
        "overall_r2": res_cnn_base["overall"]["r2"],
        "delta_mae": 0.0
    })

    # Variant C1: No Dilation (d=[1,1,1,1])
    cnn_no_dil = Dilated1DCNN(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        channels=64, dilations=[1, 1, 1, 1], dropout=0.1,
        use_adaptive_graph=True, use_residual_head=True
    )
    cnn_no_dil = train_ablation_model(cnn_no_dil, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "CNN No Dilation", args.max_batches)
    res_cnn_no_dil = evaluate_model(cnn_no_dil, test_loader, scaler, device, logger, "CNN: No Dilation [1,1,1,1]")
    ablation_records.append({
        "model_family": "Dilated 1D-CNN",
        "author": "Jayesh Agarwal (230953348)",
        "ablation_variant": "No Dilation [1,1,1,1]",
        "is_baseline": False,
        "h15_mae": res_cnn_no_dil["horizon_3_15min"]["mae"],
        "h30_mae": res_cnn_no_dil["horizon_6_30min"]["mae"],
        "h60_mae": res_cnn_no_dil["horizon_12_60min"]["mae"],
        "overall_mae": res_cnn_no_dil["overall"]["mae"],
        "overall_rmse": res_cnn_no_dil["overall"]["rmse"],
        "overall_mape": res_cnn_no_dil["overall"]["mape"],
        "overall_r2": res_cnn_no_dil["overall"]["r2"],
        "delta_mae": res_cnn_no_dil["overall"]["mae"] - res_cnn_base["overall"]["mae"]
    })

    # Variant C2: Shallow Dilation (d=[1,2,4])
    cnn_shallow = Dilated1DCNN(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        channels=64, dilations=[1, 2, 4], dropout=0.1,
        use_adaptive_graph=True, use_residual_head=True
    )
    cnn_shallow = train_ablation_model(cnn_shallow, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "CNN Shallow Dilation", args.max_batches)
    res_cnn_shallow = evaluate_model(cnn_shallow, test_loader, scaler, device, logger, "CNN: Shallow Dilation [1,2,4]")
    ablation_records.append({
        "model_family": "Dilated 1D-CNN",
        "author": "Jayesh Agarwal (230953348)",
        "ablation_variant": "Shallow Dilation [1,2,4]",
        "is_baseline": False,
        "h15_mae": res_cnn_shallow["horizon_3_15min"]["mae"],
        "h30_mae": res_cnn_shallow["horizon_6_30min"]["mae"],
        "h60_mae": res_cnn_shallow["horizon_12_60min"]["mae"],
        "overall_mae": res_cnn_shallow["overall"]["mae"],
        "overall_rmse": res_cnn_shallow["overall"]["rmse"],
        "overall_mape": res_cnn_shallow["overall"]["mape"],
        "overall_r2": res_cnn_shallow["overall"]["r2"],
        "delta_mae": res_cnn_shallow["overall"]["mae"] - res_cnn_base["overall"]["mae"]
    })

    # Variant C3: w/o Adaptive Graph (Pure 1D Temporal CNN)
    cnn_no_graph = Dilated1DCNN(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        channels=64, dilations=[1, 2, 4, 8], dropout=0.1,
        use_adaptive_graph=False, use_residual_head=True
    )
    cnn_no_graph = train_ablation_model(cnn_no_graph, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "CNN w/o Graph", args.max_batches)
    res_cnn_no_graph = evaluate_model(cnn_no_graph, test_loader, scaler, device, logger, "CNN: w/o Adaptive Graph")
    ablation_records.append({
        "model_family": "Dilated 1D-CNN",
        "author": "Jayesh Agarwal (230953348)",
        "ablation_variant": "w/o Adaptive Graph",
        "is_baseline": False,
        "h15_mae": res_cnn_no_graph["horizon_3_15min"]["mae"],
        "h30_mae": res_cnn_no_graph["horizon_6_30min"]["mae"],
        "h60_mae": res_cnn_no_graph["horizon_12_60min"]["mae"],
        "overall_mae": res_cnn_no_graph["overall"]["mae"],
        "overall_rmse": res_cnn_no_graph["overall"]["rmse"],
        "overall_mape": res_cnn_no_graph["overall"]["mape"],
        "overall_r2": res_cnn_no_graph["overall"]["r2"],
        "delta_mae": res_cnn_no_graph["overall"]["mae"] - res_cnn_base["overall"]["mae"]
    })

    # -------------------------------------------------------------
    # 4. Multi-Layer LSTM Ablation (Mihika Bardhan)
    # -------------------------------------------------------------
    logger.info("--- Evaluating Multi-Layer LSTM Ablation Family ---")
    lstm_num_layers = 3
    lstm_base = MultiLayerLSTM(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        hidden_dim=64, num_layers=lstm_num_layers, dropout=0.1,
        use_node_embedding=True, use_layernorm=True
    ).to(device)

    loaded_lstm = False
    output_dir = ROOT_DIR / "output"
    for r in sorted(list(output_dir.glob("run-lstm-*")), reverse=True):
        ckpt_candidate = r / "best_model.pt"
        if ckpt_candidate.exists():
            try:
                ckpt = torch.load(ckpt_candidate, map_location=device)
                lstm_base.load_state_dict(ckpt["model_state_dict"])
                logger.info(f"Loaded base LSTM checkpoint from {ckpt_candidate}")
                loaded_lstm = True
                break
            except Exception as e:
                logger.info(f"Checkpoint {ckpt_candidate.parent.name} incompatible ({e}), checking older run...")
                continue

    if not loaded_lstm:
        logger.info("Training base LSTM...")
        lstm_base = train_ablation_model(lstm_base, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "LSTM Base", args.max_batches)

    res_lstm_base = evaluate_model(lstm_base, test_loader, scaler, device, logger, f"LSTM: Full Base ({lstm_num_layers}-Layer)")
    ablation_records.append({
        "model_family": "Multi-Layer LSTM",
        "author": "Mihika Bardhan (230953122)",
        "ablation_variant": f"{lstm_num_layers}-Layer + Embeddings (Base)",
        "is_baseline": True,
        "h15_mae": res_lstm_base["horizon_3_15min"]["mae"],
        "h30_mae": res_lstm_base["horizon_6_30min"]["mae"],
        "h60_mae": res_lstm_base["horizon_12_60min"]["mae"],
        "overall_mae": res_lstm_base["overall"]["mae"],
        "overall_rmse": res_lstm_base["overall"]["rmse"],
        "overall_mape": res_lstm_base["overall"]["mape"],
        "overall_r2": res_lstm_base["overall"]["r2"],
        "delta_mae": 0.0
    })

    # Variant D1: 1-Layer w/o Node Embeddings
    lstm_1l_no_emb = MultiLayerLSTM(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        hidden_dim=64, num_layers=1, dropout=0.0,
        use_node_embedding=False, use_layernorm=False
    )
    lstm_1l_no_emb = train_ablation_model(lstm_1l_no_emb, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "LSTM 1-Layer No Emb", args.max_batches)
    res_lstm_1l = evaluate_model(lstm_1l_no_emb, test_loader, scaler, device, logger, "LSTM: 1-Layer w/o Node Embeddings")
    ablation_records.append({
        "model_family": "Multi-Layer LSTM",
        "author": "Mihika Bardhan (230953122)",
        "ablation_variant": "1-Layer w/o Embeddings",
        "is_baseline": False,
        "h15_mae": res_lstm_1l["horizon_3_15min"]["mae"],
        "h30_mae": res_lstm_1l["horizon_6_30min"]["mae"],
        "h60_mae": res_lstm_1l["horizon_12_60min"]["mae"],
        "overall_mae": res_lstm_1l["overall"]["mae"],
        "overall_rmse": res_lstm_1l["overall"]["rmse"],
        "overall_mape": res_lstm_1l["overall"]["mape"],
        "overall_r2": res_lstm_1l["overall"]["r2"],
        "delta_mae": res_lstm_1l["overall"]["mae"] - res_lstm_base["overall"]["mae"]
    })

    # Variant D2: 2-Layer w/o Node Embeddings
    lstm_2l_no_emb = MultiLayerLSTM(
        num_nodes=207, in_features=2, in_steps=12, out_steps=12,
        hidden_dim=64, num_layers=2, dropout=0.1,
        use_node_embedding=False, use_layernorm=True
    )
    lstm_2l_no_emb = train_ablation_model(lstm_2l_no_emb, train_loader, val_loader, scaler, device, args.epochs, 1e-3, logger, "LSTM 2-Layer No Emb", args.max_batches)
    res_lstm_2l_no_emb = evaluate_model(lstm_2l_no_emb, test_loader, scaler, device, logger, "LSTM: 2-Layer w/o Node Embeddings")
    ablation_records.append({
        "model_family": "Multi-Layer LSTM",
        "author": "Mihika Bardhan (230953122)",
        "ablation_variant": "2-Layer w/o Embeddings",
        "is_baseline": False,
        "h15_mae": res_lstm_2l_no_emb["horizon_3_15min"]["mae"],
        "h30_mae": res_lstm_2l_no_emb["horizon_6_30min"]["mae"],
        "h60_mae": res_lstm_2l_no_emb["horizon_12_60min"]["mae"],
        "overall_mae": res_lstm_2l_no_emb["overall"]["mae"],
        "overall_rmse": res_lstm_2l_no_emb["overall"]["rmse"],
        "overall_mape": res_lstm_2l_no_emb["overall"]["mape"],
        "overall_r2": res_lstm_2l_no_emb["overall"]["r2"],
        "delta_mae": res_lstm_2l_no_emb["overall"]["mae"] - res_lstm_base["overall"]["mae"]
    })

    # -------------------------------------------------------------
    # Synthesis & Export
    # -------------------------------------------------------------
    df_results = pd.DataFrame(ablation_records)
    print("\n" + "=" * 80)
    print("MASTER ABLATION STUDY RESULTS SUMMARY")
    print("=" * 80)
    print(tabulate(
        df_results[["model_family", "ablation_variant", "h15_mae", "h30_mae", "h60_mae", "overall_mae", "delta_mae"]],
        headers=["Model Family", "Ablation Variant", "15m MAE", "30m MAE", "60m MAE", "Overall MAE", "Delta MAE"],
        tablefmt="github",
        floatfmt=".3f"
    ))

    # Export CSV & JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_metrics.csv"
    json_path = output_dir / "ablation_study_summary.json"
    assets_fig = ROOT_DIR / "assets" / "06_ablation_study.png"
    output_fig = output_dir / "06_ablation_study.png"

    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(ablation_records, f, indent=4)
    logger.info(f"Saved tabular ablation summary to: {csv_path}")
    logger.info(f"Saved JSON ablation summary to: {json_path}")

    # Render figure
    logger.info(f"Generating 4-panel publication ablation figure: {assets_fig}")
    plot_ablation_suite(df_results, assets_fig)
    if output_fig != assets_fig:
        import shutil
        shutil.copyfile(assets_fig, output_fig)

    logger.info("Ablation study execution and synthesis completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Systematic Architectural Ablation Study on METR-LA")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"],
                        help="Hardware execution device")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per ablation variant")
    parser.add_argument("--max-batches", type=int, default=50, help="Max train batches per epoch for rapid ablation convergence")
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "data"), help="Data directory")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "output"), help="Output directory")

    args = parser.parse_args()
    run_ablation_study(args)


if __name__ == "__main__":
    main()
