#!/usr/bin/env python3
"""
scripts/error_analysis.py
In-depth failure case analysis and regime-specific evaluation for deep spatio-temporal
traffic forecasting architectures on the METR-LA benchmark.

Analyzes prediction errors across diurnal cycles (rush-hour vs. off-peak), identifies
real-world congestion shockwave failure cases, and assesses spatial sensor error variance.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import build_traffic_dataloaders, StandardScaler
from src.metrics import evaluate_horizon_metrics
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
    logger = logging.getLogger("error_analysis")
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
        dev = torch.device("cuda")
        logger.info(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        return dev
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


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[nn.Module, Dict[str, Any], StandardScaler, List[str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    run_config_path = checkpoint_path.parent / "run_config.json"
    if run_config_path.exists():
        with open(run_config_path, "r") as f:
            config = json.load(f)
    else:
        config = checkpoint.get("config", {})

    model_type = config.get("model", "temporal_transformer").lower()
    num_sensors = config.get("num_sensors", 207)
    d_model = config.get("d_model", 64)
    nhead = config.get("nhead", 4)
    num_layers = config.get("num_layers", 3)
    d_ff = config.get("d_ff", 128)
    dropout = config.get("dropout", 0.1)

    if "transformer" in model_type:
        model = TemporalTransformer(
            num_nodes=num_sensors,
            in_features=2,
            in_steps=12,
            out_steps=12,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        )
    elif "cnn" in model_type:
        model = Dilated1DCNN(
            num_nodes=num_sensors,
            in_features=2,
            in_steps=12,
            out_steps=12,
            hidden_dim=d_model,
            dropout=dropout
        )
    elif "gnn" in model_type:
        model = SpatioTemporalGNN(
            num_nodes=num_sensors,
            in_features=2,
            in_steps=12,
            out_steps=12,
            hidden_dim=d_model,
            cheb_k=3,
            dropout=dropout
        )
    elif "lstm" in model_type:
        model = MultiLayerLSTM(
            num_nodes=num_sensors,
            in_features=2,
            in_steps=12,
            out_steps=12,
            hidden_dim=96,
            num_layers=2,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unsupported model type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler_dict = config.get("scaler", {})
    mean_val = scaler_dict.get("mean", 58.46)
    std_val = scaler_dict.get("std", 13.02)
    scaler = StandardScaler(mean=mean_val, std=std_val)

    sensor_ids = [str(i) for i in range(num_sensors)]
    return model, config, scaler, sensor_ids


@torch.no_grad()
def run_detailed_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Executing test-set inference with temporal feature extraction...")
    all_preds = []
    all_targets = []
    all_inputs = []
    all_tods = []

    for x_batch, y_batch in tqdm(dataloader, desc="Inference"):
        tod = x_batch[:, -1, :, 1].numpy()
        x_raw = scaler.inverse_transform(x_batch[..., 0].numpy())

        x_batch = x_batch.to(device)
        preds = model(x_batch)

        preds_np = preds.detach().cpu().numpy()
        preds_rescaled = scaler.inverse_transform(preds_np)
        y_rescaled = scaler.inverse_transform(y_batch.numpy())

        all_preds.append(preds_rescaled)
        all_targets.append(y_rescaled)
        all_inputs.append(x_raw)
        all_tods.append(tod)

    preds_all = np.concatenate(all_preds, axis=0)
    targets_all = np.concatenate(all_targets, axis=0)
    inputs_all = np.concatenate(all_inputs, axis=0)
    tods_all = np.concatenate(all_tods, axis=0)
    return preds_all, targets_all, inputs_all, tods_all


def compute_diurnal_regime_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    tods: np.ndarray,
    logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    logger.info("Analyzing error breakdown across diurnal operating regimes...")
    hours = (tods[:, 0] * 24.0).astype(int)

    regimes = {
        "Morning Peak (07:00-09:00)": (hours >= 7) & (hours < 9),
        "Midday Inter-Peak (11:00-14:00)": (hours >= 11) & (hours < 14),
        "Evening Peak (16:30-18:30)": ((hours >= 16) & (hours < 19)),
        "Night Free-Flow (22:00-05:00)": (hours >= 22) | (hours < 5),
    }

    results = {}
    for regime_name, mask in regimes.items():
        if np.sum(mask) == 0:
            continue
        p_sub = preds[mask]
        t_sub = targets[mask]

        valid = t_sub > 0.0
        diff = np.abs(p_sub - t_sub)
        mae_overall = np.mean(diff[valid])
        rmse_overall = np.sqrt(np.mean((diff ** 2)[valid]))

        v3 = (t_sub[:, 2, :] > 0.0)
        mae_15m = np.mean(diff[:, 2, :][v3])

        v6 = (t_sub[:, 5, :] > 0.0)
        mae_30m = np.mean(diff[:, 5, :][v6])

        v12 = (t_sub[:, 11, :] > 0.0)
        mae_60m = np.mean(diff[:, 11, :][v12])

        results[regime_name] = {
            "samples": int(np.sum(mask)),
            "mae_overall": float(mae_overall),
            "rmse_overall": float(rmse_overall),
            "mae_15m": float(mae_15m),
            "mae_30m": float(mae_30m),
            "mae_60m": float(mae_60m)
        }
        logger.info(f"{regime_name}: MAE 15m={mae_15m:.2f}, 30m={mae_30m:.2f}, 60m={mae_60m:.2f}, Overall={mae_overall:.2f}")

    return results


def find_congestion_shockwave_case(
    inputs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    logger: logging.Logger
) -> Tuple[int, int, Dict[str, Any]]:
    logger.info("Scanning test sequences for severe congestion shockwave failure case...")
    B, T_out, N = targets.shape

    max_drop = -1.0
    best_b = 0
    best_n = 0

    for b in range(B):
        for n in range(N):
            history_end = inputs[b, -1, n]
            if history_end < 50.0:
                continue

            target_min = np.min(targets[b, :, n])
            if target_min <= 0.0:
                continue

            drop = history_end - target_min
            if drop > max_drop:
                max_drop = drop
                best_b = b
                best_n = n

    case_info = {
        "batch_idx": int(best_b),
        "sensor_idx": int(best_n),
        "initial_speed": float(inputs[best_b, -1, best_n]),
        "minimum_target_speed": float(np.min(targets[best_b, :, best_n])),
        "maximum_speed_drop_mph": float(max_drop),
        "history_speed": [float(x) for x in inputs[best_b, :, best_n]],
        "target_speed": [float(y) for y in targets[best_b, :, best_n]],
        "predicted_speed": [float(p) for p in preds[best_b, :, best_n]]
    }
    logger.info(f"Identified shockwave at sequence {best_b}, sensor {best_n}: Drop {case_info['initial_speed']:.1f} -> {case_info['minimum_target_speed']:.1f} mph (Delta {max_drop:.1f} mph)")
    return best_b, best_n, case_info


def compute_spatial_sensor_rankings(
    preds: np.ndarray,
    targets: np.ndarray,
    sensor_ids: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = targets > 0.0
    abs_err = np.where(valid, np.abs(preds - targets), np.nan)
    sensor_mae = np.nanmean(abs_err, axis=(0, 1))
    sensor_rmse = np.sqrt(np.nanmean(np.where(valid, (preds - targets) ** 2, np.nan), axis=(0, 1)))

    df_sensors = pd.DataFrame({
        "sensor_idx": np.arange(len(sensor_mae)),
        "sensor_id": sensor_ids,
        "mae": sensor_mae,
        "rmse": sensor_rmse
    })

    top_worst = df_sensors.sort_values(by="mae", ascending=False).head(8)
    top_best = df_sensors.sort_values(by="mae", ascending=True).head(8)
    return top_worst, top_best


def plot_failure_case_suite(
    diurnal_metrics: Dict[str, Dict[str, float]],
    shockwave_case: Dict[str, Any],
    top_worst: pd.DataFrame,
    top_best: pd.DataFrame,
    hourly_mae_matrix: np.ndarray,
    output_path: Path
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Diurnal Regime Error Comparison
    ax_a = axes[0, 0]
    regimes = list(diurnal_metrics.keys())
    short_names = ["Morning Peak", "Midday", "Evening Peak", "Night Free-Flow"]
    mae_15m = [diurnal_metrics[r]["mae_15m"] for r in regimes]
    mae_30m = [diurnal_metrics[r]["mae_30m"] for r in regimes]
    mae_60m = [diurnal_metrics[r]["mae_60m"] for r in regimes]

    x = np.arange(len(short_names))
    width = 0.25
    ax_a.bar(x - width, mae_15m, width, label="15-min (T=3)", color="#2b5c8f")
    ax_a.bar(x, mae_30m, width, label="30-min (T=6)", color="#d95f02")
    ax_a.bar(x + width, mae_60m, width, label="60-min (T=12)", color="#7570b3")
    ax_a.set_ylabel("Masked MAE (mph)")
    ax_a.set_title("A: Forecast Error across Operating Regimes", fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_names, rotation=12)
    ax_a.legend(loc="upper left")
    ax_a.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Congestion Shockwave Case Study
    ax_b = axes[0, 1]
    hist = shockwave_case["history_speed"]
    targ = shockwave_case["target_speed"]
    pred = shockwave_case["predicted_speed"]

    t_hist = np.arange(-11, 1) * 5
    t_pred = np.arange(1, 13) * 5

    ax_b.plot(t_hist, hist, "o-", color="#333333", label="Historical Observation (1h)", linewidth=2)
    ax_b.plot(t_pred, targ, "s-", color="#d95f02", label="Ground Truth (Actual Drop)", linewidth=2.5)
    ax_b.plot(t_pred, pred, "^--", color="#2b5c8f", label="Model Prediction (Lag)", linewidth=2)
    ax_b.axvline(0, color="gray", linestyle=":", label="Forecast Origin (t=0)")
    ax_b.set_xlabel("Time from Forecast Origin (Minutes)")
    ax_b.set_ylabel("Traffic Speed (mph)")
    ax_b.set_title(rf"B: Congestion Shockwave Case (Sensor {shockwave_case['sensor_idx']}, $\Delta$={shockwave_case['maximum_speed_drop_mph']:.1f} mph)", fontweight="bold")
    ax_b.legend(loc="upper right", framealpha=0.9)
    ax_b.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Top Vulnerable vs Predictable Sensors
    ax_c = axes[1, 0]
    worst_labels = [f"S{r['sensor_idx']}" for _, r in top_worst.iterrows()]
    best_labels = [f"S{r['sensor_idx']}" for _, r in top_best.iterrows()]
    y_pos_worst = np.arange(len(worst_labels))
    y_pos_best = np.arange(len(best_labels)) + len(worst_labels) + 1

    ax_c.barh(y_pos_worst, top_worst["mae"], color="#e41a1c", alpha=0.85, label="High Error (Bottlenecks)")
    ax_c.barh(y_pos_best, top_best["mae"], color="#4daf4a", alpha=0.85, label="Low Error (Free-Flow)")
    ax_c.set_yticks(list(y_pos_worst) + list(y_pos_best))
    ax_c.set_yticklabels(worst_labels + best_labels, fontsize=9)
    ax_c.set_xlabel("Masked MAE (mph)")
    ax_c.set_title("C: Spatial Vulnerability (Top 8 Bottlenecks vs Stable Sensors)", fontweight="bold")
    ax_c.legend(loc="lower right")
    ax_c.grid(True, linestyle="--", alpha=0.5)

    # Panel D: 2D Temporal Error Heatmap (Hour of Day vs Forecast Horizon)
    ax_d = axes[1, 1]
    sns.heatmap(
        hourly_mae_matrix,
        cmap="YlOrRd",
        cbar_kws={"label": "MAE (mph)"},
        ax=ax_d,
        xticklabels=[f"T+{t*5}m" for t in range(1, 13)],
        yticklabels=[f"{h:02d}:00" for h in range(24)]
    )
    ax_d.set_xlabel("Forecasting Horizon")
    ax_d.set_ylabel("Hour of Day")
    ax_d.set_title("D: Spatio-Temporal Error Heatmap (Hour of Day vs Horizon)", fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_error_analysis(args: argparse.Namespace):
    logger = setup_logger()
    device = select_device(args.device, logger)

    checkpoint_path = Path(args.checkpoint)
    model, config, scaler, sensor_ids = load_model_from_checkpoint(checkpoint_path, device, logger)

    logger.info("Building METR-LA DataLoaders...")
    train_loader, val_loader, test_loader, dataset_scaler, dataset_sensor_ids = build_traffic_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        in_steps=12,
        out_steps=12,
        num_workers=0
    )
    if dataset_sensor_ids:
        sensor_ids = dataset_sensor_ids

    preds, targets, inputs, tods = run_detailed_inference(
        model=model,
        dataloader=test_loader,
        scaler=scaler,
        device=device,
        logger=logger
    )

    horizon_results = evaluate_horizon_metrics(preds, targets, horizons=[3, 6, 12])
    diurnal_metrics = compute_diurnal_regime_metrics(preds, targets, tods, logger)
    best_b, best_n, shockwave_case = find_congestion_shockwave_case(inputs, preds, targets, logger)
    top_worst, top_best = compute_spatial_sensor_rankings(preds, targets, sensor_ids)

    # Compute 24h x 12 horizon error matrix
    hours = (tods[:, 0] * 24.0).astype(int)
    hourly_mae_matrix = np.zeros((24, 12))
    for h in range(24):
        mask = (hours == h)
        if np.sum(mask) == 0:
            continue
        valid = targets[mask] > 0.0
        diff = np.abs(preds[mask] - targets[mask])
        for t in range(12):
            v_t = valid[:, t, :]
            hourly_mae_matrix[h, t] = np.mean(diff[:, t, :][v_t]) if np.sum(v_t) > 0 else 0.0

    assets_fig = ROOT_DIR / "assets" / "05_failure_case_analysis.png"
    output_fig = Path(args.output_dir) / "05_failure_case_analysis.png"

    logger.info(f"Rendering publication figure to: {assets_fig}")
    plot_failure_case_suite(
        diurnal_metrics=diurnal_metrics,
        shockwave_case=shockwave_case,
        top_worst=top_worst,
        top_best=top_best,
        hourly_mae_matrix=hourly_mae_matrix,
        output_path=assets_fig
    )

    if output_fig != assets_fig:
        import shutil
        output_fig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets_fig, output_fig)

    analysis_report = {
        "model_analyzed": config.get("model", "temporal_transformer"),
        "checkpoint_used": str(checkpoint_path),
        "horizon_metrics": horizon_results,
        "diurnal_regime_breakdown": diurnal_metrics,
        "shockwave_failure_case": shockwave_case,
        "top_bottleneck_sensors": top_worst.to_dict(orient="records"),
        "top_predictable_sensors": top_best.to_dict(orient="records")
    }

    report_path = Path(args.output_dir) / "failure_case_analysis.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(analysis_report, f, indent=4)

    logger.info(f"Saved analysis summary to: {report_path}")
    logger.info("Failure case and error analysis completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Failure Case and Error Analysis on METR-LA")
    parser.add_argument("--checkpoint", type=str,
                        default=str(ROOT_DIR / "output" / "run-temporal_transformer-20260826_222415" / "best_model.pt"),
                        help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"],
                        help="Hardware execution device")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "data"), help="Data directory")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "output"), help="Output directory")

    args = parser.parse_args()
    run_error_analysis(args)


if __name__ == "__main__":
    main()
