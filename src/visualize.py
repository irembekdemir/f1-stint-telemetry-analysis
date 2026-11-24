from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.modeling import ModelEvaluation


def generate_portfolio_plots(
    clean_df: pd.DataFrame,
    eval_res: ModelEvaluation,
    output_dir: str = "./outputs",
    title_prefix: str = "2024 Bahrain GP",
) -> list[Path]:
    """Generate clean, self-explanatory figures for the portfolio README."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = [c for c in ["SOFT", "MEDIUM", "HARD"] if c in clean_df["Compound"].unique()]
    data_by_compound = [
        clean_df.loc[clean_df["Compound"] == c, "lap_time_sec"].dropna() for c in order
    ]
    ax.boxplot(data_by_compound, labels=order, patch_artist=True)
    ax.set_title(f"{title_prefix}: Clean Lap Time Distribution by Tyre Compound")
    ax.set_xlabel("Tyre Compound")
    ax.set_ylabel("Lap Time (seconds)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p1 = out_path / "lap_times_by_compound.png"
    fig.savefig(p1)
    plt.close(fig)
    saved_files.append(p1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    stint_summary = (
        clean_df[clean_df["Compound"].isin(order)]
        .groupby(["Compound", "TyreLife"], as_index=False)["lap_time_sec"]
        .agg(["mean", "count"])
        .reset_index()
    )
    stint_summary = stint_summary[stint_summary["count"] >= 3]
    for comp in order:
        sub = stint_summary[stint_summary["Compound"] == comp].sort_values("TyreLife")
        if not sub.empty:
            ax.plot(sub["TyreLife"], sub["mean"], marker="o", linewidth=2, label=comp)

    ax.set_title(f"{title_prefix}: Mean Lap Time vs. Tyre Age (n >= 3 per point)")
    ax.set_xlabel("Tyre Life (laps on set)")
    ax.set_ylabel("Mean Lap Time (seconds)")
    ax.legend(title="Compound")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    p2 = out_path / "tyre_degradation_curve.png"
    fig.savefig(p2)
    plt.close(fig)
    saved_files.append(p2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(eval_res.y_pred, eval_res.residuals, alpha=0.55, s=22)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_title(f"{title_prefix}: Model Residuals vs. Predicted Lap Time")
    ax.set_xlabel("Predicted Lap Time (seconds)")
    ax.set_ylabel("Residual: Actual - Predicted (seconds)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    p3 = out_path / "residuals_vs_predicted.png"
    fig.savefig(p3)
    plt.close(fig)
    saved_files.append(p3)

    return saved_files