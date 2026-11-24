from __future__ import annotations

import argparse
from src.data_loader import load_clean_session_laps
from src.inference import compare_two_compounds
from src.modeling import train_and_evaluate_lap_model
from src.visualize import generate_portfolio_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 Stint & Lap Time Data Science Pipeline")
    parser.add_argument("--year", type=int, default=2024, help="Season year (e.g., 2024)")
    parser.add_argument("--gp", type=str, default="Bahrain", help="Grand Prix name or round")
    parser.add_argument("--session", type=str, default="R", help="Session code (R, Q, FP1, etc.)")
    parser.add_argument("--cache-dir", type=str, default="./f1_cache", help="FastF1 cache directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Plot output directory")
    args = parser.parse_args()

    raw_laps, clean_laps = load_clean_session_laps(
        year=args.year,
        grand_prix=args.gp,
        session_type=args.session,
        cache_dir=args.cache_dir,
    )

    print(f"=== {args.year} {args.gp} ({args.session}) Summary ===")
    print(f"Raw laps: {len(raw_laps)} | Clean green-flag non-pit laps: {len(clean_laps)}")

    comp_res = compare_two_compounds(clean_laps, group_a="SOFT", group_b="MEDIUM")
    if comp_res is not None:
        print(
            f"\n[Inference] {comp_res.group_a} (n={comp_res.n_a}, mean={comp_res.mean_a:.3f}s) vs "
            f"{comp_res.group_b} (n={comp_res.n_b}, mean={comp_res.mean_b:.3f}s)"
        )
        print(
            f"Mean diff ({comp_res.group_a} - {comp_res.group_b}): {comp_res.diff_mean:+.3f}s | "
            f"95% Bootstrap CI: [{comp_res.ci_low:+.3f}s, {comp_res.ci_high:+.3f}s] | "
            f"Permutation p-value: {comp_res.p_value:.4f}"
        )
    else:
        print("\n[Inference] Not enough laps for SOFT vs MEDIUM comparison in this session.")

    model_res = train_and_evaluate_lap_model(clean_laps)
    print("\n[Modeling] Lap Time Prediction")
    print(f"Baseline RMSE (TyreLife only): {model_res.baseline_rmse:.3f}s")
    print(f"Ridge Model RMSE:              {model_res.model_rmse:.3f}s")
    print(f"Ridge Model MAE:               {model_res.model_mae:.3f}s")
    print(f"Ridge Model R^2:               {model_res.model_r2:.3f}")
    print(
        f"Residuals -> mean: {model_res.residuals.mean():+.4f}s, "
        f"std: {model_res.residuals.std():.3f}s, max |err|: {model_res.residuals.abs().max():.3f}s"
    )
    print("\nFeature Weights (standardized continuous / one-hot categorical):")
    print(model_res.feature_weights.to_string(index=False))

    plots = generate_portfolio_plots(
        clean_df=clean_laps,
        eval_res=model_res,
        output_dir=args.output_dir,
        title_prefix=f"{args.year} {args.gp} ({args.session})",
    )
    print("\n[Plots Saved]")
    for p in plots:
        print(f" - {p}")


if __name__ == "__main__":
    main()