from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelEvaluation:
    baseline_rmse: float
    model_rmse: float
    model_mae: float
    model_r2: float
    residuals: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    feature_weights: pd.DataFrame


def train_and_evaluate_lap_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    alpha: float = 1.0,
    seed: int = 42,
) -> ModelEvaluation:
    """Train a baseline and a multivariate Ridge regression model for lap times."""
    valid_compounds = ["SOFT", "MEDIUM", "HARD"]
    model_df = df[df["Compound"].isin(valid_compounds)].dropna(
        subset=["TyreLife", "race_progress", "TrackTemp", "Compound", "lap_time_sec"]
    ).copy()

    y = model_df["lap_time_sec"]

    X_base = model_df[["TyreLife"]]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_base, y, test_size=test_size, random_state=seed
    )
    baseline = Ridge(alpha=alpha).fit(Xb_train, yb_train)
    baseline_pred = baseline.predict(Xb_test)
    baseline_rmse = float(np.sqrt(mean_squared_error(yb_test, baseline_pred)))

    num_cols = ["TyreLife", "race_progress", "TrackTemp"]
    cat_cols = ["Compound"]
    X_full = model_df[num_cols + cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, random_state=seed
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    model_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    model_mae = float(mean_absolute_error(y_test, y_pred))
    model_r2 = float(r2_score(y_test, y_pred))
    residuals = pd.Series(y_test.to_numpy() - y_pred, index=y_test.index, name="residual_sec")

    fitted_prep = pipeline.named_steps["preprocessor"]
    fitted_ridge = pipeline.named_steps["model"]
    ohe = fitted_prep.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names

    weights_df = pd.DataFrame(
        {
            "feature": all_feature_names,
            "coefficient_sec_per_unit": fitted_ridge.coef_,
        }
    ).sort_values(by="coefficient_sec_per_unit", key=abs, ascending=False)

    return ModelEvaluation(
        baseline_rmse=baseline_rmse,
        model_rmse=model_rmse,
        model_mae=model_mae,
        model_r2=model_r2,
        residuals=residuals,
        y_test=y_test,
        y_pred=y_pred,
        feature_weights=weights_df,
    )