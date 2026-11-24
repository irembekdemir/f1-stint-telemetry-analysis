from __future__ import annotations

import os
from pathlib import Path
import fastf1
import pandas as pd


def load_clean_session_laps(
    year: int = 2024,
    grand_prix: str = "Bahrain",
    session_type: str = "R",
    cache_dir: str = "./f1_cache",
    max_lap_ratio: float = 1.07,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load F1 session laps merged with weather data and filter pit/SC outliers."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, grand_prix, session_type)
    session.load(telemetry=False, weather=True, messages=False)

    raw_laps = session.laps.copy()
    weather = session.weather_data.copy()

    raw_laps["lap_time_sec"] = raw_laps["LapTime"].dt.total_seconds()

    laps_sorted = raw_laps.dropna(subset=["Time", "lap_time_sec"]).sort_values("Time")
    weather_sorted = weather.dropna(subset=["Time"]).sort_values("Time")

    merged = pd.merge_asof(
        laps_sorted,
        weather_sorted[["Time", "TrackTemp", "AirTemp", "Humidity", "Rainfall"]],
        on="Time",
        direction="nearest",
    )

    # Pit giris/cikis turlarini, eksik degerleri ve hatali zamanlari cikar
    base_filtered = merged[
        merged["PitInTime"].isna()
        & merged["PitOutTime"].isna()
        & merged["Compound"].notna()
        & merged["TyreLife"].notna()
        & (merged["TyreLife"] > 0)
        & (merged["lap_time_sec"] > 0)
        & merged["TrackTemp"].notna()
    ].copy()

    # Guvenlik araci (SC/VSC), trafik veya olay nedeniyle asiri uzayan turlari ayikla
    median_lap = base_filtered["lap_time_sec"].median()
    clean_laps = base_filtered[
        base_filtered["lap_time_sec"] <= median_lap * max_lap_ratio
    ].copy()

    if "LapNumber" in clean_laps.columns and clean_laps["LapNumber"].max() > 0:
        clean_laps["race_progress"] = clean_laps["LapNumber"] / clean_laps["LapNumber"].max()
    else:
        clean_laps["race_progress"] = 0.0

    clean_laps["TyreLife"] = clean_laps["TyreLife"].astype(float)
    clean_laps["LapNumber"] = clean_laps["LapNumber"].astype(float)

    return raw_laps, clean_laps