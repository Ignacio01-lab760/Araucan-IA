import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import FirePanel
from .weather_features import WeatherPanel


@lru_cache(maxsize=1)
def get_panel():
    if not C.HOTSPOTS_CSV.exists():
        raise RuntimeError(f"No se encontró {C.HOTSPOTS_CSV}")
    df = pd.read_csv(C.HOTSPOTS_CSV, usecols=["latitude", "longitude", "acq_date", "frp"])
    return FirePanel(df)


@lru_cache(maxsize=1)
def get_wx_panel():
    """Meteorología NASA POWER alineada con el panel; None si no está descargada."""
    try:
        wx = WeatherPanel(get_panel(), C.WX_DIR)
    except FileNotFoundError:
        return None
    return wx if wx.coverage >= 0.9 else None


@lru_cache(maxsize=4)
def _load(path):
    return joblib.load(path) if path.exists() else None


def _meta(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _pick_model(use_weather: bool):
    """Devuelve (final, backtest, meta, wx_panel). Prefiere el modelo con meteorología si existe."""
    if use_weather and C.HISTW_MODEL_PATH.exists() and get_wx_panel() is not None:
        return _load(C.HISTW_MODEL_PATH), _load(C.HISTW_BACKTEST_PATH), _meta(C.HISTW_META_PATH), get_wx_panel()
    if not C.HIST_MODEL_PATH.exists():
        raise RuntimeError("Falta models/risk_model_hist.joblib. Ejecuta: python train_risk_model.py")
    return _load(C.HIST_MODEL_PATH), _load(C.HIST_BACKTEST_PATH), _meta(C.HIST_META_PATH), None


def level_for(p, thresholds):
    if p >= thresholds["alto"]:
        return "alto"
    if p >= thresholds["medio"]:
        return "medio"
    return "bajo"


def risk_map(as_of=None, use_weather=True):
    """Riesgo de detección térmica en cada zona durante las 24 h posteriores a `as_of`."""
    panel = get_panel()
    final, backtest, meta, wx = _pick_model(use_weather)
    cols = meta["features"]
    thresholds = meta.get("metrics", {}).get("thresholds", {"medio": 0.15, "alto": 0.40})

    di = panel.T - 1 if as_of is None else panel.day_index(as_of)
    has_truth = di + 1 < panel.T
    # Fechas desde 2023: modelo entrenado solo con datos hasta 2022 (backtest honesto).
    if backtest is not None and panel.dates[di] >= pd.Timestamp(C.BACKTEST_START) and has_truth:
        model, model_used = backtest, "backtest_entrenado_hasta_2022"
    else:
        model, model_used = final, "final_todo_el_historico"

    zi = np.arange(panel.Z)
    dd = np.full(panel.Z, di)
    X = panel.features(zi, dd)
    Xw = wx.features(zi, dd) if wx is not None else None
    Xm = pd.concat([X, Xw], axis=1)[cols] if wx is not None else X[cols]
    p = model.predict_proba(Xm)[:, 1]
    half = panel.zone_deg / 2

    zones = []
    for z in range(panel.Z):
        row = {
            "zone_id": int(z),
            "latitude": round(float(panel.lat_c[z]), 4),
            "longitude": round(float(panel.lon_c[z]), 4),
            "bounds": [[float(panel.lat_c[z] - half), float(panel.lon_c[z] - half)],
                       [float(panel.lat_c[z] + half), float(panel.lon_c[z] + half)]],
            "risk_probability": round(float(p[z]), 4),
            "risk_level": level_for(float(p[z]), thresholds),
            "detections_1d": int(X["det_1d"].iloc[z]),
            "detections_7d": int(X["det_7d"].iloc[z]),
            "detections_30d": int(X["det_30d"].iloc[z]),
            "days_since_last": int(X["days_since"].iloc[z]),
        }
        if Xw is not None:
            g = lambda k: None if pd.isna(Xw[k].iloc[z]) else round(float(Xw[k].iloc[z]), 1)
            row["weather_obs"] = {"tmax_c": g("wx_tmax"), "rh_pct": g("wx_rh"), "wind_ms": g("wx_wind"),
                                  "precip_7d_mm": g("wx_precip_7d"), "days_since_rain": g("wx_days_since_rain")}
        if has_truth:
            row["actual_next_day"] = int(panel.C[z, di + 1])
        zones.append(row)
    return {
        "as_of": str(panel.dates[di].date()),
        "horizon_hours": 24,
        "data_start": str(panel.start.date()),
        "data_end": str(panel.end.date()),
        "has_truth": bool(has_truth),
        "uses_weather": wx is not None,
        "model_used": model_used,
        "in_sample": bool(has_truth and model_used == "final_todo_el_historico"),
        "thresholds": thresholds,
        "metrics": meta.get("metrics"),
        "zones": zones,
    }
