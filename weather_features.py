"""
Features meteorológicos diarios por zona, a partir de los CSV de NASA POWER
descargados con build_weather_dataset.py (carpeta data/weather/).

NASA POWER (MERRA-2) tiene celdas de 0,5° lat x 0,625° lon (~50 km): varias zonas
de 0,25° comparten la misma celda meteorológica. En zonas de cordillera esto suaviza
mucho la temperatura real.
"""
import re
import numpy as np
import pandas as pd

LAT_STEP, LON_STEP = 0.5, 0.625
MAX_DAYS_SINCE_RAIN = 120

WX_FEATURES = [
    "wx_tmax", "wx_tmax_3d", "wx_tmax_7d",
    "wx_rh", "wx_rh_3d",
    "wx_wind", "wx_wind_3d",
    "wx_precip_7d", "wx_precip_30d", "wx_days_since_rain",
]

_FNAME = re.compile(r"power_([+-][\d.]+)_([+-][\d.]+)\.csv$")


def snap_cell(lat, lon):
    """Centro de la celda NASA POWER que contiene (lat, lon)."""
    return round(round(lat / LAT_STEP) * LAT_STEP, 3), round(round(lon / LON_STEP) * LON_STEP, 3)


def cell_filename(lat, lon):
    return f"power_{lat:+.3f}_{lon:+.3f}.csv"


class WeatherPanel:
    """Matrices (celda x día) alineadas con las fechas de un FirePanel."""

    def __init__(self, panel, wx_dir):
        files = sorted(p for p in wx_dir.glob("power_*.csv") if _FNAME.search(p.name))
        if not files:
            raise FileNotFoundError(f"No hay CSV de NASA POWER en {wx_dir}. Ejecuta build_weather_dataset.py")
        self.cells = {}
        mats = {f: [] for f in WX_FEATURES}
        for k, path in enumerate(files):
            m = _FNAME.search(path.name)
            self.cells[(round(float(m.group(1)), 3), round(float(m.group(2)), 3))] = k
            feats = self._cell_features(pd.read_csv(path, parse_dates=["date"]), panel)
            for f in WX_FEATURES:
                mats[f].append(feats[f])
        self.M = {f: np.vstack(v).astype(np.float32) for f, v in mats.items()}
        # zona -> índice de celda (-1 si no se descargó)
        self.zone_cell = np.array(
            [self.cells.get(snap_cell(a, b), -1) for a, b in zip(panel.lat_c, panel.lon_c)]
        )
        self.coverage = float((self.zone_cell >= 0).mean())

    @staticmethod
    def _cell_features(df, panel):
        s = df.set_index("date").reindex(panel.dates)
        ffill = lambda x: x.ffill(limit=3)
        tmax, rh, wind = ffill(s["tmax"]), ffill(s["rh"]), ffill(s["wind"])
        precip = s["precip"].fillna(0)
        ar = np.arange(panel.T)
        rain_idx = np.maximum.accumulate(np.where(precip.to_numpy() >= 1.0, ar, -10**6))
        return {
            "wx_tmax": tmax.to_numpy(),
            "wx_tmax_3d": tmax.rolling(3, min_periods=1).mean().to_numpy(),
            "wx_tmax_7d": tmax.rolling(7, min_periods=1).mean().to_numpy(),
            "wx_rh": rh.to_numpy(),
            "wx_rh_3d": rh.rolling(3, min_periods=1).mean().to_numpy(),
            "wx_wind": wind.to_numpy(),
            "wx_wind_3d": wind.rolling(3, min_periods=1).max().to_numpy(),
            "wx_precip_7d": precip.rolling(7, min_periods=1).sum().to_numpy(),
            "wx_precip_30d": precip.rolling(30, min_periods=1).sum().to_numpy(),
            "wx_days_since_rain": np.minimum(ar - rain_idx, MAX_DAYS_SINCE_RAIN).astype(float),
        }

    def features(self, zi, di) -> pd.DataFrame:
        zi, di = np.asarray(zi), np.asarray(di)
        c = self.zone_cell[zi]
        ok = c >= 0
        out = {}
        for f in WX_FEATURES:
            v = np.full(len(zi), np.nan, dtype=np.float32)
            v[ok] = self.M[f][c[ok], di[ok]]
            out[f] = v
        return pd.DataFrame(out)[WX_FEATURES]
