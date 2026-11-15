"""
Panel zona x día construido SOLO con el histórico de detecciones VIIRS.

Cada zona es una celda de ZONE_DEG grados. Para cada (zona, día) se calculan
features con información disponible hasta ese día (sin mirar el futuro) y la
etiqueta: ¿habrá al menos una detección térmica en esa zona el día siguiente?

Importante: la etiqueta es "detección satelital", no "incendio confirmado".
"""
import numpy as np
import pandas as pd
from scipy import sparse

ZONE_DEG = 0.25
MIN_ZONE_TOTAL = 10   # zonas con menos detecciones históricas se ignoran
MAX_DAYS_SINCE = 120

FEATURES = [
    "det_1d", "det_3d", "det_7d", "det_30d",
    "frp_3d", "frp_7d",
    "nb_3d", "nb_7d",
    "days_since", "clim_month_rate", "rate_all",
    "doy_sin", "doy_cos", "lat", "lon",
]


def _zone_key(lat, lon, zone_deg):
    zi = np.floor(np.asarray(lat) / zone_deg).astype(np.int64)
    zj = np.floor(np.asarray(lon) / zone_deg).astype(np.int64)
    return zi, zj


class FirePanel:
    def __init__(self, df: pd.DataFrame, zone_deg=ZONE_DEG, min_total=MIN_ZONE_TOTAL):
        self.zone_deg = zone_deg
        d = pd.to_datetime(df["acq_date"], errors="coerce")
        ok = d.notna() & df["latitude"].notna() & df["longitude"].notna()
        df = df[ok]
        d = d[ok].dt.normalize()
        frp = df["frp"].fillna(0).to_numpy(dtype=np.float64)

        zi, zj = _zone_key(df["latitude"], df["longitude"], zone_deg)
        key = (zi + 5000) * 100000 + (zj + 5000)
        keys, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        keep_zone = counts >= min_total
        remap = -np.ones(len(keys), dtype=np.int64)
        remap[keep_zone] = np.arange(keep_zone.sum())
        zidx = remap[inv]
        mask = zidx >= 0

        self.zone_keys = keys[keep_zone]
        self.Z = int(keep_zone.sum())
        self.zi = (self.zone_keys // 100000) - 5000
        self.zj = (self.zone_keys % 100000) - 5000
        self.lat_c = (self.zi + 0.5) * zone_deg
        self.lon_c = (self.zj + 0.5) * zone_deg

        self.start = d.min()
        self.end = d.max()
        self.T = int((self.end - self.start).days) + 1
        didx = (d - self.start).dt.days.to_numpy()

        flat = zidx[mask] * self.T + didx[mask]
        size = self.Z * self.T
        self.C = np.bincount(flat, minlength=size).reshape(self.Z, self.T).astype(np.int32)
        self.F = np.bincount(flat, weights=frp[mask], minlength=size).reshape(self.Z, self.T).astype(np.float32)

        # Vecinos (8-conectados) entre zonas seleccionadas
        pos = {(a, b): i for i, (a, b) in enumerate(zip(self.zi, self.zj))}
        rows, cols = [], []
        for i, (a, b) in enumerate(zip(self.zi, self.zj)):
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da == 0 and db == 0:
                        continue
                    j = pos.get((a + da, b + db))
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
        A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(self.Z, self.Z))
        N = np.asarray(A @ self.C.astype(np.float32))

        zero = np.zeros((self.Z, 1))
        self.cumC = np.concatenate([zero, np.cumsum(self.C, axis=1)], axis=1).astype(np.int64)
        self.cumF = np.concatenate([zero, np.cumsum(self.F, axis=1)], axis=1)
        self.cumN = np.concatenate([zero, np.cumsum(N, axis=1)], axis=1)

        ar = np.arange(self.T)[None, :]
        last = np.where(self.C > 0, ar, -10**6)
        self.last_idx = np.maximum.accumulate(last, axis=1)

        # Calendario
        self.dates = pd.date_range(self.start, periods=self.T, freq="D")
        self.doy = self.dates.dayofyear.to_numpy()
        self.month = self.dates.month.to_numpy()
        self.year = self.dates.year.to_numpy()
        self._build_climatology()

    def _build_climatology(self):
        years = np.unique(self.year)
        ny = len(years)
        yi = np.searchsorted(years, self.year)
        mi = self.month - 1
        ym = yi * 12 + mi
        # detecciones por zona y (año, mes)
        M = np.zeros((self.Z, ny * 12))
        for z in range(self.Z):
            M[z] = np.bincount(ym, weights=self.C[z], minlength=ny * 12)
        days = np.bincount(ym, minlength=ny * 12).astype(np.float64)
        M = M.reshape(self.Z, ny, 12)
        days = days.reshape(ny, 12)
        # acumulado de años ANTERIORES (estricto) para el mismo mes
        prevM = np.cumsum(M, axis=1) - M
        prevD = np.cumsum(days, axis=0) - days
        with np.errstate(divide="ignore", invalid="ignore"):
            self.clim = np.where(prevD[None] > 0, prevM / prevD[None], np.nan).astype(np.float32)
        self._yi = yi
        self._mi = mi

    # ------------------------------------------------------------------
    def features(self, zi, di) -> pd.DataFrame:
        zi = np.asarray(zi)
        di = np.asarray(di)

        def win(cum, w):
            lo = np.maximum(di + 1 - w, 0)
            return cum[zi, di + 1] - cum[zi, lo]

        days_since = np.minimum(di - self.last_idx[zi, di], MAX_DAYS_SINCE)
        doy = self.doy[di]
        out = pd.DataFrame({
            "det_1d": win(self.cumC, 1),
            "det_3d": win(self.cumC, 3),
            "det_7d": win(self.cumC, 7),
            "det_30d": win(self.cumC, 30),
            "frp_3d": win(self.cumF, 3),
            "frp_7d": win(self.cumF, 7),
            "nb_3d": win(self.cumN, 3),
            "nb_7d": win(self.cumN, 7),
            "days_since": days_since,
            "clim_month_rate": self.clim[zi, self._yi[di], self._mi[di]],
            "rate_all": self.cumC[zi, di + 1] / (di + 1),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "lat": self.lat_c[zi],
            "lon": self.lon_c[zi],
        })
        return out[FEATURES].astype(np.float32)

    def label(self, zi, di):
        """1 si hay detección en la zona el día siguiente (di+1)."""
        return (self.C[np.asarray(zi), np.asarray(di) + 1] > 0).astype(np.int8)

    def day_index(self, date) -> int:
        di = int((pd.Timestamp(date).normalize() - self.start).days)
        if di < 0 or di >= self.T:
            raise ValueError(
                f"Fecha fuera del rango del histórico ({self.start.date()} a {self.end.date()})."
            )
        return di
