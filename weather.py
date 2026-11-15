import time
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_CACHE = {}
CACHE_SECONDS = 15 * 60


def _hour_floor(ts):
    """'2026-11-14T15:45' -> '2026-11-14T15:00' (Open-Meteo 'current' viene en pasos de 15 min)."""
    return ts[:13] + ":00" if ts else None


def forecast_24h(latitude: float, longitude: float):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "soil_moisture_0_to_7cm",
        ],
        "forecast_days": 2,
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise RuntimeError("La API meteorológica no devolvió datos horarios.")

    current_time = _hour_floor(payload.get("current", {}).get("time"))
    start_index = times.index(current_time) if current_time in times else 0
    end_index = min(start_index + 24, len(times))

    def values(name):
        return [v for v in hourly.get(name, [])[start_index:end_index] if v is not None]

    temp = values("temperature_2m")
    humidity = values("relative_humidity_2m")
    wind = values("wind_speed_10m")
    gust = values("wind_gusts_10m")
    precip = values("precipitation")
    soil = values("soil_moisture_0_to_7cm")

    return {
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": payload.get("timezone"),
        "period_start": times[start_index],
        "period_end": times[end_index - 1],
        "forecast_horizon_hours": 24,
        "air_temp_mean_c": round(sum(temp) / len(temp), 2) if temp else None,
        "air_temp_max_c": round(max(temp), 2) if temp else None,
        "relative_humidity_min_pct": round(min(humidity), 1) if humidity else None,
        "wind_speed_max_ms": round(max(wind), 2) if wind else None,
        "wind_gust_max_ms": round(max(gust), 2) if gust else None,
        "precipitation_sum_mm": round(sum(precip), 2) if precip else None,
        "soil_moisture_mean_m3m3": round(sum(soil) / len(soil), 4) if soil else None,
        "hourly": {
            "time": times[start_index:end_index],
            "temperature_2m_c": hourly.get("temperature_2m", [])[start_index:end_index],
            "relative_humidity_2m_pct": hourly.get("relative_humidity_2m", [])[start_index:end_index],
            "wind_speed_10m_ms": hourly.get("wind_speed_10m", [])[start_index:end_index],
            "precipitation_mm": hourly.get("precipitation", [])[start_index:end_index],
        },
        "source": "Open-Meteo forecast API",
        "note": "Pronóstico meteorológico, no probabilidad de incendio. Debe combinarse con un modelo entrenado."
    }


def temperature_series(latitude: float, longitude: float):
    """Últimas 24 h + próximas 24 h de temperatura, para graficar la evolución."""
    params = {
        "latitude": latitude, "longitude": longitude,
        "current": "temperature_2m",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "past_days": 1, "forecast_days": 2,
        "timezone": "auto", "wind_speed_unit": "ms",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    h = j["hourly"]
    times = h["time"]
    now = _hour_floor(j.get("current", {}).get("time"))
    i = times.index(now) if now in times else min(24, len(times) - 1)
    lo, hi = max(i - 24, 0), min(i + 25, len(times))
    return {
        "now_index": i - lo,
        "time": times[lo:hi],
        "temperature_c": h["temperature_2m"][lo:hi],
        "humidity_pct": h["relative_humidity_2m"][lo:hi],
        "wind_ms": h["wind_speed_10m"][lo:hi],
        "current_temp_c": j.get("current", {}).get("temperature_2m"),
        "source": "Open-Meteo",
    }


def critical_flags(temp_max, rh_min, wind_max):
    """Regla simple y transparente (NO es un índice oficial de peligro de incendio)."""
    flags = []
    if temp_max is not None and temp_max >= 30:
        flags.append("temperatura máx ≥ 30 °C")
    if rh_min is not None and rh_min <= 30:
        flags.append("humedad mín ≤ 30 %")
    if wind_max is not None and wind_max >= 8:
        flags.append("viento ≥ 8 m/s")
    return flags


def weather_for_points(points):
    """points: lista de (lat, lon). Una sola llamada a Open-Meteo, con caché de 15 min."""
    if not points:
        return []
    key = tuple((round(a, 2), round(b, 2)) for a, b in points)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    params = {
        "latitude": ",".join(str(round(a, 3)) for a, _ in points),
        "longitude": ",".join(str(round(b, 3)) for _, b in points),
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": 2, "timezone": "auto", "wind_speed_unit": "ms",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=40)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):           # un solo punto -> objeto, no lista
        data = [data]

    out = []
    for j in data:
        h = j.get("hourly", {})
        times = h.get("time", [])
        now = _hour_floor(j.get("current", {}).get("time"))
        i = times.index(now) if now in times else 0
        sl = slice(i, i + 24)
        t = [v for v in h.get("temperature_2m", [])[sl] if v is not None]
        rh = [v for v in h.get("relative_humidity_2m", [])[sl] if v is not None]
        w = [v for v in h.get("wind_speed_10m", [])[sl] if v is not None]
        cur = j.get("current", {})
        tmax, rhmin, wmax = (max(t) if t else None), (min(rh) if rh else None), (max(w) if w else None)
        out.append({
            "temp_now_c": cur.get("temperature_2m"),
            "humidity_now_pct": cur.get("relative_humidity_2m"),
            "wind_now_ms": cur.get("wind_speed_10m"),
            "temp_max_24h_c": round(tmax, 1) if tmax is not None else None,
            "humidity_min_24h_pct": round(rhmin, 0) if rhmin is not None else None,
            "wind_max_24h_ms": round(wmax, 1) if wmax is not None else None,
            "critical_flags": critical_flags(tmax, rhmin, wmax),
        })
    _CACHE[key] = (time.time(), out)
    return out
