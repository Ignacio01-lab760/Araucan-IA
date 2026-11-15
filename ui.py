"""
Apartado "Predicción" de FireWatch para una app Streamlit.
Uso:   from firewatch.ui import render_prediccion ;  render_prediccion()
"""
from datetime import date

import pandas as pd
import pydeck as pdk
import streamlit as st

from .risk import risk_map, get_panel
from .weather import weather_for_points, temperature_series

COLORS = {"alto": [229, 56, 59], "medio": [245, 158, 11], "bajo": [59, 130, 246]}
ALPHA = {"alto": 170, "medio": 130, "bajo": 35}
PRESETS = {"2 de febrero de 2023": date(2023, 2, 2),
           "2 de febrero de 2024": date(2024, 2, 2),
           "10 de febrero de 2025": date(2025, 2, 10)}


# ---------------------- lógica pura (sin Streamlit) ----------------------
def zone_polygons(r):
    """Zonas como polígonos [lon, lat] para pydeck."""
    rows = []
    for z in r["zones"]:
        (lat0, lon0), (lat1, lon1) = z["bounds"]
        lvl = z["risk_level"]
        rows.append({
            "polygon": [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]],
            "color": COLORS[lvl] + [ALPHA[lvl]],
            "nivel": lvl,
            "prob": f"{z['risk_probability'] * 100:.1f} %",
            "det7": z["detections_7d"],
        })
    return rows


def top_zones(r, n=12):
    return sorted(r["zones"], key=lambda z: z["risk_probability"], reverse=True)[:n]


def zones_table(top, has_truth, weather=None):
    rows = []
    for i, z in enumerate(top):
        row = {
            "Probabilidad": f"{z['risk_probability'] * 100:.1f} %",
            "Nivel": z["risk_level"],
            "Zona (lat, lon)": f"{z['latitude']:.2f}, {z['longitude']:.2f}",
            "Detecciones 7 d": z["detections_7d"],
        }
        if has_truth:
            row["¿Hubo detección al día siguiente?"] = "Sí" if z.get("actual_next_day", 0) > 0 else "No"
        if weather:
            w = weather[i]
            row["Temp. ahora (°C)"] = w["temp_now_c"]
            row["Máx. 24 h (°C)"] = w["temp_max_24h_c"]
            row["HR mín. 24 h (%)"] = w["humidity_min_24h_pct"]
            row["Condiciones críticas"] = ", ".join(w["critical_flags"]) or "—"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------- interfaz ----------------------
@st.cache_data(show_spinner="Calculando riesgo por zona…")
def _risk(as_of, use_weather):
    return risk_map(as_of, use_weather)


@st.cache_data(ttl=900, show_spinner="Consultando temperatura en tiempo real…")
def _weather(points):
    return weather_for_points(list(points))


@st.cache_data(ttl=900)
def _series(lat, lon):
    return temperature_series(lat, lon)


def render_prediccion():
    st.title("🔥 Predicción de focos de calor (24 h)")
    st.warning("Prototipo experimental basado en detecciones satelitales NASA VIIRS. "
               "No es un sistema oficial de alerta ni predice incendios confirmados.")

    panel = get_panel()
    dmin, dmax = panel.start.date(), panel.end.date()

    # --- selector de fecha ---
    opciones = list(PRESETS) + ["Último dato del histórico", "Elegir otra fecha"]
    eleccion = st.selectbox("Fecha de análisis (replay histórico)", opciones, index=1)
    if eleccion in PRESETS:
        as_of = PRESETS[eleccion]
    elif eleccion == "Elegir otra fecha":
        as_of = st.date_input("Fecha", value=date(2024, 2, 2), min_value=dmin, max_value=dmax)
    else:
        as_of = dmax
    as_of_str = None if eleccion == "Último dato del histórico" else str(as_of)

    try:
        r = _risk(as_of_str, True)
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return

    # --- avisos sobre el modelo ---
    if r["has_truth"]:
        st.caption("Backtest honesto: el modelo se entrenó solo con datos hasta 2022." if not r["in_sample"]
                   else "Esta fecha es de entrenamiento del modelo (no es una prueba honesta).")
    else:
        st.caption("Último día del histórico: no hay datos del día siguiente para comparar.")
    st.caption("Modelo con meteorología (NASA POWER)." if r["uses_weather"]
               else "Modelo basado solo en el histórico de detecciones.")

    # --- resumen ---
    zonas = r["zones"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Zonas en riesgo ALTO", sum(z["risk_level"] == "alto" for z in zonas))
    c2.metric("Zonas en riesgo MEDIO", sum(z["risk_level"] == "medio" for z in zonas))
    if r["has_truth"]:
        c3.metric("Zonas con detección real al día siguiente", sum(z.get("actual_next_day", 0) > 0 for z in zonas))

    # --- mapa ---
    layer = pdk.Layer("PolygonLayer", data=zone_polygons(r), get_polygon="polygon",
                      get_fill_color="color", get_line_color=[90, 90, 90, 80],
                      line_width_min_pixels=1, stroked=True, filled=True, pickable=True)
    deck = pdk.Deck(layers=[layer],
                    initial_view_state=pdk.ViewState(latitude=-35.5, longitude=-71.0, zoom=3.6),
                    tooltip={"html": "<b>Riesgo {nivel}</b> · {prob}<br/>Detecciones en 7 días: {det7}"})
    st.pydeck_chart(deck)
    st.caption("Rojo = riesgo alto · naranja = medio · azul tenue = bajo. Cada zona mide ≈ 28 km.")

    # --- zonas más críticas + temperatura en tiempo real ---
    st.subheader("Zonas de mayor riesgo")
    top = top_zones(r, 12)
    con_clima = st.checkbox("Agregar temperatura en tiempo real (Open-Meteo)", value=True)
    clima = None
    if con_clima:
        try:
            clima = _weather(tuple((z["latitude"], z["longitude"]) for z in top))
        except Exception as exc:
            st.info(f"No se pudo consultar la temperatura en tiempo real ({exc}). Se muestra solo el riesgo.")
    st.dataframe(zones_table(top, r["has_truth"], clima), hide_index=True)

    # --- evolución de temperatura de una zona ---
    st.subheader("Evolución de la temperatura")
    etiquetas = [f"{z['latitude']:.2f}, {z['longitude']:.2f} · riesgo {z['risk_level']}" for z in top]
    i = st.selectbox("Zona", range(len(top)), format_func=lambda k: etiquetas[k])
    try:
        s = _series(top[i]["latitude"], top[i]["longitude"])
        df = pd.DataFrame({"Temperatura (°C)": s["temperature_c"]}, index=pd.to_datetime(s["time"]))
        st.line_chart(df)
        st.caption(f"Últimas 24 h y próximas 24 h (Open-Meteo). Temperatura actual: {s['current_temp_c']} °C.")
    except Exception as exc:
        st.info(f"No se pudo obtener la temperatura: {exc}")

    m = r.get("metrics")
    if m:
        with st.expander("¿Qué tan confiable es? (validación 2023–2026)"):
            st.write(
                f"Precisión media del modelo: **{m['model']['average_precision']:.2f}**, contra "
                f"{m['baseline_persistence']['average_precision']:.2f} de 'mañana = hoy' y "
                f"{m['baseline_climatology']['average_precision']:.2f} de la climatología. "
                f"Tasa base de zona-días con detección: {m['base_rate'] * 100:.1f} %.")
            st.write("El riesgo es la probabilidad de que el satélite detecte al menos un foco térmico en la zona "
                     "durante las próximas 24 h; no equivale a incendio confirmado.")
