import base64
import inspect
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Araucan-IA", page_icon="🔥", layout="wide")

try:
    import shapely
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
except ImportError:
    st.error("Falta instalar shapely. Ejecuta en la terminal:  pip install shapely")
    st.stop()

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
ORANGE, BORDER = "#ff9f1c", "#1c2c4a"
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ETIQUETAS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"

# ── Rutas ──
CARPETA = Path(__file__).parent
RUTA_LOGO = CARPETA / "logo.png"
RUTA_BANNER = CARPETA / "nasa_x_araucania.png"
RUTA_DETECCIONES = CARPETA / "detecciones.zip"
RUTA_POLIGONOS = CARPETA / "poligonos.zip"
RUTA_CACHE_POL = CARPETA / "geodatos" / "poligonos_cache.pkl"   # se genera solo la primera vez
RUTA_ORIGINAL = str(CARPETA / "geodatos" / "regiones_chile.geojson")
RUTA_GEOJSON = str(CARPETA / "geodatos" / "regiones_chile_ligero.geojson")
CAMPO_NOMBRE = "REGION"  # si no detecta bien el nombre de la región, escribe aquí la propiedad, ej. "Region"
TODO_CHILE = "Todo Chile"

# ── Compatibilidad entre versiones de Streamlit / Plotly ──
NUEVO_ST = "width" in inspect.signature(st.button).parameters
STRETCH = {"width": "stretch"} if NUEVO_ST else {"use_container_width": True}
USA_MAP = hasattr(go, "Scattermap")
TraceMapa = go.Scattermap if USA_MAP else go.Scattermapbox
CLAVE_MAPA = "map" if USA_MAP else "mapbox"

# ── Colores de los niveles de actividad de los polígonos (celeste -> magenta) ──
PALETAS_NIVEL = {
    1: ["#f72585"],
    2: ["#4cc9f0", "#f72585"],
    3: ["#4cc9f0", "#7b2ff7", "#f72585"],
    4: ["#90e0ef", "#4cc9f0", "#7b2ff7", "#f72585"],
}


def colores_niveles(n: int):
    if n in PALETAS_NIVEL:
        return PALETAS_NIVEL[n]
    base = PALETAS_NIVEL[4]
    return [base[int(i * (len(base) - 1) / max(n - 1, 1) + 0.5)] for i in range(n)]


def H(s: str) -> str:
    """Compacta el HTML en una línea (evita que Markdown lo rompa)."""
    return "".join(line.strip() for line in s.strip().splitlines())


def normalizar(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn").lower()


def img_b64(ruta: Path) -> str:
    """Convierte una imagen local a texto para poder usarla dentro del HTML."""
    ext = ruta.suffix.lower().lstrip(".")
    mime = {"svg": "svg+xml", "jpg": "jpeg"}.get(ext, ext)
    return f"data:image/{mime};base64," + base64.b64encode(ruta.read_bytes()).decode()


def fmt(n):
    return f"{int(round(n)):,}".replace(",", ".")


def fmt1(n, d=1):
    """Número con coma decimal: 1234.5 -> 1.234,5"""
    return f"{n:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


# ═════════════════════════ CSS ═════════════════════════
st.markdown("""
<style>
:root{ --bg:#08111f; --card:#0f1b2f; --border:#1c2c4a; --txt:#e8eefc; --muted:#9fb0d0; --blue:#1976d2; }
.stApp{ background:var(--bg); color:var(--txt); }

/* Oculta solo Deploy, el menú y la decoración (NO la barra de arriba completa) */
[data-testid="stDecoration"], [data-testid="stMainMenu"], [data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"], .stAppDeployButton, footer{ display:none !important; }
header[data-testid="stHeader"]{ background:transparent; height:2.5rem; }

.block-container{ padding:1rem 1.2rem 1rem 1.2rem; max-width:100%; }

/* Botón para volver a abrir la barra lateral */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"]{
    display:flex !important; visibility:visible !important;
    position:fixed; top:10px; left:10px; z-index:999999;
    background:#12305a; border:1px solid #1c2c4a; border-radius:8px; padding:2px 4px; }
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *{
    color:#fff !important; fill:#fff !important; }

/* Cuando la barra está cerrada, baja un poco el contenido para no tapar el logo */
.stApp:has(section[data-testid="stSidebar"][aria-expanded="false"]) .block-container{ padding-top:3rem; }

[data-testid="stVerticalBlock"]{ gap:0.8rem; }

/* ---------- Barra lateral ---------- */
section[data-testid="stSidebar"][aria-expanded="true"]{ min-width:210px !important; width:210px !important; }
section[data-testid="stSidebar"]{ background:#0a1526; border-right:1px solid var(--border); }
section[data-testid="stSidebar"] > div{ padding-top:2.2rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:0.35rem; }
section[data-testid="stSidebar"] .stButton button{
    justify-content:flex-start; border:0; background:transparent; color:#dbe5fa;
    height:46px; border-radius:8px; font-weight:500; gap:10px; }
section[data-testid="stSidebar"] .stButton button:hover{ background:#12233f; color:#fff; }
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"]{
    background:var(--blue); color:#fff; font-weight:700; }
section[data-testid="stSidebar"]::after{
    content:""; position:fixed; left:0; bottom:0; width:210px; height:170px; pointer-events:none;
    background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='210' height='160' viewBox='0 0 210 160'><g fill='rgb(30,62,62)' fill-opacity='0.55'><polygon points='5,160 25,75 45,160'/><polygon points='35,160 60,50 85,160'/><polygon points='80,160 100,88 120,160'/><polygon points='115,160 140,62 165,160'/><polygon points='160,160 185,95 215,160'/></g></svg>") no-repeat bottom;
}

/* ---------- Encabezado ---------- */
.topbar{ display:flex; align-items:center; gap:26px; padding:4px 6px 14px 6px; }
.topbar .sep{ width:1px; height:58px; background:#2a3c5e; }
.brand{ display:flex; align-items:center; gap:16px; }
.brand .name{ font-size:34px; font-weight:800; color:#fff; line-height:1.05; }
.brand .sub{ font-size:15px; color:#dbe5fa; }
.ttl{ display:flex; align-items:center; gap:14px; }
.ttl .flame{ font-size:46px; line-height:1; }
.ttl .t1{ font-size:26px; font-weight:800; color:#fff; line-height:1.1; }
.ttl .t2{ font-size:16px; color:#c9d4ec; }
.tag{ margin-left:auto; display:flex; align-items:center; gap:18px; font-style:italic; color:#c9d4ec; font-size:13px; text-align:right; }

/* ---------- Filtros ---------- */
[data-testid="stForm"]{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:10px 14px; }
[data-testid="stForm"] [data-testid="stVerticalBlock"]{ gap:0; }
.flabel{ color:#dbe5fa; font-size:15px; padding-top:8px; white-space:nowrap; }
div[data-baseweb="select"] > div{ background:#0c1830; border:1px solid var(--border); border-radius:8px; }
[data-testid="stForm"] button[data-testid="stBaseButton-primaryFormSubmit"]{
    background:#1d84e8; border:0; color:#fff; font-weight:600; height:42px; border-radius:8px; }

/* ---------- KPI ---------- */
.kpi{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:10px 20px;
      display:flex; align-items:center; gap:18px; min-height:108px; box-sizing:border-box;
      overflow:hidden; margin-bottom:8px; }
.kpi .ico{ width:64px; height:64px; min-width:64px; border-radius:50%; display:flex;
           align-items:center; justify-content:center; font-size:30px; }
.kpi .lbl{ color:#dbe5fa; font-size:15px; }
.kpi .val{ font-size:34px; font-weight:800; color:#fff; line-height:1.15; white-space:nowrap; }
.kpi .sub{ font-size:13px; color:var(--muted); white-space:nowrap; }
.up{ color:#3ddc84; } .down{ color:#ff6b6b; }

/* ---------- Tarjetas / paneles ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--card); border:1px solid var(--border) !important; border-radius:12px; }
.ctitle{ font-size:18px; font-weight:700; color:#fff; display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.ctitle .box{ background:#12305a; border-radius:6px; padding:2px 6px; font-size:15px; }
.panel{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px 20px; height:470px; }
.panel .ptitle{ font-size:20px; font-weight:700; color:#fff; padding-bottom:10px; border-bottom:1px solid var(--border); margin-bottom:6px; }
.panel .row{ display:flex; gap:12px; margin:18px 0; color:#c9d4ec; font-size:14px; line-height:1.5; }
.panel .row b{ color:#fff; font-size:15px; }
.panel .note{ border-top:1px solid var(--border); padding-top:14px; margin-top:20px; font-size:12.5px; color:var(--muted); display:flex; gap:12px; }

/* ---------- Mapa (contenedor) ---------- */
.st-key-mapbox{ margin-top:6px; position:relative; z-index:0; border:1px solid var(--border); border-radius:12px; overflow:hidden; }
.st-key-mapbox [data-testid="stVerticalBlock"]{ gap:0; }
.gradbar{ height:14px; border-radius:7px; background:linear-gradient(90deg,#8fd3f4,#fde68a,#fb923c,#dc2626,#7f0a0a); flex:1; }
.leyenda{ display:flex; gap:16px; align-items:center; padding-top:6px; color:#dbe5fa; font-size:13px; flex-wrap:wrap; }
.dot{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }

/* ---------- Pie ---------- */
.footer{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:0 18px;
         height:62px; display:flex; align-items:center; gap:16px; color:#dbe5fa; font-size:14.5px; }
.footer .ok{ width:36px; height:36px; border-radius:50%; background:#1f8f4e; display:flex; align-items:center; justify-content:center; color:#0b1426; font-weight:900; }
.footer .vbar{ width:1px; height:22px; background:#2a3c5e; }
div[data-testid="stDownloadButton"] button{ background:#1d84e8; color:#fff; border:0; font-weight:600; height:46px; border-radius:8px; }
.st-key-exp button{ background:transparent !important; border:1px solid #cfd8ea !important; }

/* KPI en 6 columnas (va al final para que sobrescriba los estilos de arriba) */
.kpi{ padding:0 14px; gap:12px; min-height:112px; }
.kpi .ico{ width:52px; height:52px; min-width:52px; font-size:24px; }
.kpi .lbl{ font-size:14px; }
.kpi .val{ font-size:28px; }
.kpi .sub{ font-size:12px; white-space:normal; line-height:1.3; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════ Datos: detecciones.csv ═════════════════════════
def nivel_confianza(serie: pd.Series) -> pd.Categorical:
    """Unifica la confianza: numérica (0-100) o por letras (l/n/h) -> baja / nominal / alta."""
    s = serie.astype(str).str.strip().str.lower()
    n = pd.to_numeric(s, errors="coerce").to_numpy()
    con_numero = ~np.isnan(n)
    nivel = np.select(
        [con_numero & (n >= 80), con_numero & (n >= 30), con_numero,
         s.isin(["h", "high", "alta", "alto"]).to_numpy(),
         s.isin(["n", "nominal", "media", "medio"]).to_numpy(),
         s.isin(["l", "low", "baja", "bajo"]).to_numpy()],
        ["alta", "nominal", "baja", "alta", "nominal", "baja"], default="sin dato")
    return pd.Categorical(nivel, categories=["baja", "nominal", "alta", "sin dato"])


@st.cache_data(show_spinner="📥 Cargando las detecciones (solo la primera vez, puede tardar un poco)...")
def cargar_datos() -> pd.DataFrame:
    cols = ["latitude", "longitude", "fecha", "frp_mw", "confidence", "sensor", "satellite"]
    d = pd.read_csv(RUTA_DETECCIONES, usecols=cols, low_memory=False,
                    dtype={"confidence": str, "sensor": "category", "satellite": "category"})
    d = d.rename(columns={"latitude": "latitud", "longitude": "longitud", "frp_mw": "frp",
                          "sensor": "fuente", "satellite": "satelite"})
    d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
    d = d.dropna(subset=["latitud", "longitud", "fecha"]).reset_index(drop=True)
    d["confianza"] = nivel_confianza(d.pop("confidence"))
    d["frp"] = d["frp"].astype("float32")
    d["año"] = d["fecha"].dt.year.astype("int16")
    d["mes"] = d["fecha"].dt.month.astype("int8")
    return d


if not RUTA_DETECCIONES.exists():
    st.error(f"No encontré detecciones.csv. Debe estar junto a Interfaz.py, en: {CARPETA}")
    st.stop()

df_all = cargar_datos()
y_min, y_max = int(df_all["año"].min()), int(df_all["año"].max())


# ═════════════════════════ GeoJSON de regiones ═════════════════════════
CANDIDATOS_NOMBRE = ["Region", "REGION", "region", "Región", "REGIÓN", "NOM_REG", "NOM_REGION", "nom_reg",
                     "NOMBRE", "Nombre", "nombre", "NAME_1", "name", "NAME", "Name"]


def preparar_geojson_ligero(origen: str, destino: str, campo_forzado=None, tolerancia=0.005):
    """Se ejecuta UNA sola vez: simplifica el GeoJSON pesado y guarda una copia liviana."""
    with open(origen, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]

    props0 = feats[0].get("properties") or {}
    campo = campo_forzado or next((c for c in CANDIDATOS_NOMBRE if c in props0), None)
    if campo is None:
        campo = next((k for k, v in props0.items() if isinstance(v, str)), None)

    grupos = {}
    for i, ft in enumerate(feats):
        if not ft.get("geometry"):
            continue
        props = ft.get("properties") or {}
        nombre = str(props.get(campo, f"Región {i + 1}")).strip() if campo else f"Región {i + 1}"
        grupos.setdefault(nombre, []).append(shape(ft["geometry"]).simplify(tolerancia, preserve_topology=True))

    salida = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"nombre": n},
         "geometry": mapping(unary_union([g.buffer(0) for g in gs]))}
        for n, gs in grupos.items()]}
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(salida, f)


@st.cache_resource(show_spinner="🗺️ Cargando los mapas, por favor espera unos minutos....")
def cargar_regiones(ruta: str):
    """Lee el GeoJSON liviano. Devuelve {nombre: {geom, dibujo, bounds}} y el contorno de todas las regiones."""
    with open(ruta, encoding="utf-8") as f:
        gj = json.load(f)

    regiones, todas = {}, []
    for ft in gj["features"]:
        nombre = ft["properties"]["nombre"]
        geom = shape(ft["geometry"])
        # para centrar/zoom se usa la parte continental más grande (ignora islas lejanas)
        base = max(geom.geoms, key=lambda g: g.area) if geom.geom_type == "MultiPolygon" else geom
        regiones[nombre] = {"geom": geom,
                            "dibujo": mapping(geom.simplify(0.01, preserve_topology=True)),
                            "bounds": base.bounds}
        todas.append({"type": "Feature", "properties": {},
                      "geometry": mapping(geom.simplify(0.03, preserve_topology=True))})
    return regiones, {"type": "FeatureCollection", "features": todas}


regiones, CONTORNO = {}, None
if Path(RUTA_ORIGINAL).exists() or Path(RUTA_GEOJSON).exists():
    if not Path(RUTA_GEOJSON).exists():
        with st.spinner("Preparando el mapa de regiones (solo la primera vez)..."):
            preparar_geojson_ligero(RUTA_ORIGINAL, RUTA_GEOJSON, CAMPO_NOMBRE)
    regiones, CONTORNO = cargar_regiones(RUTA_GEOJSON)
    xs = [v["bounds"] for v in regiones.values()]
    if xs and (min(b[0] for b in xs) < -180 or max(b[2] for b in xs) > 180 or
               min(b[1] for b in xs) < -90 or max(b[3] for b in xs) > 90):
        st.error("El GeoJSON no está en lon/lat (EPSG:4326). Reproyéctalo a WGS84 y vuelve a intentar.")
        st.stop()
else:
    st.warning("No encontré regiones_chile.geojson dentro de la carpeta geodatos: se muestra solo «Todo Chile».")


@st.cache_data(show_spinner="📍 Filtrando los focos por región (solo la primera vez)...")
def mascara_region(nombre: str) -> np.ndarray:
    """True para los focos que caen dentro del polígono de la región."""
    return shapely.contains_xy(regiones[nombre]["geom"], df_all["longitud"].to_numpy(), df_all["latitud"].to_numpy())


def filtrar_detecciones(region_sel, ini, fin, fuente_sel):
    """ini y fin son fechas. fin NO se incluye (es el día siguiente al último día elegido)."""
    m = (df_all["fecha"] >= ini) & (df_all["fecha"] < fin)
    if region_sel != TODO_CHILE:
        m &= mascara_region(region_sel)
    if fuente_sel != "Serie armonizada":
        m &= (df_all["fuente"] == fuente_sel)
    return df_all[m]


@st.cache_data(show_spinner=False)
def csv_filtrado(region_sel, ini, fin, fuente_sel, max_filas=500_000) -> bytes:
    d = filtrar_detecciones(region_sel, ini, fin, fuente_sel).drop(columns=["año", "mes"])
    return d.head(max_filas).to_csv(index=False).encode("utf-8")


# ═════════════════════════ Datos: poligonos.geojson ═════════════════════════
def _redondear(coords, n=4):
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            return [round(c, n) for c in coords]
        return [_redondear(c, n) for c in coords]
    return coords


def preparar_poligonos(origen: Path, destino: Path):
    """Se ejecuta UNA sola vez: lee el GeoJSON grande y guarda una versión compacta (tabla + una geometría por celda)."""
    with open(origen, encoding="utf-8") as f:
        gj = json.load(f)

    celdas, geoms, lons, lats = {}, [], [], []
    cell, per, cant, frpm, niv = [], [], [], [], []
    for ft in gj["features"]:
        p = ft.get("properties") or {}
        clave = (p.get("celda_x"), p.get("celda_y"))
        i = celdas.get(clave)
        if i is None:                      # cada celda se guarda una sola vez
            i = len(geoms)
            celdas[clave] = i
            g = ft["geometry"]
            minx, miny, maxx, maxy = shape(g).bounds
            lons.append((minx + maxx) / 2)
            lats.append((miny + maxy) / 2)
            geoms.append({"type": g["type"], "coordinates": _redondear(g["coordinates"])})
        cell.append(i)
        per.append(p.get("periodo"))
        cant.append(p.get("cantidad_detecciones"))
        frpm.append(p.get("frp_maximo_mw"))
        niv.append(p.get("nivel_actividad"))
    del gj

    props = pd.DataFrame({
        "cell": np.asarray(cell, dtype="int32"),
        "fecha": pd.to_datetime(pd.Series(per), errors="coerce"),
        "cantidad": pd.to_numeric(pd.Series(cant), errors="coerce"),
        "frp_max": pd.to_numeric(pd.Series(frpm), errors="coerce"),
        "nivel": pd.Series(niv, dtype="category"),
    })
    props = props[props["fecha"].notna()].reset_index(drop=True)
    props["año"] = props["fecha"].dt.year.astype("int16")
    props["cantidad"] = props["cantidad"].fillna(0).astype("int32")
    props["frp_max"] = props["frp_max"].astype("float32")

    destino.parent.mkdir(exist_ok=True)
    pd.to_pickle({"props": props, "lon_c": np.asarray(lons), "lat_c": np.asarray(lats), "geoms": geoms}, destino)


@st.cache_resource(show_spinner="🗺️ Preparando los polígonos de actividad. La primera vez puede tardar varios minutos; no es un error...")
def cargar_poligonos(mtime: float):
    if (not RUTA_CACHE_POL.exists()) or RUTA_CACHE_POL.stat().st_mtime < RUTA_POLIGONOS.stat().st_mtime:
        preparar_poligonos(RUTA_POLIGONOS, RUTA_CACHE_POL)
    datos = pd.read_pickle(RUTA_CACHE_POL)
    if "fecha" not in datos["props"].columns:          # caché antigua (sin fechas): se vuelve a generar
        preparar_poligonos(RUTA_POLIGONOS, RUTA_CACHE_POL)
        datos = pd.read_pickle(RUTA_CACHE_POL)
    props, lon_c, lat_c, geoms = datos["props"], datos["lon_c"], datos["lat_c"], datos["geoms"]
    props = props.drop(columns=["temp_max"], errors="ignore")

    # niveles ordenados de menor a mayor actividad (según el promedio de detecciones de cada nivel)
    niveles = props.groupby("nivel", observed=True)["cantidad"].mean().sort_values().index.astype(str).tolist()
    rank = {n: i for i, n in enumerate(niveles)}
    props["nivel_r"] = props["nivel"].astype(object).map(rank).fillna(0).astype("int8")

    # región de cada celda según su centro
    region_celda = np.full(len(lon_c), "Sin región", dtype=object)
    for nombre, info in regiones.items():
        dentro = shapely.contains_xy(info["geom"], lon_c, lat_c) & (region_celda == "Sin región")
        region_celda[dentro] = nombre
    props["region"] = pd.Categorical(region_celda[props["cell"].to_numpy()])
    return props, geoms, lon_c, lat_c, niveles


if RUTA_POLIGONOS.exists():
    poligonos_all, GEOMS, LON_C, LAT_C, NIVELES = cargar_poligonos(RUTA_POLIGONOS.stat().st_mtime)
else:
    poligonos_all, GEOMS, LON_C, LAT_C, NIVELES = None, [], np.array([]), np.array([]), []
    st.warning("No encontré poligonos.geojson junto a Interfaz.py: no se mostrarán los polígonos.")


def filtrar_poligonos(region_sel, ini, fin):
    m = (poligonos_all["fecha"] >= ini) & (poligonos_all["fecha"] < fin)
    if region_sel != TODO_CHILE:
        m &= (poligonos_all["region"] == region_sel)
    return poligonos_all[m]


@st.cache_data(show_spinner=False)
def agregar_celdas(region_sel, ini, fin, max_celdas=20000) -> pd.DataFrame:
    """Junta todos los días del período por celda (para dibujar menos polígonos)."""
    pol = filtrar_poligonos(region_sel, ini, fin)
    agg = pol.groupby("cell", sort=False, observed=True).agg(
        det=("cantidad", "sum"), dias=("cantidad", "size"),
        frp_max=("frp_max", "max"), nivel_r=("nivel_r", "max"))
    if len(agg) > max_celdas:
        agg = agg.nlargest(max_celdas, "det")
    return agg


# ═════════════════════════ Barra lateral ═════════════════════════
if "pagina" not in st.session_state:
    st.session_state.pagina = "Resumen"

NAV = [("Resumen", ":material/home:"), ("Mapa de actividad", ":material/location_on:"),
       ("Calendario", ":material/calendar_month:"), ("Estadísticas", ":material/bar_chart:"),
       ("Datos y descargas", ":material/download:")]
with st.sidebar:
    for nombre, icono in NAV:
        activo = st.session_state.pagina == nombre
        if st.button(nombre, icon=icono, key=f"nav_{nombre}", type="primary" if activo else "secondary", **STRETCH):
            st.session_state.pagina = nombre
            st.rerun()
pagina = st.session_state.pagina

# ═════════════════════════ Encabezado ═════════════════════════
MONTE = """<svg width='90' height='34' viewBox='0 0 90 34'><polyline points='0,32 22,12 32,22 50,2 70,24 78,16 90,32'
fill='none' stroke='white' stroke-width='2' stroke-linejoin='round'/></svg>"""

if RUTA_LOGO.exists():
    LOGO = f"<img src='{img_b64(RUTA_LOGO)}' style='height:66px;width:auto;border-radius:8px'>"
else:
    st.warning(f"No encontré el logo en esta ruta: {RUTA_LOGO}")
    LOGO = """<svg width='66' height='66' viewBox='0 0 64 64'><circle cx='32' cy='32' r='30' fill='#0b3d91'/>
    <text x='32' y='37' font-size='15' font-weight='700' fill='white' text-anchor='middle' font-family='Arial'>NASA</text></svg>"""

st.markdown(H(f"""
<div class="topbar">
  <div class="brand">{LOGO}<div><div class="name">Araucan-IA</div><div class="sub">NASA Space Apps Challenge 2026</div></div></div>
  <div class="sep"></div>
  <div class="ttl"><div class="flame">📅</div>
     <div><div class="t1">Calendario de Actividad de Incendios</div><div class="t2">Serie histórica armonizada MODIS + VIIRS</div></div></div>
  <div class="tag"><div>Datos que informan,<br>para un Chile más seguro</div>{MONTE}</div>
</div>"""), unsafe_allow_html=True)

# ═════════════════════════ Filtros (con botón Aplicar) ═════════════════════════
f_min, f_max = df_all["fecha"].min().date(), df_all["fecha"].max().date()
opciones_region = [TODO_CHILE] + list(regiones.keys())
idx_region = next((i for i, n in enumerate(opciones_region) if "araucan" in normalizar(n)), 0)
opciones_fuente = ["Serie armonizada"] + [str(x) for x in df_all["fuente"].cat.categories]

with st.form("filtros", border=False):
    c = st.columns([0.55, 2.0, 0.55, 1.35, 0.55, 1.35, 0.7, 1.6, 1.6])
    c[0].markdown('<div class="flabel">Región:</div>', unsafe_allow_html=True)
    region = c[1].selectbox("Región", opciones_region, index=idx_region, label_visibility="collapsed")
    c[2].markdown('<div class="flabel">Desde:</div>', unsafe_allow_html=True)
    desde_f = c[3].date_input("Desde", value=f_min, min_value=f_min, max_value=f_max,
                              format="DD/MM/YYYY", label_visibility="collapsed")
    c[4].markdown('<div class="flabel">Hasta:</div>', unsafe_allow_html=True)
    hasta_f = c[5].date_input("Hasta", value=f_max, min_value=f_min, max_value=f_max,
                              format="DD/MM/YYYY", label_visibility="collapsed")
    c[6].markdown('<div class="flabel">Fuente:</div>', unsafe_allow_html=True)
    fuente = c[7].selectbox("Fuente", opciones_fuente, label_visibility="collapsed")
    c[8].form_submit_button("Aplicar filtros", icon=":material/search:", type="primary", **STRETCH)

if desde_f > hasta_f:
    desde_f, hasta_f = hasta_f, desde_f

ini = pd.Timestamp(desde_f)                    # primer día elegido
fin_dia = pd.Timestamp(hasta_f)                # último día elegido (incluido)
fin = fin_dia + pd.Timedelta(days=1)           # límite superior, no incluido
desde, hasta = ini.year, fin_dia.year          # años del rango (para el calendario y los gráficos anuales)

df = filtrar_detecciones(region, ini, fin, fuente)
nombre_region = "Chile" if region == TODO_CHILE else f"{region}, Chile"

# ═════════════════════════ Métricas ═════════════════════════
anual = df.groupby("año", observed=True).size()
total = len(df)
prom = anual.mean() if len(anual) else 0
ultimo = anual.iloc[-1] if len(anual) else 0
hist = anual.iloc[:-1].mean() if len(anual) > 1 else prom
anomalia = (ultimo / hist - 1) * 100 if hist else 0
año_max = int(anual.idxmax()) if len(anual) else 0
val_max = int(anual.max()) if len(anual) else 0
top = df.groupby("mes", observed=True).size().sort_values(ascending=False).head(3).index.sort_values()
txt_meses = f"{MESES[top[0] - 1]} – {MESES[top[-1] - 1]}" if len(top) else "-"

# variación frente al período inmediatamente anterior y de la misma duración
ini_prev = ini - (fin - ini)
if ini_prev >= pd.Timestamp(f_min):
    total_prev = len(filtrar_detecciones(region, ini_prev, ini, fuente))
    delta = (total / total_prev - 1) * 100 if total_prev else None
else:
    delta = None

# ── KPI desde la tabla de detecciones ──
tiene = total > 0
frp_prom = float(df["frp"].mean()) if tiene else None
frp_max = float(df["frp"].max()) if tiene else None
alta_pct = float((df["confianza"] == "alta").mean() * 100) if tiene else None

# día con más detecciones dentro del rango elegido
if tiene:
    por_dia = df.groupby("fecha").size()
    dia_pico = por_dia.idxmax().strftime("%d-%m-%Y")
    dia_pico_n = int(por_dia.max())
else:
    dia_pico, dia_pico_n = None, 0

# ── KPI desde la tabla de polígonos ──
if poligonos_all is not None and NIVELES:
    pol = filtrar_poligonos(region, ini, fin)
    n_pol = len(pol)
    n_alto = int((pol["nivel_r"] == len(NIVELES) - 1).sum())
    nivel_top = NIVELES[-1]
else:
    n_pol = n_alto = None
    nivel_top = "alto"


def kpi(icono, bg, etiqueta, valor, sub):
    return H(f"""<div class="kpi"><div class="ico" style="background:{bg}">{icono}</div>
    <div><div class="lbl">{etiqueta}</div><div class="val">{valor}</div><div class="sub">{sub}</div></div></div>""")


def base_layout(fig, h):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d4ec", size=12))
    return fig


# ═════════════════════════ Figuras ═════════════════════════
def fig_calendario(h=290):
    n_años = hasta - desde + 1
    if len(df):
        idx = (df["año"].to_numpy().astype(np.int64) - desde) * 12 + df["mes"].to_numpy().astype(np.int64) - 1
        z = np.bincount(idx, minlength=n_años * 12)[: n_años * 12].reshape(n_años, 12).astype(float)
    else:
        z = np.zeros((n_años, 12))
    z[0, : ini.month - 1] = np.nan          # meses anteriores al rango elegido
    z[-1, fin_dia.month:] = np.nan          # meses posteriores al rango elegido
    fig = go.Figure(go.Heatmap(
        z=z, x=MESES, y=list(range(desde, hasta + 1)), showscale=False, xgap=1, ygap=1, hoverongaps=False,
        colorscale=[[0, "#8fd3f4"], [0.3, "#fde68a"], [0.55, "#fb923c"], [0.8, "#dc2626"], [1, "#7f0a0a"]]))
    ticks = list(range(desde, hasta + 1, 5))
    if hasta not in ticks:
        ticks.append(hasta)
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=ticks, showgrid=False)
    return base_layout(fig, h)


def fig_mensual():
    m = df.groupby("mes", observed=True).size().reindex(range(1, 13), fill_value=0)
    fig = go.Figure(go.Bar(x=MESES, y=m.values, marker_color=ORANGE))
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return base_layout(fig, 290)


def fig_anual():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anual.index, y=anual.values, mode="lines+markers", name="Actividad anual",
                             line=dict(color=ORANGE, width=2), marker=dict(size=6),
                             fill="tozeroy", fillcolor="rgba(255,159,28,0.15)"))
    fig.add_trace(go.Scatter(x=[desde, hasta], y=[prom, prom], mode="lines", name="Promedio histórico",
                             line=dict(color="#4da3ff", dash="dash", width=1.5)))
    if len(anual):
        fig.add_annotation(x=año_max, y=val_max, text=f"{año_max}: {fmt(val_max)} focos", ax=0, ay=-28,
                           bgcolor="#8b1e1e", font=dict(color="white"), arrowcolor="#ff4d4d", borderpad=4)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0.1))
    return base_layout(fig, 290)


# ═════════════════════════ Componentes ═════════════════════════
def render_kpis():
    def v(valor, unidad="", dec=1):
        """Muestra '—' mientras no haya dato."""
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            return "—"
        return fmt1(valor, dec) + unidad

    k = st.columns(6)
    if delta is None:
        texto_delta = "Sin período anterior comparable"
    else:
        cls, fl = ("up", "▲") if delta >= 0 else ("down", "▼")
        texto_delta = f'<span class="{cls}">{fl} {delta:+.0f}% vs. periodo anterior</span>'

    k[0].markdown(kpi("🔥", "#4a2a1f", "Focos detectados", fmt(total),
                      f'{texto_delta}<br>Anomalía {anomalia:+.0f}% · prom. {fmt(prom)}/año'),
                  unsafe_allow_html=True)
    k[1].markdown(kpi("⚡", "#3a3320", "FRP promedio", v(frp_prom, " MW"),
                      f"Máximo: {v(frp_max, ' MW')}"), unsafe_allow_html=True)
    k[2].markdown(kpi("📆", "#4a2020", "Día más activo", dia_pico or "—",
                      f"con {fmt(dia_pico_n)} focos"), unsafe_allow_html=True)
    k[3].markdown(kpi("✅", "#1f4a2e", "Alta confianza", v(alta_pct, "%", 0),
                      "de las detecciones"), unsafe_allow_html=True)
    k[4].markdown(kpi("🗺️", "#11315a", "Polígonos en nivel alto", v(None if n_alto is None else float(n_alto), "", 0),
                      f"nivel «{nivel_top}» · de {v(None if n_pol is None else float(n_pol), '', 0)} polígonos"),
                  unsafe_allow_html=True)
    k[5].markdown(kpi("🏆", "#2a2552", "Año de mayor actividad", str(año_max or "-"),
                      f"con {fmt(val_max)} focos"), unsafe_allow_html=True)


def vista_mapa(altura, ancho=1000):
    """Centro y zoom del mapa según la región seleccionada."""
    if region == TODO_CHILE or region not in regiones:
        return dict(lat=-36.0, lon=-72.0), 3.0
    minx, miny, maxx, maxy = regiones[region]["bounds"]
    lon_span, lat_span = max(maxx - minx, 0.01), max(maxy - miny, 0.01)
    lat_c, lon_c = (miny + maxy) / 2, (minx + maxx) / 2
    z_lon = np.log2(360 * (ancho / 512) / (lon_span * 1.15))
    z_lat = np.log2(360 * (altura / 512) * np.cos(np.radians(lat_c)) / (lat_span * 1.15))
    return dict(lat=lat_c, lon=lon_c), float(np.clip(min(z_lon, z_lat), 2.5, 10))


def render_mapa(altura=470, ver_pol=True, ver_foc=True):
    capas = [dict(below="traces", sourcetype="raster", source=[ESRI])]
    if CONTORNO:  # contorno tenue de todas las regiones
        capas.append(dict(below="traces", sourcetype="geojson", source=CONTORNO, type="line",
                          color="rgba(255,255,255,0.35)", line=dict(width=0.8)))

    fig = go.Figure()

    # ── Capa de polígonos de actividad (tabla poligonos) ──
    if ver_pol and poligonos_all is not None and NIVELES:
        agg = agregar_celdas(region, ini, fin)
        if len(agg):
            cols = colores_niveles(len(NIVELES))
            for r in sorted(agg["nivel_r"].unique()):
                ids = agg.index[agg["nivel_r"] == r]
                fc = {"type": "FeatureCollection",
                      "features": [{"type": "Feature", "properties": {}, "geometry": GEOMS[i]} for i in ids]}
                capas.append(dict(below="traces", sourcetype="geojson", source=fc, type="fill",
                                  color=cols[int(r)], opacity=0.5,
                                  fill=dict(outlinecolor="rgba(255,255,255,0.3)")))
            # puntos invisibles en el centro de cada celda, solo para mostrar el detalle al pasar el cursor
            texto = ("<b>Polígono de actividad</b><br>Detecciones: " + agg["det"].astype(int).astype(str)
                     + "<br>Días con actividad: " + agg["dias"].astype(str)
                     + "<br>FRP máx.: " + agg["frp_max"].round(1).astype(str) + " MW"
                     + "<br>Nivel: " + agg["nivel_r"].map(lambda r: NIVELES[int(r)].capitalize()))
            fig.add_trace(TraceMapa(lat=LAT_C[agg.index], lon=LON_C[agg.index], mode="markers",
                                    marker=dict(size=14, opacity=0.01), text=texto,
                                    hovertemplate="%{text}<extra></extra>", showlegend=False))

    # ── Focos de calor (tabla detecciones), coloreados por FRP ──
    if ver_foc and len(df):
        d = df.sample(min(len(df), 15000), random_state=1)
        q1, q2 = d["frp"].quantile([0.5, 0.85])
        grupos = [
            ("FRP baja", d[d["frp"] < q1], "#ffd60a"),
            ("FRP media", d[(d["frp"] >= q1) & (d["frp"] < q2)], "#ff8c00"),
            ("FRP alta", d[d["frp"] >= q2], "#e61e1e"),
        ]
        for nombre, g, color in grupos:
            if g.empty:
                continue
            texto = ("<b>" + g["fecha"].dt.strftime("%d-%m-%Y") + "</b><br>"
                     + g["fuente"].astype(str) + " · " + g["satelite"].astype(str)
                     + "<br>FRP: " + g["frp"].round(1).astype(str) + " MW"
                     + "<br>Confianza: " + g["confianza"].astype(str))
            fig.add_trace(TraceMapa(lat=g["latitud"], lon=g["longitud"], mode="markers", name=nombre,
                                    text=texto, hovertemplate="%{text}<extra></extra>",
                                    marker=dict(size=5, color=color, opacity=0.8)))

    if region in regiones:  # región seleccionada: relleno suave + borde blanco
        feat = {"type": "Feature", "properties": {}, "geometry": regiones[region]["dibujo"]}
        capas.append(dict(below="traces", sourcetype="geojson", source=feat, type="fill",
                          color="#ffffff", opacity=0.06))
        capas.append(dict(below="traces", sourcetype="geojson", source=feat, type="line",
                          color="white", line=dict(width=2.5)))

    # Nombres de ciudades y localidades (encima de todo)
    capas.append(dict(sourcetype="raster", source=[ESRI_ETIQUETAS], opacity=0.95))

    centro, zoom = vista_mapa(altura)
    anotaciones = [dict(x=0.995, y=0.01, xref="paper", yref="paper", text="Imagen y etiquetas: Esri",
                        showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=10, color="#cfd8ea"))]
    if region != TODO_CHILE:
        anotaciones.append(dict(x=0.5, y=0.97, xref="paper", yref="paper", text=f"<b>{region}</b>",
                                showarrow=False, yanchor="top", font=dict(size=15, color="white"),
                                bgcolor="rgba(8,17,31,0.85)", borderpad=6))
    fig.update_layout(
        height=altura, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=13),
        legend=dict(orientation="h", x=0.012, y=0.03, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(8,17,31,0.85)", itemsizing="constant"),
        annotations=anotaciones,
        **{CLAVE_MAPA: dict(style="white-bg", center=centro, zoom=zoom, layers=capas)},
    )
    with st.container(key="mapbox"):
        st.plotly_chart(fig, config={"scrollZoom": True, "displayModeBar": False}, **STRETCH)


def bloque_mapa(altura=470):
    """Interruptores de capas + leyenda de polígonos + mapa."""
    t1, t2, t3 = st.columns([1.5, 1.2, 3])
    ver_pol = t1.toggle("Polígonos de actividad", value=True, key="ver_pol", disabled=poligonos_all is None)
    ver_foc = t2.toggle("Focos de calor", value=True, key="ver_foc")
    if ver_pol and NIVELES:
        cols = colores_niveles(len(NIVELES))
        chips = "".join(f'<span><i class="dot" style="background:{c}"></i>{n.capitalize()}</span>'
                        for n, c in zip(NIVELES, cols))
        t3.markdown(H(f'<div class="leyenda"><b>Nivel de actividad:</b>{chips}</div>'), unsafe_allow_html=True)
    render_mapa(altura, ver_pol, ver_foc)


def render_panel():
    st.markdown(H(f"""<div class="panel"><div class="ptitle">📊 Resumen de la selección</div>
      <div class="row"><div>📍</div><div>Región:<br><b>{nombre_region}</b></div></div>
      <div class="row"><div>🗓️</div><div>Período:<br><b>{ini:%d-%m-%Y} – {fin_dia:%d-%m-%Y}</b></div></div>
      <div class="row"><div>🔥</div><div>Meses con mayor actividad:<br><b>{txt_meses}</b></div></div>
      <div class="note"><div>ℹ️</div><div>{fmt(total)} focos detectados en {nombre_region}
      entre el {ini:%d-%m-%Y} y el {fin_dia:%d-%m-%Y} (fuente: {fuente.lower()}).</div></div></div>"""),
                unsafe_allow_html=True)


def card(icono, titulo, fig, extra=""):
    with st.container(border=True):
        st.markdown(f'<div class="ctitle"><span class="box">{icono}</span>{titulo}</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, config={"displayModeBar": False}, **STRETCH)
        if extra:
            st.markdown(extra, unsafe_allow_html=True)


def render_graficos():
    # Fila 1: solo el calendario de actividad (a todo el ancho)
    barra = H("""<div style="display:flex;align-items:center;gap:10px;color:#c9d4ec;font-size:13px">
             <span>Menor actividad</span><div class="gradbar"></div><span>Mayor actividad</span></div>""")
    card("📅", f"Calendario de actividad ({desde}–{hasta})", fig_calendario(380), barra)

    # Fila 2: actividad mensual y anual en paralelo
    g1, g2 = st.columns(2)
    with g1:
        card("📊", "Actividad mensual", fig_mensual())
    with g2:
        card("📈", "Actividad anual vs. promedio histórico", fig_anual())


def render_footer():
    f = st.columns([4.2, 1.3, 1.5])
    f[0].markdown(H(f"""<div class="footer"><div class="ok">✓</div>
        <span>MODIS + VIIRS integrados en una serie continua</span><div class="vbar"></div>
        <span>Datos desde {y_min} hasta la actualidad</span></div>"""), unsafe_allow_html=True)
    f[1].download_button("Descargar CSV filtrado", csv_filtrado(region, ini, fin, fuente),
                         file_name="incendios_filtrado.csv", mime="text/csv", icon=":material/download:", **STRETCH)
    figs = [fig_calendario(), fig_mensual(), fig_anual()]
    html = "".join(fg.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                   for i, fg in enumerate(figs))
    f[2].download_button("Exportar visualizaciones", html.encode(), file_name="visualizaciones.html",
                         mime="text/html", icon=":material/bar_chart:", key="exp", **STRETCH)


# ═════════════════════════ Páginas ═════════════════════════
if pagina == "Resumen":
    render_kpis()
    m1, m2 = st.columns([3.1, 1])
    with m1:
        bloque_mapa(470)
    with m2:
        render_panel()
    render_graficos()
    render_footer()
elif pagina == "Mapa de actividad":
    bloque_mapa(700)
elif pagina == "Calendario":
    card("📅", f"Calendario de actividad ({desde}–{hasta})", fig_calendario(420))
elif pagina == "Estadísticas":
    render_kpis()
    e1, e2 = st.columns(2)
    with e1:
        card("📊", "Actividad mensual", fig_mensual())
    with e2:
        card("📈", "Actividad anual vs. promedio histórico", fig_anual())
else:
    st.caption(f"Se muestran las primeras 5.000 filas de {fmt(total)}. Usa «Descargar CSV filtrado» para obtener más.")
    st.dataframe(df.drop(columns=["año", "mes"]).head(5000), height=520, **STRETCH)
    render_footer()
