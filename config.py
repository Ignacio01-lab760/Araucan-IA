from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
MODEL_DIR = PKG_DIR / "models"

HOTSPOTS_CSV = DATA_DIR / "detecciones_vegetacion_limpias.csv"

# Modelo basado en histórico VIIRS
HIST_MODEL_PATH = MODEL_DIR / "risk_model_hist.joblib"
HIST_META_PATH = MODEL_DIR / "risk_model_hist.json"
HIST_BACKTEST_PATH = MODEL_DIR / "risk_model_hist_backtest.joblib"   # entrenado solo hasta 2022
BACKTEST_START = "2023-01-01"

# Modelo con meteorología NASA POWER (opcional: copia aquí lo que generaste en FireWatch_Chile)
WX_DIR = DATA_DIR / "weather"
HISTW_MODEL_PATH = MODEL_DIR / "risk_model_wx.joblib"
HISTW_BACKTEST_PATH = MODEL_DIR / "risk_model_wx_backtest.joblib"
HISTW_META_PATH = MODEL_DIR / "risk_model_wx.json"
