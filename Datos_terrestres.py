#!/usr/bin/env python3
"""
script: descargar_datos_terrestres_miteco.py

Descarga datos de calidad del aire de estaciones terrestres desde la API del
Ministerio para la Transición Ecológica y el Reto Demográfico (MITECO).

Requisitos:
    - Python 3.7+
    - requests
"""

import os
import requests
from datetime import datetime

# ─────────── Parámetros ─────────────────────────────────────────────────────────

OUTPUT_DIR = "datos_terrestres"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Punto de acceso a la API CSV de MITECO
MITECO_API_URL = "https://www.miteco.gob.es/es/calidad-y-evaluacion-ambiental/temas/atmosfera/calidad-del-aire/datos/"

# Reemplaza con tu clave de API proporcionada por MITECO
MITECO_API_KEY = "TU_API_KEY_MITECO"

# Rango de fechas a descargar (YYYY-MM-DD)
FECHA_INICIO = "2019-01-01"
FECHA_FIN    = "2024-12-31"

# Nombre del fichero de salida
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"terrestres_{FECHA_INICIO}_{FECHA_FIN}.csv")


def descargar_datos_terrestres():
    """
    Descarga datos horarios de estaciones terrestres (NO2, PM10, PM2.5, O3, SO2, etc.)
    desde la API CSV de MITECO siguiendo el formato DCF-MITECO.
    """
    params = {
        "api_key": MITECO_API_KEY,
        "start_date": FECHA_INICIO,
        "end_date": FECHA_FIN,
        "formato": "csv"
    }
    print(f"Solicitando datos terrestres de {FECHA_INICIO} a {FECHA_FIN}...")
    resp = requests.get(MITECO_API_URL, params=params)
    resp.raise_for_status()

    with open(OUTPUT_CSV, "wb") as f:
        f.write(resp.content)

    print(f"Datos terrestres guardados en: {OUTPUT_CSV}")
    print(f"Tamaño del archivo: {os.path.getsize(OUTPUT_CSV) / 1024:.2f} KB")
    print(f"Proceso completado el {datetime.utcnow().isoformat()} UTC")


if __name__ == "__main__":
    descargar_datos_terrestres()
