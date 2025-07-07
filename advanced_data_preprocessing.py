"""
advanced_data_preprocessing.py
--------------------------------
Pipeline cloud-native que implementa todo el flujo descrito en la sección «2.2 Preprocesamiento Avanzado de Datos».

REQUISITOS
==========
Python ≥ 3.9 y los paquetes:
    pandas, geopandas, numpy, requests, shapely, rasterio,
    sentinelhub (≥3.11), cdsapi, holidays, tqdm

VARIABLES DE ENTORNO NECESARIAS
--------------------------------
    SH_CLIENT_ID        Credencial OAuth2 de Sentinel Hub / CDSE
    SH_CLIENT_SECRET    Credencial OAuth2 de Sentinel Hub / CDSE

ESTRUCTURA DEL PIPELINE
-----------------------
0. Configuración global y utilidades
1. Extracción cloud-native vía Statistical API
2. Carga de tablas base (cuatro CSVs en /outputs_preprocessing/raw)
3. Preprocesamiento avanzado (filtrado, emparejamiento, enriquecimiento)
4. Ejemplo de uso: genera /outputs_preprocessing/processed/data_for_modeling.csv

Para ejecutar:
    python advanced_data_preprocessing.py
"""

from __future__ import annotations
import os
import json
import datetime as dt
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    BBox,
    CRS,
    bbox_to_dimensions,
    SentinelHubStatistical,
    Geometry,
)

import cdsapi
import holidays

# -----------------------------------------------------------------------------
# 0. Configuración global y utilidades
# -----------------------------------------------------------------------------
WORKDIR = Path("outputs_preprocessing")
RAW_DIR = WORKDIR / "raw"
PROCESSED_DIR = WORKDIR / "processed"
for p in (RAW_DIR, PROCESSED_DIR):
    p.mkdir(parents=True, exist_ok=True)

ES_HOLIDAYS = holidays.country_holidays("ES")

CONFIG = SHConfig()
if not CONFIG.sh_client_id:
    CONFIG.sh_client_id = os.getenv("SH_CLIENT_ID")
    CONFIG.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
if not (CONFIG.sh_client_id and CONFIG.sh_client_secret):
    raise RuntimeError("Credenciales Sentinel Hub/CDSE no encontradas en variables de entorno")


def buffer_bbox(b: Tuple[float, float, float, float], d: float = 0.0):
    minx, miny, maxx, maxy = b
    return [minx - d, miny - d, maxx + d, maxy + d]

# -----------------------------------------------------------------------------
# 1. Ejemplo de extracción cloud-native (NO₂ diario)
# -----------------------------------------------------------------------------

def fetch_no2_time_series(bbox: Tuple[float, float, float, float],
                          start: str,
                          end: str,
                          aggregation: str = "P1D") -> pd.DataFrame:
    """Obtiene media diaria de NO₂ troposférico (μmol/m²) mediante Statistical API."""
    geometry = Geometry(buffer_bbox(bbox), CRS.WGS84)

    request = SentinelHubStatistical(
        aggregation={
            "timeRange": {"from": start, "to": end},
            "aggregationInterval": {"of": aggregation},
            "reducers": [{"name": "mean"}]
        },
        input_data=[{
            "dataCollection": DataCollection.SENTINEL5P_L2,
            "identifier": "S5P_NO2",
            "bandNames": ["L2__NO2___"],
            "processing": {
                "filter": {"name": "qa_value", "value": 0.75, "operation": ">="}
            }
        }],
        geometry=geometry,
        config=CONFIG)

    records = []
    for item in request.get_data()[0]["data"]:
        records.append({
            "date": item["intervalFrom"].split("T")[0],
            "no2_satellite": item["outputs"]["S5P_NO2_mean"]
        })
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# 2. Carga de las cuatro tablas base
# -----------------------------------------------------------------------------

def load_satellite_pixels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_ground_stations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.rename(columns={"timestamp": "date"}, inplace=True)
    return df

# -----------------------------------------------------------------------------
# 3. Pipeline de preprocesamiento avanzado
# -----------------------------------------------------------------------------

def preprocess_pipeline(sat_df: pd.DataFrame,
                        ground_df: pd.DataFrame,
                        aux_geo: pd.DataFrame,
                        era5_df: pd.DataFrame) -> pd.DataFrame:
    """Implementa todas las etapas 2.2.1-2.2.5"""
    # 3.1 Filtrado de calidad
    sat_df = sat_df[(sat_df["qa_value"] >= 0.75) & (sat_df["cloud_fraction"] <= 0.3)].copy()
    ground_df = ground_df[ground_df["data_quality"] == "valido"].copy()

    # 3.2 Emparejamiento espaciotemporal
    sat_gdf = gpd.GeoDataFrame(
        sat_df,
        geometry=gpd.points_from_xy(sat_df.longitude, sat_df.latitude), crs="EPSG:4326")
    ground_gdf = gpd.GeoDataFrame(
        ground_df,
        geometry=gpd.points_from_xy(ground_df.longitude, ground_df.latitude), crs="EPSG:4326")

    sat_sindex = sat_gdf.sindex
    records = []
    for _, g in tqdm(ground_gdf.iterrows(), total=len(ground_gdf), desc="Join espaciotemporal"):
        buffer = g.geometry.buffer(0.05)  # ~5 km
        idx = list(sat_sindex.intersection(buffer.bounds))
        if not idx:
            continue
        cand = sat_gdf.iloc[idx]
        cand = cand[(cand.date >= g.date - pd.Timedelta(hours=1)) &
                    (cand.date <= g.date + pd.Timedelta(hours=1))]
        if cand.empty:
            continue
        cand["dist"] = cand.distance(g.geometry)
        s = cand.nsmallest(1, "dist").iloc[0]
        records.append({
            "date": g.date.date(),
            "longitude": g.longitude,
            "latitude": g.latitude,
            "no2_satellite": s.no2_tropospheric_column,
            "no2_ground": g.no2_concentration
        })
    df = pd.DataFrame(records)

    # 3.3 Enriquecimiento contextual
    df = df.merge(aux_geo, on=["longitude", "latitude"], how="left")
    df = df.merge(era5_df, on=["date", "longitude", "latitude"], how="left")

    # 3.4 Variables temporales y eventos
    df["weekday"] = pd.to_datetime(df.date).dt.weekday + 1
    df["is_holiday"] = df.date.apply(lambda d: d in ES_HOLIDAYS)
    covid_start, covid_end = dt.date(2020, 3, 14), dt.date(2020, 5, 21)
    df["covid_lockdown"] = df.date.between(covid_start, covid_end)
    df["saharan_dust_episode"] = False  # placeholder; sustituir por lógica real

    # 3.5 Limpieza final
    df.dropna(subset=["no2_satellite", "no2_ground"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index.name = "id"
    return df.reset_index()

# -----------------------------------------------------------------------------
# 4. Ejecución de ejemplo
# -----------------------------------------------------------------------------

def main():
    sat_path = RAW_DIR / "satellite_pixels.csv"
    ground_path = RAW_DIR / "ground_stations.csv"
    aux_path = RAW_DIR / "aux_geo.csv"
    era5_path = RAW_DIR / "era5.csv"
    if not all(p.exists() for p in (sat_path, ground_path, aux_path, era5_path)):
        print("Faltan archivos de entrada en 'outputs_preprocessing/raw'. Abortando ejemplo.")
        return
    sat_df = load_satellite_pixels(sat_path)
    ground_df = load_ground_stations(ground_path)
    aux_geo = pd.read_csv(aux_path)
    era5_df = pd.read_csv(era5_path, parse_dates=["date"])

    final_df = preprocess_pipeline(sat_df, ground_df, aux_geo, era5_df)
    out_csv = PROCESSED_DIR / "data_for_modeling.csv"
    final_df.to_csv(out_csv, index=False)
    print("Tabla final guardada en", out_csv)

if __name__ == "__main__":
    main()
