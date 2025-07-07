#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mapas_no2_espana.py

Genera el mapa de distribución espacial de NO₂ satelital en España
para el TFG “Análisis de Datos Satelitales para el Estudio Espacio-Temporal
de la Contaminación Atmosférica en España”.

Requisitos:
    Python ≥ 3.9
    pandas, geopandas, matplotlib

Entradas:
    - spain_no2_daily_2019_2024.csv  # Serie diaria satelital media por fecha y píxel

Salidas:
    - plots/mapa_no2_espana.png
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def load_no2_data(csv_path: str) -> gpd.GeoDataFrame:
    """
    Carga el CSV con columnas ['longitude','latitude','no2_satellite'] y
    devuelve un GeoDataFrame con geometría de puntos (EPSG:4326).
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    # Filtrar filas con NO2 válido
    df = df[['longitude', 'latitude', 'no2_satellite']].dropna()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    return gdf

def plot_no2_map(gdf: gpd.GeoDataFrame, out_dir: str):
    """
    Dibuja el mapa de España con puntos coloreados por concentración de NO2.
    Guarda la figura en 'plots/mapa_no2_espana.png'.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibuja la base de España (si se dispone de shapefile, usarlo;
    # aquí se traza la costa mínima aproximada)
    spain_outline = {
        'lon': [-9.5, -8.5, -7, -6, -4, -1.5, 1, 3.5, 3.2, 3, 2.5, 2, 1, 0.5, -1, -2.5, -4, -6, -7.5, -9, -9.5],
        'lat': [42, 43, 43.5, 43.2, 43.5, 43.5, 42.8, 42, 40.5, 39.5, 38, 36, 36.2, 36.5, 36, 36.5, 36.8, 36, 37.5, 38, 42]
    }
    ax.plot(spain_outline['lon'], spain_outline['lat'],
            '-', color='black', linewidth=1.5, alpha=0.5)

    # Plot de puntos satelitales
    gdf.plot(
        column='no2_satellite',
        cmap='YlOrRd',
        markersize=5,
        alpha=0.7,
        legend=True,
        legend_kwds={'label': "NO₂ satelital (μmol/m²)"},
        ax=ax
    )

    ax.set_title('Distribución Espacial de NO₂ Satelital en España', fontsize=14)
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_xlim(-10, 5)
    ax.set_ylim(35, 44)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, 'mapa_no2_espana.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Mapa guardado en: {out_path}")

def main():
    # Ruta al CSV generado en data_processor.py / extract_copernicus_data.py
    csv_path = "spain_no2_daily_2019_2024.csv"
    plots_dir = "plots"

    gdf = load_no2_data(csv_path)
    plot_no2_map(gdf, plots_dir)

if __name__ == "__main__":
    main()
