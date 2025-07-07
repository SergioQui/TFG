#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analisis_exploratorio.py

Análisis Exploratorio de Datos para el TFG “Análisis de Datos Satelitales
para el Estudio Espacio-Temporal de la Contaminación Atmosférica en España”.

Requisitos:
    Python ≥ 3.9
    pandas, numpy, matplotlib, seaborn, geopandas

Entradas:
    data_for_modeling.csv  # Salida del preprocesamiento (fase 2.2)

Salidas:
    - plots/descriptive_stats.png
    - plots/spatial_distribution.png
    - plots/temporal_patterns.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def plot_descriptive_stats(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    sat = df['no2_satellite'].dropna()
    ground = df['no2_ground'].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histograma NO₂ satelital
    sns.histplot(sat, bins=50, kde=True, color='steelblue', ax=axes[0])
    axes[0].set_title('NO₂ Satelital\nMedia: {:.2f}, σ: {:.2f}'.format(sat.mean(), sat.std()))
    axes[0].set_xlabel('μmol/m²')
    axes[0].set_ylabel('Densidad')

    # Histograma NO₂ terrestre
    sns.histplot(ground, bins=50, kde=True, color='seagreen', ax=axes[1])
    axes[1].set_title('NO₂ Terrestre\nMedia: {:.2f}, σ: {:.2f}'.format(ground.mean(), ground.std()))
    axes[1].set_xlabel('μg/m³')
    axes[1].set_ylabel('Densidad')

    plt.tight_layout()
    plt.savefig(f"{out_dir}/descriptive_stats.png", dpi=300)
    plt.close()

def plot_spatial_distribution(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Crear GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df.dropna(subset=['longitude', 'latitude', 'no2_satellite']),
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(column='no2_satellite', cmap='viridis', markersize=5, alpha=0.6, legend=True, ax=ax)
    ax.set_title('Distribución Espacial de NO₂ Satelital')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/spatial_distribution.png", dpi=300)
    plt.close()

def plot_temporal_patterns(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = df.groupby('date')['no2_satellite'].mean().dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts.index, ts.values, color='tab:blue', linewidth=1)
    ax.set_title('Serie Temporal NO₂ Satelital (media diaria)')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('μmol/m²')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/temporal_patterns.png", dpi=300)
    plt.close()

def main():
    data_path = "data_for_modeling.csv"
    plots_dir = "plots"

    df = load_data(data_path)
    plot_descriptive_stats(df, plots_dir)
    plot_spatial_distribution(df, plots_dir)
    plot_temporal_patterns(df, plots_dir)

    print("Análisis exploratorio completado. Gráficos en:", plots_dir)

if __name__ == "__main__":
    main()
