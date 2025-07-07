#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualizacion_momentos_extremos.py

Genera dos visualizaciones de momentos extremos alineadas con el TFG:
1) Impacto de COVID-19 en NO₂ para Madrid (media mensual y tendencia).
2) Influencia de episodios de polvo sahariano en PM₂.₅ (diario, media móvil de 30 días, línea base y episodios).

Requisitos:
    Python ≥ 3.9
    pandas, numpy, matplotlib, seaborn

Entradas:
    - spain_no2_daily_2019_2024.csv    # Con columnas date (datetime), no2_satellite, longitude, latitude
    - spain_pm25_daily_2019_2025.csv   # Con columnas date (datetime), pm25_satellite

Salidas:
    - plots/no2_madrid_covid_impact.png
    - plots/pm25_saharan_dust_influence.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

def plot_no2_madrid_covid(df: pd.DataFrame, out_path: str):
    """
    Grafica el impacto COVID-19 en NO₂ para Madrid.
    Calcula la media mensual de no2_satellite para el polígono de Madrid
    y traza observaciones y tendencia lineal.
    """
    # Filtrar datos de Madrid (aprox bbox)
    madrid_bbox = [-3.8, 40.2, -3.5, 40.6]
    df_mad = df[
        (df.longitude.between(madrid_bbox[0], madrid_bbox[2])) &
        (df.latitude.between(madrid_bbox[1], madrid_bbox[3]))
    ].copy()
    df_mad["month"] = df_mad.date.dt.to_period("M")
    monthly = df_mad.groupby("month")["no2_satellite"].mean().rename("mean_no2").to_timestamp()

    # Línea de tendencia
    x = np.arange(len(monthly))
    slope, intercept = np.polyfit(x, monthly.values, 1)
    trend = intercept + slope * x

    plt.figure()
    plt.plot(monthly.index, monthly.values, marker="o", linestyle="-", label="Media mensual NO₂", color="tab:blue")
    plt.plot(monthly.index, trend, linestyle="--", label="Tendencia lineal", color="tab:orange")
    # Resaltar período COVID
    covid_start, covid_end = "2020-03", "2020-05"
    plt.axvspan(pd.to_datetime(covid_start), pd.to_datetime(covid_end) + pd.offsets.MonthEnd(0),
                color="red", alpha=0.2, label="Confinamiento COVID-19")
    plt.title("NO₂ Madrid – Impacto COVID-19 (2020)")
    plt.xlabel("Fecha")
    plt.ylabel("NO₂ (10⁻⁵ mol/m²)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_pm25_saharan(df: pd.DataFrame, out_path: str):
    """
    Grafica la influencia del polvo del Sahara en PM2.5.
    Trazados: observaciones diarias, media móvil 30 días, línea base (percentil 50),
    y fondo coloreado para episodios.
    """
    df = df.set_index("date").sort_index()
    daily = df["pm25_satellite"].resample("D").mean()
    ma30 = daily.rolling(30, center=True, min_periods=15).mean()
    baseline = daily.quantile(0.50)

    # Detectar episodios (> percentile 95)
    p95 = daily.quantile(0.95)
    episodes = daily > p95

    plt.figure()
    plt.plot(daily.index, daily.values, color="tab:cyan", linewidth=0.8, label="PM2.5 diario")
    plt.plot(ma30.index, ma30.values, color="goldenrod", linewidth=1.5, label="Media móvil 30 días")
    plt.hlines(baseline, daily.index.min(), daily.index.max(),
               color="gray", linestyle="--", linewidth=1, label="Línea base (p50)")
    # Sombrear episodios
    for start, group in episodes.astype(int).diff().fillna(0).pipe(
            lambda s: s[s != 0]).pipe(lambda s: zip(s.index, s.values)):
        # group=1 empieza episodio, -1 termina
        pass
    # Más sencillo: relleno continuo donde episodes True
    plt.fill_between(daily.index, 0, daily.values,
                     where=episodes, color="peachpuff", alpha=0.4,
                     label="Episodios saharianos (>p95)")

    plt.title("Influencia del polvo del Sahara en PM2.5")
    plt.xlabel("Fecha")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Cargar datos
    df_no2 = pd.read_csv("spain_no2_daily_2019_2024.csv", parse_dates=["date"])
    df_pm25 = pd.read_csv("spain_pm25_daily_2019_2025.csv", parse_dates=["date"])

    plot_no2_madrid_covid(df_no2, "plots/no2_madrid_covid_impact.png")
    plot_pm25_saharan(df_pm25, "plots/pm25_saharan_dust_influence.png")
