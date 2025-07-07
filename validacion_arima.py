#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validacion_arima.py

Validación del modelo ARIMA para NO₂ satelital vs observaciones terrestres
en el TFG “Análisis de Datos Satelitales para el Estudio Espacio-Temporal
de la Contaminación Atmosférica en España”.

Requisitos:
    Python ≥ 3.9
    pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, statsmodels

Entradas:
    - spain_no2_daily_2019_2024.csv    # Serie diaria satélite
    - ground_no2_daily_2019_2024.csv   # Serie diaria terrestre emparejada

Salidas:
    - arima_validation_scatter.png
    - arima_residuals_analysis.png
    - validation_metrics.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Configuración global
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

def load_series(sat_path: str, ground_path: str) -> pd.DataFrame:
    """
    Carga y empareja series diarias satélite y terrestre.
    """
    df_sat = pd.read_csv(sat_path, parse_dates=["date"])
    df_sat.rename(columns={"no2_satellite": "no2_sat"}, inplace=True)
    df_gnd = pd.read_csv(ground_path, parse_dates=["date"])
    df_gnd.rename(columns={"no2_observed": "no2_obs"}, inplace=True)
    df = pd.merge(df_sat, df_gnd, on="date", how="inner")
    df.dropna(subset=["no2_sat", "no2_obs"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def fit_arima_and_predict(series: pd.Series, order: tuple[int,int,int]= (1,1,1)) -> np.ndarray:
    """
    Ajusta ARIMA y devuelve la serie ajustada (in-sample).
    """
    model = ARIMA(series, order=order)
    res = model.fit()
    return res.fittedvalues

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula métricas de validación estándar.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}

def plot_scatter(df: pd.DataFrame, metrics: dict, out_png: str):
    """
    Diagrama de dispersión NO2 satélite vs terrestre con línea 1:1.
    """
    plt.figure()
    plt.scatter(df["no2_sat"], df["no2_obs"], alpha=0.6, s=20, color="tab:blue")
    maxv = max(df["no2_sat"].max(), df["no2_obs"].max())
    plt.plot([0, maxv], [0, maxv], "k--", lw=1)
    plt.title("Validación ARIMA NO₂ satélite vs terrestre")
    plt.xlabel("NO₂ sat (μmol/m²)")
    plt.ylabel("NO₂ obs (μg/m³)")
    txt = f"RMSE={metrics['RMSE']:.2f}\nMAE={metrics['MAE']:.2f}\nR²={metrics['R2']:.3f}\nBias={metrics['Bias']:.2f}"
    plt.text(0.02*maxv, 0.95*maxv, txt, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_residuals(df: pd.DataFrame, out_png: str):
    """
    Análisis de residuales: histograma y QQ-plot.
    """
    residuals = df["no2_sat_pred"] - df["no2_obs"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(residuals, bins=40, kde=True, ax=axs[0], color="skyblue")
    axs[0].set_title("Histograma de residuales")
    axs[0].set_xlabel("Residuales")
    # QQ-plot
    import scipy.stats as st
    ax = axs[1]
    st.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q–Q plot de residuales")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    # Rutas de entrada
    sat_csv    = "spain_no2_daily_2019_2024.csv"
    ground_csv = "ground_no2_daily_2019_2024.csv"
    # Cargar y emparejar
    df = load_series(sat_csv, ground_csv)
    # Ajuste ARIMA y predicción in-sample
    df["no2_sat_pred"] = fit_arima_and_predict(df["no2_sat"], order=(1,1,1))
    # Calcular métricas
    m = compute_metrics(df["no2_obs"].values, df["no2_sat_pred"].values)
    # Guardar métricas
    pd.DataFrame([m]).to_csv("validation_metrics.csv", index=False)
    # Visualizaciones
    plot_scatter(df, m, "arima_validation_scatter.png")
    plot_residuals(df, "arima_residuals_analysis.png")
    print("Validación completada. Métricas y gráficos generados.")

if __name__ == "__main__":
    main()
