#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validacion_glm_gam.py

Validación comparativa de modelos GLM y GAM para NO₂ satelital
vs observaciones terrestres en el TFG “Análisis de Datos Satelitales
para el Estudio Espacio-Temporal de la Contaminación Atmosférica en España”.

Requisitos:
    Python ≥ 3.9
    pandas, numpy, statsmodels, pygam, scikit-learn

Entradas:
    - spain_no2_daily_2019_2024.csv    # Serie diaria satélite
    - ground_no2_daily_2019_2024.csv   # Serie diaria terrestre emparejada

Salidas:
    - glm_vs_gam_scatter.png
    - glm_gam_metrics_comparison.csv
    - validation_by_station_type.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from pygam import LinearGAM, s
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

def load_and_merge(sat_path: str, ground_path: str) -> pd.DataFrame:
    """Carga y empareja series diarias satélite y terrestre."""
    df_sat = pd.read_csv(sat_path, parse_dates=["date"])
    df_sat.rename(columns={"no2_satellite": "no2_sat"}, inplace=True)
    df_gnd = pd.read_csv(ground_path, parse_dates=["date"])
    df_gnd.rename(columns={"no2_concentration": "no2_obs"}, inplace=True)
    df = pd.merge(df_sat, df_gnd, on="date", how="inner")
    df.dropna(subset=["no2_sat", "no2_obs"], inplace=True)
    df["weekday"] = df["date"].dt.weekday + 1
    return df

def fit_glm(df: pd.DataFrame) -> glm:
    """Ajusta modelo GLM Gaussian."""
    formula = (
        "no2_sat ~ no2_obs + C(weekday)"
    )
    model = glm(formula, data=df, family=Gaussian()).fit()
    return model

def fit_gam(df: pd.DataFrame) -> LinearGAM:
    """Ajusta modelo GAM con splines en no2_obs."""
    X = df[["no2_obs"]].values
    gam = LinearGAM(s(0)).fit(X, df["no2_sat"].values)
    return gam

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas RMSE, MAE, R2 y bias."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}

def scatter_plot(df: pd.DataFrame, y_glm: np.ndarray, y_gam: np.ndarray, out_png: str):
    """Diagrama de dispersión comparativo GLM vs GAM."""
    plt.figure()
    plt.scatter(df["no2_sat"], df["no2_obs"], s=10, color="gray", alpha=0.5, label="Observado")
    plt.scatter(df["no2_sat"], y_glm, s=10, color="blue", alpha=0.6, label="GLM predicho")
    plt.scatter(df["no2_sat"], y_gam, s=10, color="green", alpha=0.6, label="GAM predicho")
    mn = min(df["no2_sat"].min(), df["no2_obs"].min())
    mx = max(df["no2_sat"].max(), df["no2_obs"].max())
    plt.plot([mn, mx], [mn, mx], "k--", lw=1, label="1:1")
    plt.xlabel("NO₂ Satelital (μmol/m²)")
    plt.ylabel("NO₂ Terrestre (μg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    sat_csv = "spain_no2_daily_2019_2024.csv"
    ground_csv = "ground_no2_daily_2019_2024.csv"

    df = load_and_merge(sat_csv, ground_csv)

    glm_model = fit_glm(df)
    df["glm_pred"] = glm_model.predict(df)

    gam_model = fit_gam(df)
    df["gam_pred"] = gam_model.predict(df[["no2_obs"]].values)

    # Métricas
    glm_metrics = compute_metrics(df["no2_sat"].values, df["glm_pred"].values)
    gam_metrics = compute_metrics(df["no2_sat"].values, df["gam_pred"].values)
    metrics_df = pd.DataFrame([{"Model": "GLM", **glm_metrics},
                               {"Model": "GAM", **gam_metrics}])
    metrics_df.to_csv("glm_gam_metrics_comparison.csv", index=False)

    # Scatter
    scatter_plot(df, df["glm_pred"].values, df["gam_pred"].values, "glm_vs_gam_scatter.png")

    # Métricas por tipo de estación (ej. urbano/suburbano/rural si existiera)
    # Aquí asumimos columna "station_type" en ground CSV
    if "station_type" in df.columns:
        by_type = []
        for stype, grp in df.groupby("station_type"):
            m_glm = compute_metrics(grp["no2_sat"].values, grp["glm_pred"].values)
            m_gam = compute_metrics(grp["no2_sat"].values, grp["gam_pred"].values)
            by_type.append({"station_type": stype,
                            "GLM_R2": m_glm["R2"], "GAM_R2": m_gam["R2"],
                            "GLM_RMSE": m_glm["RMSE"], "GAM_RMSE": m_gam["RMSE"]})
        pd.DataFrame(by_type).to_csv("validation_by_station_type.csv", index=False)

    print("Validación GLM vs GAM completada.")
    print(" - Scatter saved: glm_vs_gam_scatter.png")
    print(" - Metrics saved: glm_gam_metrics_comparison.csv")
    if "station_type" in df.columns:
        print(" - Per-station metrics: validation_by_station_type.csv")

if __name__ == "__main__":
    main()
