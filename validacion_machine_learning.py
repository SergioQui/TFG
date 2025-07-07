#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validacion_machine_learning.py

Validación comparativa de modelos Random Forest y XGBoost para NO₂ satelital vs observaciones
terrestres en el TFG “Análisis de Datos Satelitales para el Estudio Espacio-Temporal
de la Contaminación Atmosférica en España”.

Este script utiliza datos reales extraídos de la API de Copernicus (Sentinel-5P Statistical API)
y mediciones de estaciones terrestres emparejadas espacio-temporalmente.
No genera datos sintéticos.

Requisitos:
    Python ≥ 3.9
    pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

Entradas:
    - spain_no2_daily_2019_2024.csv    # Serie diaria de NO₂ satelital (μmol/m²)
    - ground_no2_daily_2019_2024.csv   # Serie diaria de NO₂ terrestre (μg/m³), emparejada

Salidas:
    - rf_vs_xgb_scatter.png
    - rf_xgb_metrics_comparison.csv
    - rf_xgb_station_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

def load_and_merge(sat_path: str, ground_path: str) -> pd.DataFrame:
    """
    Carga las series diarias satélite y terrestre, las empareja por fecha,
    y elimina pares con datos faltantes.
    """
    df_sat = pd.read_csv(sat_path, parse_dates=["date"])
    df_sat.rename(columns={"no2_satellite": "no2_sat"}, inplace=True)
    df_gnd = pd.read_csv(ground_path, parse_dates=["date"])
    df_gnd.rename(columns={"no2_concentration": "no2_obs"}, inplace=True)
    df = pd.merge(df_sat, df_gnd, on="date", how="inner")
    df.dropna(subset=["no2_sat", "no2_obs"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Construye matriz de predictores y vector objetivo.
    Incluye:
      - no2_sat como objetivo
      - no2_obs como predictor principal
      - variables temporales derivadas (weekday, month)
    """
    df["weekday"] = df["date"].dt.weekday + 1
    df["month"] = df["date"].dt.month
    X = df[["no2_obs", "weekday", "month"]]
    # One-hot encoding para weekday y month
    X = pd.get_dummies(X, columns=["weekday", "month"], drop_first=True)
    y = df["no2_sat"]
    return X, y

def evaluate_models(X: pd.DataFrame, y: pd.Series):
    """
    Ajusta Random Forest y XGBoost en rolling time-series splits
    y devuelve métricas promedio.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    rf, xgb = RandomForestRegressor(n_estimators=100, random_state=42), \
              XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    metrics = {"RandomForest": [], "XGBoost": []}

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        rf.fit(X_tr, y_tr)
        y_rf = rf.predict(X_te)
        xgb.fit(X_tr, y_tr)
        y_xgb = xgb.predict(X_te)

        for name, y_pred in [("RandomForest", y_rf), ("XGBoost", y_xgb)]:
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            mae  = mean_absolute_error(y_te, y_pred)
            r2   = r2_score(y_te, y_pred)
            metrics[name].append({"RMSE": rmse, "MAE": mae, "R2": r2})

    # Promedio de métricas
    summary = {}
    for model, vals in metrics.items():
        dfm = pd.DataFrame(vals)
        summary[model] = {
            "RMSE": dfm["RMSE"].mean(),
            "MAE": dfm["MAE"].mean(),
            "R2":  dfm["R2"].mean()
        }
    return summary, rf, xgb

def plot_comparison(df: pd.DataFrame, rf_model, xgb_model, out_png: str):
    """
    Diagrama de dispersión NO₂ sat vs predicciones RF y XGB con línea 1:1.
    """
    X, y = prepare_features(df)
    df_pred = df.copy()
    df_pred["rf_pred"]  = rf_model.predict(X)
    df_pred["xgb_pred"] = xgb_model.predict(X)

    plt.figure()
    plt.scatter(df_pred["no2_sat"], df_pred["rf_pred"], 
                alpha=0.5, s=20, label="Random Forest")
    plt.scatter(df_pred["no2_sat"], df_pred["xgb_pred"], 
                alpha=0.5, s=20, label="XGBoost")
    maxv = max(df_pred["no2_sat"].max(), df_pred[["rf_pred","xgb_pred"]].max().max())
    plt.plot([0, maxv], [0, maxv], "k--", lw=1)
    plt.title("NO₂ Satelital vs Predicciones RF & XGBoost")
    plt.xlabel("NO₂ sat (μmol/m²)")
    plt.ylabel("Predicción (μmol/m²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def save_metrics(summary: dict, out_csv: str):
    """
    Guarda métricas comparativas en CSV.
    """
    dfm = pd.DataFrame([
        {"Model": m, **metrics} for m, metrics in summary.items()
    ])
    dfm.to_csv(out_csv, index=False)

def main():
    sat_csv    = "spain_no2_daily_2019_2024.csv"
    ground_csv = "ground_no2_daily_2019_2024.csv"

    # 1. Carga y emparejamiento
    df = load_and_merge(sat_csv, ground_csv)

    # 2. Preparación de datos
    X, y = prepare_features(df)

    # 3. Evaluación de modelos
    summary, rf_model, xgb_model = evaluate_models(X, y)

    # 4. Guardar métricas
    save_metrics(summary, "rf_xgb_metrics_comparison.csv")

    # 5. Visualización comparativa
    plot_comparison(df, rf_model, xgb_model, "rf_vs_xgb_scatter.png")

    print("Validación ML completada.")
    print(" - Métricas guardadas en 'rf_xgb_metrics_comparison.csv'")
    print(" - Gráfico guardado en 'rf_vs_xgb_scatter.png'")

if __name__ == "__main__":
    main()
