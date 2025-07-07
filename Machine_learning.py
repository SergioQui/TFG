#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_models_no2.py

Entrenamiento y evaluación de RandomForest y XGBoost
para la predicción de NO₂ satelital diario.

Uso:
    python ml_models_no2.py \
      --input raw_tfg_data.csv \
      --output metrics_ml.csv

Requisitos:
    pip install pandas scikit-learn xgboost
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=['date'])
    agg = df.groupby('date').agg({
        'no2_satellite':'mean',
        'temperature':'mean',
        'wind_speed':'mean',
        'boundary_layer_height':'mean',
        'aod_550nm':'mean',
        'elevation':'mean',
        'distance_to_coast':'mean',
        'population_density':'mean',
        'is_holiday':'max',
        'covid_lockdown':'max',
        'saharan_dust_episode':'max'
    }).reset_index()
    # One-hot weekday
    df_wd = pd.get_dummies(df.groupby('date')['weekday'].first(), prefix='wd')
    agg = agg.join(df_wd, on='date')
    return agg.dropna().set_index('date')

def evaluate_model(X, y, model, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    rmses, maes, r2s = [], [], []
    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmses.append(np.sqrt(mean_squared_error(y_te, pred)))
        maes.append(mean_absolute_error(y_te, pred))
        r2s.append(r2_score(y_te, pred))
    return {
        'RMSE': np.mean(rmses),
        'MAE': np.mean(maes),
        'R2': np.mean(r2s)
    }

def main(args):
    df = load_and_prepare(args.input)
    y = df['no2_satellite']
    X = df.drop(columns=['no2_satellite'])
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    
    metrics = []
    for name, model in [('RandomForest', rf), ('XGBoost', xgb)]:
        m = evaluate_model(X, y, model)
        m['Model'] = name
        metrics.append(m)
    
    pd.DataFrame(metrics)[['Model','RMSE','MAE','R2']].to_csv(
        args.output, index=False)
    print("Métricas guardadas en", args.output)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help="CSV raw TFG")
    p.add_argument('--output', '-o', default='metrics_ml.csv',
                   help="CSV métricas ML")
    args = p.parse_args()
    main(args)
