#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
regresion_multivariante.py

Ajuste de GLM y GAM para media diaria de NO₂ satelital y validación de métricas
en el pipeline Copernicus.

Uso:
    python regresion_multivariante.py \
        --input raw_tfg_data.csv \
        --glm_output glm_results.txt \
        --gam_output gam_results.csv \
        --metrics_output metrics_summary.csv

Requisitos:
    pip install pandas statsmodels pygam scikit-learn numpy
"""

import argparse
import pandas as pd
import numpy as np
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from pygam import LinearGAM, s
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_aggregate(input_csv):
    df = pd.read_csv(input_csv, parse_dates=['date'])
    daily = df.groupby('date').agg({
        'no2_satellite':'mean',
        'temperature':'mean',
        'wind_speed':'mean',
        'boundary_layer_height':'mean',
        'weekday':'first',
        'is_holiday':'first',
        'day_of_year':'first',
        'elevation':'mean',
        'distance_to_coast':'mean',
        'aod_550nm':'mean'
    }).dropna().reset_index()
    return daily

def fit_glm(df, output_txt):
    formula = (
        'no2_satellite ~ temperature + wind_speed + boundary_layer_height '
        '+ C(weekday) + is_holiday + day_of_year '
        '+ elevation + distance_to_coast + aod_550nm'
    )
    model = glm(formula=formula, data=df, family=Gaussian()).fit()
    with open(output_txt, 'w') as f:
        f.write(model.summary().as_text())
    return model

def fit_gam(df, output_csv):
    X = df[['temperature','wind_speed','boundary_layer_height',
            'day_of_year']].values
    # add dummy vars for weekday and is_holiday
    dummies = pd.get_dummies(df['weekday'], prefix='wd')
    X = np.hstack([X, dummies.values, df[['is_holiday']].astype(int).values])
    y = df['no2_satellite'].values
    gam = LinearGAM(
        s(0) + s(1) + s(2) + s(3)  # splines for four continuous vars
    ).fit(X, y)
    pd.DataFrame({
        'term': [str(term) for term in gam.terms],
        'coef': gam.coef_
    }).to_csv(output_csv, index=False)
    return gam

def rolling_window_metrics(df, model_type, order=None, gam_spline_idx=None):
    """
    Realiza validación rolling-window y devuelve las métricas promedio.
    model_type: 'GLM' o 'GAM'
    """
    window_train = 365 * 3  # entrenar 3 años
    horizon = 365           # predecir 1 año
    mses, maes, r2s, biases = [], [], [], []
    dates = df['date'].values
    for start in range(0, len(df) - window_train - horizon + 1, 365):
        train = df.iloc[start:start+window_train]
        test = df.iloc[start+window_train:start+window_train+horizon]
        if model_type == 'GLM':
            mdl = fit_glm(train, '/dev/null')
            y_pred = mdl.predict(test)
        else:
            # preparar X,y similar a fit_gam
            X_train = train[['temperature','wind_speed','boundary_layer_height',
                             'day_of_year']].values
            dummies_t = pd.get_dummies(train['weekday'], prefix='wd')
            X_train = np.hstack([X_train, dummies_t.values, train[['is_holiday']].astype(int).values])
            y_train = train['no2_satellite'].values
            gam = LinearGAM(gam_spline_idx).fit(X_train, y_train)

            X_test = test[['temperature','wind_speed','boundary_layer_height',
                           'day_of_year']].values
            dummies_tt = pd.get_dummies(test['weekday'], prefix='wd')
            X_test = np.hstack([X_test, dummies_tt.reindex(columns=dummies_t.columns, fill_value=0).values,
                                test[['is_holiday']].astype(int).values])
            y_pred = gam.predict(X_test)
        y_true = test['no2_satellite'].values

        mses.append(mean_squared_error(y_true, y_pred))
        maes.append(mean_absolute_error(y_true, y_pred))
        r2s.append(r2_score(y_true, y_pred))
        biases.append((y_pred - y_true).mean())

    return {
        'RMSE': float(np.sqrt(np.mean(mses))),
        'MAE': float(np.mean(maes)),
        'R2': float(np.mean(r2s)),
        'Bias': float(np.mean(biases))
    }

def main(args):
    df = load_and_aggregate(args.input)

    # 1. Ajuste de modelos
    glm_model = fit_glm(df, args.glm_output)
    gam_model = fit_gam(df, args.gam_output)

    # 2. Validación y métricas
    glm_metrics = rolling_window_metrics(df, model_type='GLM')
    gam_metrics = rolling_window_metrics(
        df, model_type='GAM',
        gam_spline_idx=(s(0) + s(1) + s(2) + s(3))
    )

    # 3. Guardar métricas
    metrics_df = pd.DataFrame([
        {'Model': 'GLM', **glm_metrics},
        {'Model': 'GAM', **gam_metrics}
    ])
    metrics_df.to_csv(args.metrics_output, index=False)
    print(f"Métricas guardadas en {args.metrics_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regresión Multivariante con validación y métricas"
    )
    parser.add_argument('--input', '-i', required=True,
                        help="CSV de datos RAW estructurado según TFG")
    parser.add_argument('--glm_output', default='glm_results.txt',
                        help="Archivo de salida resumen GLM")
    parser.add_argument('--gam_output', default='gam_results.csv',
                        help="Archivo de salida coeficientes GAM")
    parser.add_argument('--metrics_output', default='metrics_summary.csv',
                        help="CSV de métricas de validación")
    args = parser.parse_args()
    main(args)
