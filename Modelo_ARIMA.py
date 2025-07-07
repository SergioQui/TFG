#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arima_model_pipeline.py

Ajuste y pronóstico ARIMA sobre serie diaria de NO₂ satelital
en el pipeline Cloud-Native de Copernicus.

Uso:
    python arima_model_pipeline.py \
        --input raw_tfg_data.csv \
        --output arima_forecast.csv \
        --order p d q

Requisitos:
    pip install pandas statsmodels matplotlib
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA

def load_and_aggregate(input_csv):
    # Carga y promedio diario
    df = pd.read_csv(input_csv, parse_dates=['date'])
    daily = df.groupby('date')['no2_satellite'].mean().asfreq('D')
    return daily

def test_stationarity(series):
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

def plot_acf_pacf(series, lags=30):
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].stem(acf(series.dropna(), nlags=lags))
    axes[0].set_title('ACF')
    axes[1].stem(pacf(series.dropna(), nlags=lags))
    axes[1].set_title('PACF')
    plt.tight_layout()
    plt.savefig('acf_pacf.png')
    plt.close()

def fit_arima(series, order):
    model = ARIMA(series, order=order)
    res = model.fit()
    print(res.summary())
    return res

def forecast_and_save(res, steps, output_csv):
    fc = res.get_forecast(steps=steps)
    df_fc = pd.DataFrame({
        'forecast': fc.predicted_mean,
        'lower_ci': fc.conf_int().iloc[:,0],
        'upper_ci': fc.conf_int().iloc[:,1]
    })
    df_fc.to_csv(output_csv, index_label='date')
    print(f"Forecast saved to {output_csv}")

def main(input_csv, output_csv, order):
    series = load_and_aggregate(input_csv)
    print("## Test de estacionariedad")
    test_stationarity(series)
    print("## Gráficos ACF y PACF")
    plot_acf_pacf(series)
    print(f"## Ajustando ARIMA{order}")
    res = fit_arima(series, order)
    print("## Pronóstico 30 días")
    forecast_and_save(res, steps=30, output_csv=output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modelo ARIMA sobre NO₂ satelital en pipeline Copernicus"
    )
    parser.add_argument('--input', '-i', required=True,
                        help="CSV de datos RAW estructurado según TFG")
    parser.add_argument('--output', '-o', default='arima_forecast.csv',
                        help="Salida de pronósticos")
    parser.add_argument('--order', '-r', type=int, nargs=3, metavar=('p','d','q'),
                        default=[1,1,1],
                        help="Orden ARIMA (p d q)")
    args = parser.parse_args()
    main(args.input, args.output, tuple(args.order))
