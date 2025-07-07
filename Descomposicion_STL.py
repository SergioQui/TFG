#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stl_decomposition_pipeline.py

Integración de descomposición STL en el pipeline Cloud-Native de Copernicus Data Space Ecosystem.

Requisitos:
    pip install pandas statsmodels

Uso (en el pipeline):
    python stl_decomposition_pipeline.py --input raw_tfg_data.csv --output stl_components.csv
"""

import argparse
import pandas as pd
from statsmodels.tsa.seasonal import STL

def cargar_y_agregar(input_csv):
    """
    Carga el CSV con la estructura TFG y agrega media diaria de no2_satellite.
    """
    df = pd.read_csv(input_csv, parse_dates=['date'])
    # Agrupación diaria de la columna no2_satellite
    daily_no2 = (
        df
        .groupby('date')['no2_satellite']
        .mean()
        .rename('mean_no2')
        .asfreq('D')
    )
    return daily_no2

def aplicar_stl(serie, period=365):
    """
    Ajusta la descomposición STL y devuelve los componentes.
    """
    stl = STL(serie, period=period, robust=True)
    result = stl.fit()
    comp = pd.DataFrame({
        'observed': result.observed,
        'trend':    result.trend,
        'seasonal': result.seasonal,
        'resid':    result.resid
    })
    return comp

def main(input_csv, output_csv):
    # 1. Cargar y agregar datos
    daily_series = cargar_y_agregar(input_csv)

    # 2. Aplicar STL con estacionalidad anual (365 días)
    components = aplicar_stl(daily_series, period=365)

    # 3. Guardar componentes en CSV
    components.to_csv(output_csv, index_label='date')

    # 4. Mensaje de confirmación
    print(f"STL components saved to: {output_csv}")
    print("Ejemplo de primeros registros:")
    print(components.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descomposición STL integrada en el pipeline de Copernicus"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help="Ruta al CSV con los datos estructurados según TFG"
    )
    parser.add_argument(
        '--output', '-o',
        default='stl_components.csv',
        help="Ruta de salida para los componentes STL"
    )
    args = parser.parse_args()
    main(args.input, args.output)
