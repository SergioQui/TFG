# data_processor.py

import requests
import json
import pandas as pd
from auth_manager import get_authenticated_session
from config import PROCESSING_API_URL
from datetime import datetime, timedelta

# --- Función para realizar solicitudes a la Processing API ---

def process_sentinel_data(oauth_session, evalscript, time_interval, geometry=None, bbox=None, output_format="application/json"):
    """
    Realiza una solicitud a la Processing API de Sentinel Hub para datos de contaminación atmosférica.
    
    Args:
        oauth_session: La sesión OAuth2 autenticada.
        evalscript (str): El evalscript (código JavaScript) para el procesamiento.
        time_interval (list): Un rango de tiempo [start_date, end_date] en formato ISO 8601.
        geometry (dict, optional): Un objeto GeoJSON de tipo Polygon para recortar y/o agregar datos.
        bbox (list, optional): Un bounding box [min_lon, min_lat, max_lon, max_lat].
        output_format (str): Formato de salida deseado (por defecto JSON para datos estadísticos).
    
    Returns:
        requests.Response: El objeto de respuesta de la API.
    """
    payload = {
        "input": {
            "bounds": {},  # Se rellenará con bbox o geometry
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": time_interval[0],
                            "to": time_interval[1]
                        },
                        "mosaickingOrder": "leastCC"  # Orden de mosaico: menos cobertura de nubes primero
                    },
                    "processing": {
                        "upsampling": "BICUBIC"  # Cómo manejar el re-muestreo
                    },
                    "type": "Sentinel-5P"  # Colección de datos para contaminación atmosférica
                }
            ]
        },
        "output": {
            "width": 512,  # Ancho de la imagen de salida
            "height": 512,  # Alto de la imagen de salida
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": output_format
                    }
                }
            ],
            "evalscript": evalscript
        }
    }

    # Define los límites de la solicitud (bounding box o geometría)
    if geometry:
        payload["input"]["bounds"]["geometry"] = geometry
    elif bbox:
        payload["input"]["bounds"]["bbox"] = bbox
    else:
        raise ValueError("Se debe proporcionar 'geometry' o 'bbox'.")

    print(f"Enviando solicitud a la Processing API para {time_interval}...")
    
    try:
        response = oauth_session.post(PROCESSING_API_URL, json=payload)
        response.raise_for_status()  # Lanza un error para códigos de estado 4xx/5xx
        return response
    except requests.exceptions.HTTPError as e:
        print(f"Error HTTP en la Processing API: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        print(f"Error inesperado en la Processing API: {e}")
        raise

# --- EVALSCRIPTS PARA CONTAMINANTES ATMOSFÉRICOS ---

def get_no2_evalscript():
    """
    Evalscript para NO₂ troposférico con control de calidad según TFG.
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: ["L2__NO2___", "qa_value", "dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" }
        };
    }
    
    function evaluatePixel(sample) {
        // Control de calidad según TFG: qa_value > 0.75
        if (sample.qa_value > 0.75 && sample.dataMask === 1) {
            return [sample.L2__NO2___];
        } else {
            return [NaN];
        }
    }
    """

def get_o3_evalscript():
    """
    Evalscript para O₃ total con control de calidad.
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: ["L2__O3____", "qa_value", "dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" }
        };
    }
    
    function evaluatePixel(sample) {
        if (sample.qa_value > 0.75 && sample.dataMask === 1) {
            return [sample.L2__O3____];
        } else {
            return [NaN];
        }
    }
    """

def get_so2_evalscript():
    """
    Evalscript para SO₂ troposférico con control de calidad.
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: ["L2__SO2___", "qa_value", "dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" }
        };
    }
    
    function evaluatePixel(sample) {
        if (sample.qa_value > 0.75 && sample.dataMask === 1) {
            return [sample.L2__SO2___];
        } else {
            return [NaN];
        }
    }
    """

def get_co_evalscript():
    """
    Evalscript para CO total con control de calidad.
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: ["L2__CO____", "qa_value", "dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" }
        };
    }
    
    function evaluatePixel(sample) {
        if (sample.qa_value > 0.75 && sample.dataMask === 1) {
            return [sample.L2__CO____];
        } else {
            return [NaN];
        }
    }
    """

def get_aerosol_evalscript():
    """
    Evalscript para Aerosol Index con control de calidad.
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: ["L2__AER_AI", "qa_value", "dataMask"],
            output: { bands: 1, sampleType: "FLOAT32" }
        };
    }
    
    function evaluatePixel(sample) {
        if (sample.qa_value > 0.75 && sample.dataMask === 1) {
            return [sample.L2__AER_AI];
        } else {
            return [NaN];
        }
    }
    """

# --- FUNCIÓN PARA SERIES TEMPORALES DE CONTAMINANTES ---

def extract_pollutant_timeseries(oauth_session, pollutant, bbox, time_interval, aggregation_interval="P1D"):
    """
    Extrae series temporales de contaminantes atmosféricos usando Statistical API.
    
    Args:
        oauth_session: Sesión autenticada
        pollutant (str): Tipo de contaminante ('NO2', 'O3', 'SO2', 'CO', 'AER_AI')
        bbox (list): Bounding box [min_lon, min_lat, max_lon, max_lat]
        time_interval (list): [fecha_inicio, fecha_fin] en formato ISO 8601
        aggregation_interval (str): Intervalo de agregación (P1D=diario, P1M=mensual)
    
    Returns:
        pd.DataFrame: DataFrame con la serie temporal
    """
    
    # Seleccionar evalscript según contaminante
    evalscripts = {
        'NO2': get_no2_evalscript(),
        'O3': get_o3_evalscript(),
        'SO2': get_so2_evalscript(),
        'CO': get_co_evalscript(),
        'AER_AI': get_aerosol_evalscript()
    }
    
    if pollutant not in evalscripts:
        raise ValueError(f"Contaminante {pollutant} no soportado. Opciones: {list(evalscripts.keys())}")
    
    # Payload para Statistical API
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": time_interval[0],
                            "to": time_interval[1]
                        },
                        "mosaickingOrder": "leastCC"
                    },
                    "type": "Sentinel-5P",
                    "id": "input_data"
                }
            ]
        },
        "aggregation": {
            "timeRange": {
                "from": time_interval[0],
                "to": time_interval[1]
            },
            "aggregationInterval": {
                "of": aggregation_interval
            },
            "evalscript": evalscripts[pollutant]
        }
    }
    
    try:
        # Usar Statistical API endpoint
        statistical_url = PROCESSING_API_URL.replace('/process', '/statistics')
        response = oauth_session.post(statistical_url, json=payload)
        response.raise_for_status()
        
        results = response.json()
        
        # Procesar resultados
        data_for_df = []
        for result in results.get("data", []):
            date = result.get("interval", {}).get("from", "").split("T")[0]
            value = result.get("outputs", {}).get("default", {}).get("bands", {}).get("B0", [None])[0]
            
            if value is not None and not pd.isna(value):
                data_for_df.append({
                    "date": date, 
                    f"{pollutant.lower()}_satellite": value
                })
        
        if data_for_df:
            df = pd.DataFrame(data_for_df)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by="date").reset_index(drop=True)
            return df
        else:
            print(f"No se encontraron datos válidos para {pollutant}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error extrayendo datos de {pollutant}: {e}")
        return pd.DataFrame()

# --- EJEMPLOS DE USO ESPECÍFICOS PARA EL TFG ---

if __name__ == "__main__":
    oauth_session = get_authenticated_session()
    
    # Configuración para España según TFG
    bbox_spain = [-9.7559, 35.9468, 4.3278, 43.7914]  # España continental + islas
    time_interval = ["2019-01-01T00:00:00Z", "2024-12-31T23:59:59Z"]  # Período del TFG
    
    print("\n--- EXTRACCIÓN DE DATOS DE CONTAMINACIÓN ATMOSFÉRICA PARA TFG ---")
    
    # --- Ejemplo 1: Serie temporal de NO₂ (contaminante principal del TFG) ---
    print("\n1. Extrayendo serie temporal de NO₂...")
    try:
        df_no2 = extract_pollutant_timeseries(
            oauth_session, 
            'NO2', 
            bbox_spain, 
            time_interval, 
            aggregation_interval="P1D"  # Datos diarios
        )
        
        if not df_no2.empty:
            df_no2.to_csv("spain_no2_daily_2019_2024.csv", index=False)
            print(f"Serie temporal NO₂ guardada: {len(df_no2)} registros")
            print("Primeras filas:")
            print(df_no2.head())
        else:
            print("No se pudieron extraer datos de NO₂")
            
    except Exception as e:
        print(f"Error en extracción de NO₂: {e}")
    
    # --- Ejemplo 2: Serie temporal de O₃ ---
    print("\n2. Extrayendo serie temporal de O₃...")
    try:
        df_o3 = extract_pollutant_timeseries(
            oauth_session, 
            'O3', 
            bbox_spain, 
            time_interval,
            aggregation_interval="P1D"
        )
        
        if not df_o3.empty:
            df_o3.to_csv("spain_o3_daily_2019_2024.csv", index=False)
            print(f"Serie temporal O₃ guardada: {len(df_o3)} registros")
        else:
            print("No se pudieron extraer datos de O₃")
            
    except Exception as e:
        print(f"Error en extracción de O₃: {e}")
    
    # --- Ejemplo 3: Datos mensuales de múltiples contaminantes ---
    print("\n3. Extrayendo datos mensuales de múltiples contaminantes...")
    
    pollutants = ['NO2', 'O3', 'SO2', 'CO', 'AER_AI']
    monthly_data = {}
    
    for pollutant in pollutants:
        try:
            df_monthly = extract_pollutant_timeseries(
                oauth_session,
                pollutant,
                bbox_spain,
                time_interval,
                aggregation_interval="P1M"  # Datos mensuales
            )
            
            if not df_monthly.empty:
                monthly_data[pollutant] = df_monthly
                df_monthly.to_csv(f"spain_{pollutant.lower()}_monthly_2019_2024.csv", index=False)
                print(f"  {pollutant}: {len(df_monthly)} registros mensuales")
            else:
                print(f"  {pollutant}: Sin datos disponibles")
                
        except Exception as e:
            print(f"  {pollutant}: Error - {e}")
    
    print("\n--- EXTRACCIÓN COMPLETADA ---")
    print("Archivos generados:")
    print("- spain_no2_daily_2019_2024.csv")
    print("- spain_o3_daily_2019_2024.csv")
    print("- spain_[contaminante]_monthly_2019_2024.csv")
    print("\nEstos archivos están listos para el preprocesamiento avanzado (Fase 2.2)")
