import os
import requests
from datetime import datetime


OUTPUT_DIR = "datos_terrestres"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MITECO_API_URL = "https://apidatos.miteco.gob.es/air-quality/v1/records/"

MITECO_API_KEY = "TU_API_KEY_MITECO"


FECHA_INICIO = "2019-01-01"
FECHA_FIN    = "2024-12-31"

# Parámetros fijos: estaciones, contaminantes, formato CSV
PARAMS = {
    "start_date": FECHA_INICIO,
    "end_date":   FECHA_FIN,
    "contaminants": "no2,pm10,pm25,o3",       
    "format":       "csv",
    "api_key":      MITECO_API_KEY,
}

# Nombre del fichero de salida
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"terrestres_{FECHA_INICIO}_{FECHA_FIN}.csv")

def descargar_datos_terrestres():
   
    print(f"Solicitando datos terrestres de {FECHA_INICIO} a {FECHA_FIN}...")
    resp = requests.get(MITECO_API_URL, params=PARAMS, timeout=60)
    resp.raise_for_status()

    with open(OUTPUT_CSV, "wb") as f:
        f.write(resp.content)

    tamaño_kb = os.path.getsize(OUTPUT_CSV) / 1024
    print(f"Datos terrestres guardados en: {OUTPUT_CSV}")
    print(f"Tamaño del archivo: {tamaño_kb:.2f} KB")
    print(f"Proceso completado el {datetime.utcnow().isoformat()} UTC")

if __name__ == "__main__":
    descargar_datos_terrestres()