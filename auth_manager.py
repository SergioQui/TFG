# auth_manager.py
import requests
import json
import os
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
from datetime import datetime, timedelta

# Importa tus credenciales y URL del token desde config.py
from config import CLIENT_ID, CLIENT_SECRET, TOKEN_URL

TOKEN_FILE = "sentinel_hub_token.json"

# Define el hook de cumplimiento para manejar errores del servidor
def _sentinelhub_compliance_hook(response):
    response.raise_for_status() # Lanza un HTTPError para códigos de estado 4xx/5xx
    return response

def _save_token(token):
    """Guarda el token de acceso en un archivo."""
    with open(TOKEN_FILE, "w") as f:
        json.dump(token, f, indent=4)
    print("Token guardado en sentinel_hub_token.json")

def _load_token():
    """Carga el token de acceso desde un archivo."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    return None

def _is_token_valid(token_data):
    """Verifica si el token aún es válido."""
    if not token_data:
        return False
    # El token se considera válido si le quedan más de 60 segundos antes de expirar
    # El 'expires_at' se calcula en el momento de la adquisición
    return datetime.now().timestamp() < (token_data.get("expires_at", 0) - 60)

def get_authenticated_session():
    """
    Obtiene una sesión OAuth2 autenticada, reutilizando un token si es válido,
    o solicitando uno nuevo si es necesario.
    """
    token_data = _load_token()

    if _is_token_valid(token_data):
        print("Token cargado desde archivo y aún válido.")
        client = BackendApplicationClient(client_id=CLIENT_ID)
        oauth = OAuth2Session(client=client, token=token_data)
        oauth.register_compliance_hook("access_token_response", _sentinelhub_compliance_hook)
        return oauth
    else:
        if token_data:
            print("Token en archivo expirado o próximo a expirar.")
        print("Solicitando nuevo token de acceso...")
        client = BackendApplicationClient(client_id=CLIENT_ID)
        oauth = OAuth2Session(client=client)
        oauth.register_compliance_hook("access_token_response", _sentinelhub_compliance_hook)

        try:
            # Eliminar include_client_id=True si da problemas (depende de la versión de requests-oauthlib y el servidor)
            new_token = oauth.fetch_token(
                token_url=TOKEN_URL,
                client_secret=CLIENT_SECRET,
                include_client_id=True # Es buena práctica incluirlo explícitamente
            )
            # Guardar el timestamp de expiración para futuras validaciones
            new_token['expires_at'] = datetime.now().timestamp() + new_token['expires_in']
            _save_token(new_token)
            print("Nuevo token de acceso obtenido y guardado exitosamente.")
            return oauth
        except requests.exceptions.HTTPError as e:
            print(f"Error al obtener el token: {e}")
            if e.response.status_code == 429:
                print("Has solicitado demasiados tokens. Espera un momento antes de reintentar.")
            elif e.response.status_code == 401:
                print("Error de autenticación. CLIENT_ID o CLIENT_SECRET incorrectos.")
            raise # Re-lanza el error para que el programa principal lo maneje

# Puedes añadir una ejecución directa para probar solo el token
if __name__ == "__main__":
    try:
        session = get_authenticated_session()
        print("Sesión autenticada obtenida con éxito.")
    except Exception as e:
        print(f"No se pudo obtener la sesión autenticada: {e}")