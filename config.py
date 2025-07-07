# config.py

# Tus credenciales de cliente de Copernicus Data Space Ecosystem (Sentinel Hub)
CLIENT_ID = 'sh-8c1c8810-0b35-406c-bb4f-47fd3f05f35b'
CLIENT_SECRET = 'YiZdxn1z5gfIuz5PNwf0auDsXKg2cm4r'

# URL para la obtención de tokens de autenticación (para requests_oauthlib)
# Asegúrate de que esta sea la URL CORRECTA para tu CDSE (la que has usado y funcionaba)
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# URL base para las APIs de Sentinel Hub (incluyendo Processing y Catalog)
# Esta es la URL más común para la API de Sentinel Hub dentro de CDSE
SH_BASE_URL = "https://services.dataspace.copernicus.eu"

# URL de la API de Procesamiento (basada en SH_BASE_URL)
PROCESSING_API_URL = f"{SH_BASE_URL}/api/v1/process"