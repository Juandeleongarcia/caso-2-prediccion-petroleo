"""Explicit production entry point; no deployment or provider access on import."""
import os
import re
from urllib.parse import urlsplit


def validate_environment(env):
    if env.get('BRENT_PUBLIC_ACCESS_AUTHORIZED') != '1':
        raise ValueError('Falta autorizar el acceso público a datos y derivados; no inicia la API pública.')
    if env.get('BRENT_MODEL_SOURCE') != 'hf':
        raise ValueError('La configuración de producción exige modelos verificados de Models.')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', env.get('BRENT_HF_REPO_ID', '')):
        raise ValueError('Falta el repositorio de Models.')
    if not re.fullmatch('[0-9a-f]{40}', env.get('BRENT_HF_REVISION', '')):
        raise ValueError('Se necesita una revisión inmutable de Models.')
    if not re.fullmatch('[0-9a-f]{64}', env.get('BRENT_MANIFEST_SHA256', '')):
        raise ValueError('Se necesita el hash independiente del manifiesto.')
    if not env.get('HF_TOKEN'):
        raise ValueError('El repositorio privado necesita un secreto de lectura del servidor.')
    url = urlsplit(env.get('BRENT_WEB_ORIGIN', ''))
    if url.scheme != 'https' or not url.netloc or url.username or url.password or url.path or url.query or url.fragment:
        raise ValueError('BRENT_WEB_ORIGIN debe ser un origen HTTPS exacto sin barra final.')
    if not env.get('BRENT_BUNDLE_DIR') or not env.get('BRENT_DASHBOARD_DIR') or not env.get('BRENT_LIVE_DIR'):
        raise ValueError('Se necesitan rutas explícitas para paquetes autorizados y datos recientes.')
    port = int(env.get('PORT', '8000'))
    if not 1024 <= port <= 65535:
        raise ValueError('Puerto de servicio inválido.')
    return {'host': env.get('BRENT_BIND_HOST', '127.0.0.1'), 'port': port,
            'threads': 2, 'connection_limit': 32, 'channel_timeout': 60,
            'max_request_body_size': 8192, 'clear_untrusted_proxy_headers': True}


def main():
    try:
        settings = validate_environment(os.environ)
        from waitress import serve
        import torch
        from web.backend import app
        torch.set_num_threads(1)
        app.resources()  # Fail startup before listening if authentication/data/integrity fails.
    except Exception:
        raise SystemExit('No se inicia la API: revisa permisos, secretos, paquetes y dependencias. No hay inferencia simulada.') from None
    serve(app, **settings)


if __name__ == '__main__':
    main()
