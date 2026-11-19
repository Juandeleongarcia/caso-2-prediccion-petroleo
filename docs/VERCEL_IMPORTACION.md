# Importación de Brent Forecast desde GitHub

La aplicación integral está en **deployment/vercel-app**. Es un snapshot de código generado y revisado, sin pesos ni datos privados. El manifiesto y las pruebas lo comparan con las fuentes originales normalizadas a LF. Mantiene las cinco secciones y la misma inferencia PyTorch. Streamlit y los experimentos originales no cambian.

## Pantalla de importación

Actualizar la pantalla después del push y seleccionar manualmente:

| Campo | Selección |
|---|---|
| Repositorio | Juandeleongarcia/caso-2-prediccion-petroleo |
| Rama | main |
| Root Directory | **deployment/vercel-app** |
| Framework Preset | **Other**, aunque se detecte Python |
| Build Command | **node build.mjs** |
| Output Directory | **public** |
| Install Command | Automático/sin override; dependencias Python en pyproject.toml + uv.lock y requirements.txt compatible |
| Include source files outside Root Directory | Desactivado: el paquete es autónomo |
| Python | 3.13, fijado en .python-version y >=3.13,<3.14 en pyproject.toml del paquete |
| Plan | Hobby; no activar Pro/trial ni recursos de pago |

No elegir deployment/public_app, app, web/frontend, deployment/vercel ni la raíz científica. La carpeta deployment/vercel es solo plantilla. vercel.json habilita Fluid y limita la función a300s. Los assets grandes de Plotly se sirven por CDN, no por la API.

**No pulsar Deploy todavía.** La autorización actual es publicar código seguro en GitHub, no desplegar. git.deploymentEnabled=false evita despliegues automáticos por commits para este proyecto; no es un permiso para un despliegue manual. No se modificaron ajustes remotos de Vercel.

## Variables de entorno

Configurar en Preview y Production, sin introducir credenciales en Git ni en JavaScript:

| Variable | Valor |
|---|---|
| VERCEL_SUPPORT_LARGE_FUNCTIONS | `1`, build y runtime |
| BRENT_MODEL_SOURCE | `hf` |
| BRENT_HF_REPO_ID | `Juaandeleoon/caso-2-prediccion-petroleo` |
| BRENT_HF_REVISION | `b9ad781cd90e1adcb080009358fca143e0bf0931` |
| BRENT_MANIFEST_SHA256 | `0a087144d49fe504ba54d5498688535827f7c83b57913748f8f5f8a87dcf7570` |
| HF_TOKEN | Secreto oficial de lectura; introducir únicamente en el gestor de secretos de Vercel |
| BRENT_DATA_REPO_ID | Pendiente: repositorio autorizado con ambos ZIP originales |
| BRENT_DATA_REVISION | Pendiente: commit completo fijo que contenga esos ZIP |
| BRENT_RUNTIME_DIR | `/tmp/brent-vercel-v1` |
| BRENT_ALLOW_PROVIDER_ACCESS | `0` hasta autorizar y verificar las consultas externas |
| BRENT_PUBLIC_ACCESS_AUTHORIZED | `0` mientras no se resuelvan permisos; `1` solo después de resolverlos y autorizar exposición |

El SHA indicado es el de la copia local; sigue pendiente verificarlo con descarga remota autenticada. El token debe permitir leer los repositorios necesarios sin conceder escritura. No usar variables con prefijos públicos ni enviar tokens por chat.

## Bloqueos reales

- Los seis archivos actuales de Models incluyen redes/transformadores/cuantiles, pero no los paquetes brent-frozen-v1.zip y brent-dashboard-v1.zip necesarios para todas las vistas. No se añaden a GitHub ni se autoriza su subida por esta integración.
- Faltan permisos acreditados para distribuir/exponer datos y derivados de proveedores, según PUBLICACION.md. Publicar el código propio y Plotly con su licencia no resuelve esos permisos. No activar la autorización pública para intentar saltarlos.
- Faltan autenticación HF real, origen autorizado de los ZIP, instalación/memoria Linux y aceptación efectiva de Large Functions en la cuenta/build. Hay resolución de wheels y pruebas Windows; no un despliegue validado.
- /tmp es efímero: la última copia válida de datos no se comparte entre instancias ni queda garantizada tras reinicios. No se inventan datos ante fallos. Con información insuficiente la API devuelve503 y la interfaz lo comunica.

Por tanto, esta publicación permite seleccionar el proyecto correcto y preparar su build, **no garantiza inferencia pública inmediata**. La alternativa ya verificada sigue siendo la aplicación completa local. No se ha demostrado un nuevo bloqueo de PyTorch que justifique sustituirla por ONNX.

## Mantenimiento

Las fuentes se conservan en web/, src/oil_forecast/ y deployment/vercel/. scripts/prepare_vercel.py genera el snapshot sin datos/pesos y con hashes; admite exclusivamente una carpeta nueva en artifacts/local o deployment/vercel-app. Rechaza sobrescribir una copia existente. Cualquier actualización debe revisar la diferencia, regenerar por un procedimiento controlado y superar test_vercel.py antes de publicar. No editar solo el snapshot ni duplicar experimentos.

La comparación de pruebas y recursos de las fases previas está en web/VERCEL_LARGE_FUNCTIONS.md. Las restricciones de artefactos y derechos se mantienen aunque el repositorio de modelos sea privado.

## Verificación de esta integración

### Corrección del instalador de Vercel

Se reprodujo el conflicto torch/filelock con uv0.13.9 y el requirements anterior. Su causa es la prioridad del índice CPU bajo first-index: ocultaba filelock de PyPI. Se conserva torch==2.15.0+cpu mediante su wheel oficial Linux CPython3.13 con el SHA256 original; las demás dependencias se obtienen de PyPI. No se usa unsafe-best-match ni se modifican los pesos. Se añaden pyproject.toml con Python >=3.13,<3.14 y uv.lock para fijar también el instalador basado en proyectos. El motivo exacto por el que el build anterior ignoró .python-version requiere el log completo; ahora ambos mecanismos declaran 3.13.

La instalación cruzada de los 34 paquetes Linux funciona. No acredita imports ni consumo de memoria en Linux: este equipo no tiene Docker/WSL operativo. Dockerfile.check conserva esa comprobación pendiente. Para el siguiente intento, seleccionar el commit corregido, mantener la configuración de la tabla y reconstruir sin reutilizar la caché anterior. Confirmar CPython3.13 en los logs; no desplegar de nuevo el commit fallido. No se requiere ninguna credencial adicional para esta corrección del instalador.

La corrección supera **237/237 pruebas Python (94,494 s)**, **6/6 JavaScript**, pip check y git diff --check. Incluye build CDN, arranque WSGI y paridad con Streamlit de tres redes y persistencia en tres horizontes. Los 163 archivos protegidos, 92 entradas del inventario y 21 archivos de presentación conservan sus hashes. El snapshot importable contiene 45 archivos y 44 hashes de payload, sin artefactos privados. Las 34 versiones instaladas coinciden con uv.lock; se comprobaron 90 bibliotecas ELF y uv sync --locked --dry-run. Los logs permanecen en artifacts/local/vercel-dependency-fix, excluido de Git.

Referencias oficiales: [selección de Python en Vercel](https://vercel.com/docs/functions/runtimes/python/python-version) y [prioridad de índices de uv](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).

Las cifras siguientes pertenecen a la integración previa.

236/236 pruebas Python (101,212 s) y 6/6 JavaScript superadas. El nuevo test comprueba que el snapshot coincide con las fuentes normalizadas y que sus 42 hashes y la desactivación de despliegues Git se conservan. Los 163 archivos protegidos, 92 entradas del inventario y 21 archivos de presentación mantienen sus SHA256. No hay nuevos entrenamientos ni cambios en los experimentos. Se publica solo código y documentación segura; los pesos, ZIP, históricos y credenciales continúan excluidos.
