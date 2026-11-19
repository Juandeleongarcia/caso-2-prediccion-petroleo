# Brent Forecast: proyecto Vercel integral

**Importación desde GitHub:** elegir `deployment/vercel-app` como Root Directory y **Other** como Framework Preset. No elegir `deployment/public_app`, `app`, la raíz científica ni `deployment/vercel` (plantilla). Build `node build.mjs`, Output Directory `public`, Python 3.13 y Fluid habilitado. Install Command: dejar el valor automático. No incluir archivos externos a Root Directory: este proyecto es autónomo.

La copia autónoma se genera mediante `scripts/prepare_vercel.py --output deployment/vercel-app`, con finales LF y manifiesto de hashes. Los modelos y datos privados no se copian. La fuente original permanece en web/ y src/oil_forecast/. La copia de despliegue no es otro experimento; sus módulos reproducen el mismo servicio. Las pruebas detectan divergencias. El generador rechaza sobrescritura: cualquier regeneración posterior requiere revisar primero los cambios del snapshot existente.

La carpeta `deployment/vercel` contiene la plantilla; la carpeta `deployment/vercel-app` con source_manifest.json contiene el proyecto importable. Git deduplica los blobs idénticos; se conserva una carpeta autónoma para que el build de Vercel no dependa de rutas externas o de Poetry científico.

El frontend sale por CDN y las nueve rutas API van a api/forecast.py, sin iniciar Streamlit/Waitress. La configuración permite despliegues automáticos desde main y los bloquea en las demás ramas mediante git.deploymentEnabled={"main":true,"**":false}. Publicar esta configuración en main puede iniciar un despliegue Production si la integración Git está activa; debe autorizarse antes de hacer push. Los snapshots live de datos se actualizan sin redeploy diario.

## Variables en Preview y Production

| Nombre | Valor o condición |
|---|---|
| VERCEL_SUPPORT_LARGE_FUNCTIONS | `1`, disponible para build y runtime; comprobar aceptación de la beta en la cuenta |
| BRENT_PUBLIC_ACCESS_AUTHORIZED | `1` solo tras resolver permisos y autorizar exposición pública; no concede licencias |
| BRENT_MODEL_SOURCE | `hf` |
| BRENT_HF_REPO_ID | `Juaandeleoon/caso-2-prediccion-petroleo` |
| BRENT_HF_REVISION | `b9ad781cd90e1adcb080009358fca143e0bf0931` |
| BRENT_MANIFEST_SHA256 | `0a087144d49fe504ba54d5498688535827f7c83b57913748f8f5f8a87dcf7570`; hash local, coincidencia remota autenticada pendiente |
| HF_TOKEN | Secreto de lectura restringido a repositorios necesarios, únicamente en el servidor; nunca pegarlo en chat/código |
| BRENT_DATA_REPO_ID | Repositorio autorizado con ambos ZIP originales: pendiente. Los seis archivos actuales de Models no los incluyen |
| BRENT_DATA_REVISION | Commit completo fijo de esos ZIP, pendiente; puede diferir del commit de pesos |
| BRENT_RUNTIME_DIR | `/tmp/brent-vercel-v1` |
| BRENT_ALLOW_PROVIDER_ACCESS | `0` en primera validación; `1` solo con permiso y después de comprobar tiempos/fallos |

Las carpetas de modelos, frozen, dashboard y live se fijan bajo el runtime temporal. Los registros SHA256 originales de ZIP/manifiestos anclan su autenticidad. El token se emplea solo contra Hugging Face HTTPS y no aparece en configuración estática. Recursos ausentes/corruptos o un repositorio inaccesible devuelven503, sin datos inventados ni pesos alternativos.

## Verificación y límites

Las cinco dependencias directas conservan sus versiones científicas. Python queda limitado a >=3.13,<3.14 en pyproject.toml, además de .python-version=3.13. El backend se instala desde pyproject.toml y uv.lock. requirements-linux.txt se reserva a verificaciones manuales/Docker; no debe renombrarse a requirements.txt en esta raíz.

Con CLI65.17.0, el builder estático @vercel/static-build28.0.1 detectaba requirements.txt y llamaba a runPipInstall antes del selector Python del backend. Ese helper invoca uv pip install --target sin --python y puede usar el intérprete3.14 del sistema, ignorando el pin del proyecto en esa ruta. La corrección retira ese disparador, sin overrides globales de installCommand ni cambios de dependencias. El selector oficial @vercel/python-analysis0.15.3 elige3.13 para api/forecast.py. Se verificó la rama estática antes/después con los paquetes oficiales de esa CLI; no es un despliegue Linux validado.

PyTorch CPU se descarga por su URL oficial exacta de CPython3.13/manylinux x86_64 con SHA256, idéntico al registrado en poetry.lock científico. El resto procede de PyPI. No se utiliza un extra-index global: con la estrategia first-index de uv, el índice CPU ocultaba filelock==4.2.12 de PyPI e impedía resolver. No se cambia PyTorch ni se desactiva la protección contra confusión de dependencias.

Se ha reproducido ese fallo con uv0.13.9 y comprobado la instalación cruzada de los 34 paquetes Linux en una carpeta aislada. No equivale a ejecutar/importar en Linux. Dockerfile.check prepara esa comprobación nativa con el mismo uv; necesita Docker o un host Linux y los artefactos autorizados. La aceptación final del build de Vercel sigue pendiente.

En la raíz generada, definir VERCEL_SUPPORT_LARGE_FUNCTIONS=1 y ejecutar `node build.mjs` comprueba el frontend sin publicación. `verify_runtime.py --output RUTA_NUEVA.json` comprueba inferencia/recursos sin consultar proveedores; `--require-linux` rechaza una ejecución Windows como prueba Linux. Necesita variables y artefactos autorizados. Registra arranque y, en Linux, RSS; no certifica una Preview de Vercel ni toda la concurrencia.

Vercel solo permite escritura en /tmp. Esa caché no es durable ni compartida entre instancias. Ante un fallo se conserva la copia válida de la instancia; reiniciar puede devolver al snapshot congelado. Sigue pendiente un origen versionado autorizado de observaciones recientes para garantizar continuidad entre reinicios.

La adquisición inicial de recursos no consulta proveedores. Visión general aplica después la política original de24horas cuando se habilita. No se reajustan pesos, escaladores ni cuantiles y no hay actualización manual pública. Se conservan las fechas de observación, disponibilidad, adquisición, origen y objetivo. Falta medir consultas externas dentro del presupuesto máximo de300s; no se afirma que los timeouts individuales garanticen ese límite total.

Bloqueos previos a URL pública: permisos de uso/exposición de datos y derivados, token oficial, origen autorizado y revisión de ambos ZIP, instalación/memoria Linux y build/arranque real aceptado por Hobby con la beta. No hay un bloqueo técnico demostrado que justifique sustituir PyTorch por ONNX ahora.

No pulsar Deploy ni ejecutar comandos de publicación hasta recibir autorización. La guía del repositorio recoge las pruebas y el estado real de esta preparación.
