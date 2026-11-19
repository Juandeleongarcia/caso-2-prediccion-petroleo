# Brent Forecast: web HTML y API Python

> **Nota de cierre (19/11/2026):** este documento conserva el historial de fases. El [README principal](../README.md) recoge el estado actual: web desplegada con inferencia real, canal live operativo y presentación UAX terminada. Las restricciones y verificaciones anteriores se mantienen como trazabilidad; Hugging Face sigue privado y el acceso de la profesora está pendiente.

Migración local en revisión. Streamlit (`app.py`) permanece íntegro como respaldo. No hay entrenamiento, publicación ni despliegue en esta fase. La presentación continúa pausada. La preparación posterior de producción, sus pruebas y bloqueos están en [DEPLOYMENT.md](DEPLOYMENT.md).

## Arranque local

Desde la raíz, con el entorno Poetry existente y los dos paquetes originales autorizados ya instalados:

```powershell
$env:BRENT_MODEL_SOURCE = "local"
$env:BRENT_ALLOW_PROVIDER_ACCESS = "0"
poetry run python -B web/server.py
```

Abrir <http://127.0.0.1:8765>. Un clon sin los paquetes no dispone de datos/evidencias ni pesos para ejecutar la aplicación completa. Las instrucciones originales para instalar y verificar ambos paquetes están en el README principal. El servidor solo escucha en loopback; no es un servidor de producción para Internet. No se ha instalado ninguna dependencia nueva ni cambiado pyproject/lock.

`frontend/` contiene HTML, CSS, JavaScript y Plotly.js obtenido de la versión ya instalada. Su licencia se conserva en `frontend/vendor/LICENSE.plotly.txt`. No se usan CDN ni APIs de texto. Los datos y gráficos se solicitan al backend; no se incorporan precios, checkpoints ni resultados individuales al frontend estático.

## Correspondencia con Streamlit

| Vista | Funciones conservadas |
|---|---|
| Visión general | Histórico real, periodos, acontecimientos económicos, referencia disponible y última cotización diferenciadas, indicadores, tres salidas y control de actualización |
| Predicción | RNN-D/42, LSTM-B/42, GRU-C/42 y persistencia; 5/63/252; cambios y bandas; explicación determinista y calculadora; archivo histórico con orígenes realmente disponibles |
| Comparación | Desarrollo/calibración/test, media entre semillas u operativa42, MAE/MSE/RMSE/R², mejora frente a persistencia, parámetros y tiempo |
| Evaluación | Cuatro experimentos, arquitectura, variante, semestre, horizonte, predicción frente a observación, error, pérdidas, métricas, periodos, errores mayores/menores y cobertura |
| Metodología | Arquitecturas, ventanas, selección de épocas, 204+3 ajustes y1481épocas, particiones y limitaciones |

No hay trayectorias simuladas. Los gráficos reutilizan `app_charts.py` y sus datos originales. La nueva API reutiliza `ForecastService`, `forecast_for_model`, `aggregate_metrics`, métricas enriquecidas y `RefreshController`. Los resultados archivados se leen de predicciones guardadas, nunca aplicando un checkpoint posterior a un origen anterior. Su gráfico de contexto se limita al origen; el real posterior se identifica expresamente. Las bandas de calibración son retrospectivas y la cobertura es descriptiva.

El precio objetivo, la observación de referencia, disponibilidad, adquisición y reajuste lógico son fechas diferentes. Los pesos siguen siendo los del reajuste lógico de julio2024; no se acredita su fecha física de entrenamiento. Persistencia supera las medias neuronales en los tres horizontes de test. Los resultados desfavorables y R² negativos permanecen visibles.

La web presenta las métricas por horizonte seleccionado, con tablas desplazables en móvil. No copia literalmente la distribución de componentes de Streamlit. Conservar Streamlit permite revisar cualquier detalle adicional mientras se aprueba la migración.

## API

GET `/api/health` comprueba el proceso, sin afirmar que los modelos estén cargados. `/api/models` carga/verifica los recursos y muestra fuente de pesos y fechas. Los restantes endpoints son `/api/overview`, `/api/predict`, `/api/archive-origins`, `/api/comparison` y `/api/evaluation`. Las dimensiones admitidas se validan en el servidor. Selecciones inválidas devuelven400, artefactos inaccesibles503; no se sustituyen por cifras ficticias.

POST `/api/refresh` solo está disponible desde el origen local de desarrollo y con permisos de proveedores habilitados. No hay endpoint de entrenamiento. No existe un endpoint de archivos arbitrarios: solo se sirven los ficheros de frontend; datos, secretos, paquetes y código privado no son estáticos.

Se examinó FastAPI: no está instalado en el entorno. Se utiliza WSGI y biblioteca estándar para conservar el entorno científico. Esto evita añadir FastAPI/httpx y no transforma `wsgiref` en un servidor de producción. La lógica de `Study` se puede exponer posteriormente con FastAPI si se acuerda ese cambio; no se declara una implementación FastAPI inexistente.

## Variables y Models

| Variable del servidor | Uso |
|---|---|
| `BRENT_MODEL_SOURCE` | `local` por defecto; `hf` exige todos los datos de confianza siguientes y no vuelve silenciosamente a local |
| `BRENT_HF_REPO_ID` | `Juaandeleoon/caso-2-prediccion-petroleo` |
| `BRENT_HF_REVISION` | Commit real completo de40hex, nunca main/etiqueta mutable |
| `BRENT_MANIFEST_SHA256` | SHA256 del manifiesto aprobado, obtenido por un canal independiente |
| `HF_TOKEN` | Solo secreto del servidor para repositorio privado; no va al navegador, URLs ni registros |
| `BRENT_BUNDLE_DIR` / `BRENT_DASHBOARD_DIR` | Paquetes locales completos para datos, ventana y evidencia. Por defecto los de artifacts/local |
| `BRENT_LIVE_DIR` | Copias nuevas separadas de entrenamiento; por defecto data/live |
| `BRENT_ALLOW_PROVIDER_ACCESS` | `0` por defecto. `1` permite consultas autorizadas con el controlador original |
| `BRENT_WEB_ORIGIN` | Origen HTTPS exacto del frontend cuando se separa del backend; sin comodines |

El modo HF descarga exclusivamente seis archivos del candidato mínimo, verifica primero el manifiesto contra el SHA independiente, comprueba tamaños y hashes antes de importar el cargador o abrir los pesos y utiliza CPU. Los tokens solo se envían al origen inicial HTTPS de Hugging Face y no se reenvían en redirecciones; se rechazan HTTP y destinos ajenos. La descarga tiene tiempo y tamaño máximos, instala en un directorio nuevo por revisión y no sobrescribe revisiones anteriores. Un fallo muestra indisponibilidad y conserva los archivos previos; la selección de una revisión anterior debe ser explícita.

**Estado remoto comprobado en esta fase:** desde la sesión autenticada del navegador se verificaron los seis archivos y la visibilidad privada en la revisión `b9ad781cd90e1adcb080009358fca143e0bf0931`. La API sin autenticación devuelve401. La sesión del navegador no proporciona autenticación al backend: falta configurar un secreto oficial de lectura del servidor y completar la descarga real con comprobación SHA256. Un intento de descarga del manifiesto en el navegador no se pudo completar; no se presenta como verificación de integridad remota. El manifiesto local tiene SHA256 `0a087144d49fe504ba54d5498688535827f7c83b57913748f8f5f8a87dcf7570`; debe coincidir con el remoto antes de aceptar la revisión. La existencia de una carga privada no acredita permisos de redistribución. No se solicita pegar tokens en el chat.

El backend actual sigue utilizando los paquetes locales originales para histórico/evidencia, incluso en modo HF. El paquete mínimo no contiene esas series ni la ventana reciente. Conectar los pesos de Models no proporciona automáticamente datos de mercado o gráficos de backtest. Los permisos pendientes de FRED/Yahoo y derivados siguen aplicando: no se publica el backend o su contenido individual para eludirlos.

## Actualización de datos

El histórico de entrenamiento no se modifica. Se recupera la última copia válida de data/live, o la del paquete congelado si no existe una válida. Con acceso autorizado habilitado, cada consulta a Visión general comprueba la política de24horas; la página activa consulta cada300segundos. Sin visitas no hay un planificador de fondo. El botón local permite consulta manual. Los fallos respetan reintentos y conservan la última copia válida. Una adquisición tiene que validar las diez entradas y pasar inferencia antes de activarse. No se reajustan pesos, escaladores ni cuantiles.

Una observación reciente se incorpora solo según su fecha de disponibilidad, metadatos, antigüedad y hora de decisión Madrid. Persisten los límites de revisiones retrospectivas y retardos asumidos; no se proclama ausencia absoluta de fuga histórica. No se hicieron adquisiciones nuevas durante las pruebas.

## Vercel y alternativa gratuita

La [preparación concreta del Prompt 18](VERCEL_LARGE_FUNCTIONS.md) añade un generador de proyecto integral, adaptador WSGI y plantilla Large Functions. La salida local vercel-large-ready incluye API y frontend del mismo origen; no modifica ni sustituye la configuración estática independiente conservada a continuación. La guía distingue pruebas locales, cuenta Vercel y artefactos/permisos pendientes.

La [comparación actualizada del Prompt 17](DEPLOYMENT_COMPARISON.md) recomienda comprobar primero Large Functions de 5 GB en Hobby antes de añadir Render. Incluye mediciones del backend y equivalencia ONNX en CPU y Chrome. No se ha cambiado la arquitectura activa ni desplegado: la configuración siguiente continúa siendo la preparación estática del Prompt 16.

Preparación **solo del frontend estático**: Root Directory `web/frontend`, Framework Preset `Other`, Build Command `node build.mjs` y Output Directory `public`. Definir BRENT_API_BASE con el origen HTTPS real de la API. El build copia una lista cerrada de archivos públicos y genera config.js sin secretos; no publica el directorio fuente completo. No seleccionar la raíz del proyecto: `app.py` es Streamlit y no es la API WSGI. No se ha ejecutado un deploy Vercel.

En `frontend/config.js`, `window.BRENT_API_BASE` será el origen HTTPS real del backend, sin barra final. Vacío significa mismo origen en local. Esto es configuración pública, nunca un secreto. Configurar `BRENT_WEB_ORIGIN` en el backend para ese frontend exacto. Un frontend publicado sin backend válido muestra error y no genera predicciones en JavaScript. Una página HTTPS no debe apuntar a un backend HTTP sin protección.

Revisión oficial19/11/2026: https://vercel.com/docs/functions/limitations documenta500MB sin comprimir para Python,2GB/1vCPU en Hobby y duración máxima300s con Fluid compute. https://vercel.com/docs/functions/runtimes/python admite WSGI/ASGI. El tipo de interfaz HTML no reduce la instalación de PyTorch del backend.

Medición local Windows: torch458,0MiB, NumPy21,9, pandas44,6 y Plotly33,5, aproximadamente558MiB solo en esos cuatro directorios. No son tamaños Linux ni un bundle Vercel construido. Proceso local: carga/importación8,70s, predicción con gráfico0,77s, máximo de working set326807552bytes (aproximadamente312MiB). Son una medición, no una garantía de capacidad/latencia. Los detalles están en artifacts/local/web-benchmark.json. **No se certifica que el backend actual quepa en Vercel**; hace falta un build Linux CPU minimizado y prueba real de memoria, arranque y límites, además de los permisos de uso.

Alternativa sin contratar compute: para una demostración académica autorizada, mantener la API en el ordenador del estudiante y, si se aprueba expresamente su exposición, usar un túnel HTTPS temporal. Cloudflare Quick Tunnels no requiere cuenta/dominio, ofrece URL temporal y carece de garantía de disponibilidad: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/ . El equipo debe permanecer encendido y la URL cambia. Esto puede dar inferencia real para una revisión puntual, pero no equivale a un backend gratuito autónomo24/7. No se ha instalado cloudflared ni abierto un túnel. No se promete un servicio gratuito permanente no verificado.

## Pruebas

```powershell
poetry run python -X utf8 -B -W error -m unittest discover -s tests -p test_web.py -v
node --test web/tests/frontend.test.mjs
```

Las pruebas de backend incluyen paridad exacta de las respuestas de cuatro modelos × tres horizontes con el servicio que usa Streamlit,48combinaciones de evaluación (cuatro experimentos × cuatro modelos × tres horizontes), R² negativos, archivo original, controles de casos, aislamiento de estáticos, fallo seguro sin revelar secretos y descarga fijada simulada con rechazo de corrupción/redirecciones inseguras. La descarga simulada no es una conexión real a Models. Las pruebas JavaScript verifican filtros, ausencias, separación archivo/inferencia y reinicio de dimensiones dependientes. La revisión en navegador complementa estas pruebas.

Resultado local de esta fase: **220/220 pruebas Python** (suite completa,103,686s), incluidas10nuevas de la API; **4/4 pruebas JavaScript**. `poetry check --lock`, `pip check` y `git diff --check` superados. Se revisaron las cinco vistas en escritorio y viewport móvil390×844, con tablas desplazables y sin desbordamiento de página. No equivale a una prueba en un teléfono físico. La comparación SHA256 confirmó163archivos protegidos y92entradas del inventario sin diferencias; también se conservaron los21archivos de la presentación. La captura final está en artifacts/local/app-screenshots/web-migration/desktop.jpg; logs en artifacts/local/web-full-tests.log y web-tests-final.log.

No se alteran experimentos, configuraciones científicas, CSV, transformadores, cuantiles ni checkpoints. Los logs, medidas, capturas y verificaciones quedan solo en artifacts/local. No hay commit, push, publicación de pesos, Vercel ni reentrenamiento. Los recuentos anteriores corresponden a la migración; los resultados de las pruebas adicionales de despliegue constan en DEPLOYMENT.md.
