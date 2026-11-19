# Preparación de publicación: Vercel y backend separado

**Actualización del Prompt 17:** la [comparación técnica posterior](DEPLOYMENT_COMPARISON.md) incorpora Large Functions de 5 GB y pruebas reales de ONNX en Chrome. La primera candidata pasa a ser Vercel Hobby con PyTorch y esa beta, pendiente de validación remota. Las conclusiones de tamaño estándar y los preparativos siguientes se conservan como evidencia del Prompt 16; no excluyen la beta. La configuración actual todavía publica solo el frontend.

Revisión local del 19/11/2026. No se han creado servicios, publicado artefactos, hecho commit/push ni desplegado. Streamlit, experimentos y presentación permanecen conservados.

## Decisión y evidencia

**La aplicación completa actual no cabe, sin modificar su entorno, en una función Python estándar de Vercel.** Se recomienda conservar el frontend en Vercel y ejecutar la API PyTorch aparte. Esto conserva inferencia neuronal real; Models es almacenamiento de artefactos, no un endpoint de inferencia gratuito de estas redes personalizadas.

| Alternativa | Conclusión |
|---|---|
| Todo en Vercel, PyTorch actual | El wheel Linux CPU cp313 manylinux_2_28 x86_64 oficial ocupa752.627.128 bytes sin comprimir; excede500 MB incluso antes de dependencias y aplicación. No se afirma que cualquier entorno recortado sea imposible, pero no se ha construido ni validado uno. |
| Vercel HTML + API Python separada | Arquitectura recomendada. Frontend compilable y API WSGI comprobados localmente. Falta proveedor, permisos y aprovisionamiento privado de evidencia. |
| ONNX Runtime en servidor | Candidato para reducir dependencias. No instalado ni convertido; no adoptado. Antes harían falta exportación local de las tres redes y pruebas por ventana/horizonte frente a PyTorch, con tolerancias justificadas y salidas en USD. No basta una exportación sin errores. |
| ONNX Runtime Web | Inferencia auténtica potencial, pero entregaría pesos y transformaciones a cada navegador. El repositorio privado y sus permisos no se pueden eludir. No se distribuye ni convierte nada en esta fase. |
| Render Free para API | Servicio gratuito disponible, con suspensión tras15 minutos sin tráfico, arranque lento y almacenamiento efímero. Es un candidato, **no una compatibilidad de memoria acreditada**. Verificar los recursos del plan en la cuenta, instalar Linux y medir el máximo de memoria con todas las vistas antes de recomendarlo. |
| Hugging Face Spaces | No se promete como alternativa sin coste: su documentación actual indica que crear Spaces con cómputo requiere plan de pago, aunque CPU Basic figure sin coste por hora. No contratar ni activar servicios. |
| API local + túnel HTTPS temporal | Alternativa para una demostración puntual sin contratar compute, tras autorizar exposición y contenido. Ordenador encendido, URL temporal y sin garantía de servicio24/7. No se ha abierto túnel alguno. |

La medición Linux leyó solamente1.250.998bytes del directorio central ZIP mediante HTTP Range; no instaló ni descargó el wheel completo. Archivo de evidencia local: `artifacts/local/deployment-validation/linux-wheel-size.json`. Es una medición de distribución sin recortar, no un build Vercel. Docker no está instalado: el Dockerfile está preparado, sin prueba de build Linux ni afirmación de compatibilidad final.

Vercel Hobby: Python WSGI/ASGI, límite500 MB sin comprimir,2 GB/1 vCPU,300 s con Fluid y4,5 MB por petición/respuesta de función. La caché de disco/proceso no debe suponerse persistente ni compartida entre instancias. Las cuotas gratuitas no equivalen a disponibilidad ilimitada. No se solicitan betas ni pruebas de pago para sortear límites.

Fuentes oficiales: [límites de funciones Vercel](https://vercel.com/docs/functions/limitations), [runtime Python](https://vercel.com/docs/functions/runtimes/python), [Render gratuito](https://render.com/docs/free), [Spaces](https://huggingface.co/docs/hub/spaces-overview), [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/index.html).

## Frontend preparado para Vercel

El directorio raíz del proyecto Vercel será `web/frontend`. Su configuración ejecuta `node build.mjs` y publica exclusivamente `public/`. No hay paquetes npm ni funciones Python en ese proyecto. No activar la opción de incluir archivos externos a la raíz. El build no necesita acceder al backend; su URL se suministra como variable pública `BRENT_API_BASE`.

El build admite un origen HTTPS DNS sin usuario, contraseña, ruta, parámetros ni fragmento. Falla si no está definido: no publica una interfaz apuntando silenciosamente a una API inexistente. Solo copia HTML, CSS, dos módulos de aplicación, Plotly, su licencia y config.js generado. No copia backend, datos, pesos, pruebas, documentos ni secretos. Rechaza sobrescribir una salida existente. En una repetición local usar un destino nuevo o retirar únicamente la salida generada que se haya identificado.

Comprobación local de empaquetado, sin despliegue (la URL siguiente es solo un ejemplo):

```powershell
$env:BRENT_API_BASE = "https://api.example.org"
Push-Location web/frontend
node build.mjs
Pop-Location
```

El resultado local no conectado al backend del ejemplo no ofrece inferencia. Para producción sustituirlo por el origen HTTPS real y autorizado. `config.js` nunca contiene HF_TOKEN y se sirve sin caché persistente.

## Backend y Models privados

La API reutiliza ForecastService y los mismos transformadores, máscaras, métricas y gráficos de Streamlit. El modo `hf` accede por HTTPS al endpoint oficial Hub `/resolve/{commit}/{filename}`. No se utiliza un servicio de inferencia de pago ni se afirma que esté instalado el SDK huggingface_hub. Se descargan los seis archivos permitidos; se comprueba el manifiesto antes de importar el cargador y cada tamaño/SHA256 antes de cargar los pesos en CPU.

Repositorio: `Juaandeleoon/caso-2-prediccion-petroleo`, privado. Revisión observada en el navegador: `b9ad781cd90e1adcb080009358fca143e0bf0931`. Hash del manifiesto local aprobado técnicamente: `0a087144d49fe504ba54d5498688535827f7c83b57913748f8f5f8a87dcf7570`. **Su coincidencia remota aún necesita comprobarse con autenticación oficial.** No usar main como versión ni confundir la sesión del navegador con un secreto del servidor.

| Variable del backend | Configuración |
|---|---|
| `BRENT_MODEL_SOURCE` | `hf` para el arranque de producción |
| `BRENT_HF_REPO_ID` | Repositorio anterior |
| `BRENT_HF_REVISION` | Commit completo fijo |
| `BRENT_MANIFEST_SHA256` | SHA256 independiente anterior, tras comprobar el remoto |
| `HF_TOKEN` | Secreto de lectura de alcance mínimo para ese repositorio; introducirlo solo en el gestor de secretos del backend |
| `BRENT_MODEL_CACHE_DIR` | Directorio escribible fuera del frontend; subcarpeta nueva por revisión |
| `BRENT_BUNDLE_DIR` | Ruta al paquete frozen original autorizado |
| `BRENT_DASHBOARD_DIR` | Ruta al paquete dashboard original autorizado |
| `BRENT_LIVE_DIR` | Directorio escribible para nuevas observaciones, independiente de ciencia congelada |
| `BRENT_WEB_ORIGIN` | Origen HTTPS exacto de Vercel, sin barra final; CORS no usa comodines |
| `BRENT_ALLOW_PROVIDER_ACCESS` | `0` hasta acreditar el acceso; `1` habilita comprobaciones automáticas |
| `BRENT_PUBLIC_ACCESS_AUTHORIZED` | `1` únicamente después de resolver permisos de uso/exposición y autorizar la publicación |
| `BRENT_BIND_HOST` / `PORT` | Loopback por defecto; en contenedor `0.0.0.0` y puerto exigido por proveedor |

El indicador de autorización es un bloqueo operativo; **no concede ninguna licencia**. Publicar predicciones, históricos y respuestas de API también requiere revisar permisos aunque no se ofrezca la descarga directa del dataset. Permanecen pendientes las condiciones de FRED/Yahoo y derivados documentadas en `docs/PUBLICACION.md`. No se ha autorizado ni realizado una nueva redistribución.

El paquete mínimo de Models contiene pesos/configuración/escaladores/cuantiles, pero **no histórico, ventana actual ni evidencia individual** para todas las vistas. Los paquetes frozen/dashboard siguen siendo imprescindibles. Un clon GitHub o los seis archivos HF por sí solos no aprovisionan el backend completo. Deben trasladarse por un medio privado y autorizado; no se han añadido a Git ni a una imagen pública. Su distribución/descarga remota es un bloqueo explícito, no un paso resuelto por tener un token.

Las revisiones verificadas se reutilizan sin nuevas descargas; ante corrupción no se ejecuta código remoto ni se vuelve silenciosamente a pesos locales. Los fallos de carga devuelven503 y se espacian30 s para evitar tormentas de descarga. `/api/health` comprueba el proceso; `/api/ready` carga recursos y devuelve503 si faltan, sin exponer tokens/URLs firmadas. No hay autenticación de usuario en la API: CORS no es un control de acceso y una API pública entrega sus respuestas a terceros. Un despliegue privado necesitaría protección del proveedor o autenticación adicional.

## Arranque de producción y actualización

`web/production.py` usa Waitress, dos hilos, inferencia CPU limitada a un hilo, límite de conexiones y cuerpo de petición, sin confiar automáticamente en cabeceras de proxy. Valida configuración y carga recursos antes de escuchar. `web/server.py` sigue siendo la opción local sencilla y no un servidor de producción.

Se han fijado versiones directas separadas en `web/requirements-runtime.txt`. No se cambiaron pyproject.toml/poetry.lock. Crear un entorno **distinto** para instalar esos requisitos; faltan resolución transitiva bloqueada y validación Linux. No instalar sobre la .venv científica solo para desplegar.

```powershell
# Con el entorno de despliegue y todas las variables/paquetes autorizados disponibles:
python -m web.production
```

`web/Dockerfile` es un candidato para backend separado; contexto raíz: `docker build -f web/Dockerfile -t brent-api .`. Su archivo .dockerignore específico permite solo código, configuraciones necesarias y métricas agregadas ya publicables. No hay COPY de pesos, ZIP, experimentos, presentación, datos ni credenciales. Los paquetes autorizados se aprovisionan aparte. No se ejecutó este comando ni se publicó ninguna imagen.

Fuentes actuales: FRED (Brent/WTI/gas/BCE/divisas/VIX), Yahoo (dólar/oro/cobre) y OPEC como contexto opcional, según configs/data.json. El backend inicializa desde CSV verificados locales. Con permiso habilitado, RefreshController consulta proveedores hasta la última fecha de decisión admisible; el fin histórico fijo no limita la adquisición nueva. La página Visión general consulta cada300 s mientras esté abierta; el servidor solo intenta actualizar cuando corresponde por política24 h. No existe un cron ni una promesa de actualización sin visitas.

Cada adquisición crea un snapshot separado; se comprueban fechas de disponibilidad, antigüedad, esquema e inferencia antes de activarlo. Si un proveedor de entrada falla, se rechaza el nuevo snapshot y se conserva la copia válida. No se actualizan pesos, escaladores, intervalos ni resultados históricos. Los retrasos asumidos y las revisiones retrospectivas siguen impidiendo garantizar datos históricos point-in-time.

En producción pública el botón manual está desactivado y el POST bloqueado incluso con Host/Origin falsificados; la comprobación automática autorizada permanece disponible. En un proveedor con disco efímero, reiniciar puede perder las nuevas observaciones y caché: se necesitaría recuperar un snapshot autorizado o volver al paquete congelado. No prometer continuidad de la última copia entre reinicios sin almacenamiento persistente. Un backend gratuito puede dormir; el frontend mostrará error/indisponibilidad hasta que esté listo, nunca predicciones inventadas.

## Pasos pendientes para obtener URL

1. Acreditar permisos para modelos, transformaciones, datos y respuestas individuales de la aplicación pública.
2. Configurar el secreto HF_TOKEN por un mecanismo oficial, comprobar los seis hashes remotos fijados y ejecutar inferencia real tras esa descarga.
3. Elegir backend: para demostración puntual, API local/túnel autorizado; para servicio independiente, validar un proveedor gratuito como Render con Linux/memoria/cold start y sin activar pago. Resolver el traslado privado de ambos paquetes y la conservación de snapshots.
4. Validar en ese entorno la instalación, paridad de las tres redes, todas las vistas, picos de memoria, actualización fallida y reinicio. No dar por aprobada una imagen solo por compilar.
5. Con autorización posterior, subir solo código/configuración permitidos a GitHub y desplegar la API. Esperar `/api/ready`200.
6. Autorizar después la publicación del frontend. En Vercel: Import Git Repository, elegir GitHub existente y rama acordada, Hobby, Root Directory `web/frontend`, variable `BRENT_API_BASE` con API real. Revisar build/output de vercel.json. No añadir HF_TOKEN a Vercel para un frontend estático.
7. Obtener la URL de Vercel, fijarla como BRENT_WEB_ORIGIN de la API, comprobar CORS exacto, HTTPS y las cinco páginas en móvil/escritorio. No confundir una URL de Preview con el origen definitivo. Publicar únicamente tras superar estas comprobaciones y contar con autorización.

No se requiere tarjeta, contratación ni activación de funciones de pago en esta preparación. Si una plataforma exige verificación o contratación no autorizada, detenerse. La presentación y GitHub Actions quedan fuera de esta fase.

## Verificación realizada

- Suite completa: **226/226 pruebas Python, 94,529 s**. La ejecución restringida se bloqueó en recursos de Streamlit; se detuvieron exclusivamente sus procesos verificados y la repetición autorizada terminó correctamente. Las pruebas de entrenamiento existentes usan ejemplos temporales; no entrenan las redes científicas congeladas.
- Frontend: **6/6 pruebas Node**; build real de siete archivos públicos, rechazo de URLs inseguras/credenciales y de sobrescritura. La salida local usa api.example.org como ejemplo y no constituye una conexión remota.
- HTTP real con Waitress: **16/16 consultas correctas**, incluidos los cuatro modelos por tres horizontes. La prueba fue loopback y se cerró al terminar; no prueba Linux, TLS del proveedor ni descarga privada.
- Las tres redes del candidato mínimo cargadas en CPU reproducen exactamente estimaciones, referencias, fechas y bandas del paquete original. La transferencia HF de esta prueba es simulada; no constituye autenticación remota. También se verifican paridad Streamlit, caché sin red, corrupción, errores, readiness, espera entre fallos, CORS y bloqueo manual público.
- Revisión visual de cinco secciones en escritorio y viewport 390×844, sin errores ni desbordamiento horizontal de página. No equivale a teléfono físico. Capturas locales: artifacts/local/app-screenshots/web-migration/deployment-desktop.jpg y deployment-mobile.jpg.
- pip check y git diff --check sin errores. Waitress se probó en una carpeta aislada y no se modificó el lock científico. Logs: artifacts/local/deployment-validation/full-suite-authorized.log y waitress-http.log.
- Integridad final SHA256: 163 archivos protegidos, 92 entradas del inventario y 21 archivos de la presentación sin diferencias. Búsqueda de patrones de credenciales/rutas personales en los archivos web sin coincidencias. La compilación contiene exactamente siete archivos y 4.842.690 bytes; salida, dependencia de validación y .vercel permanecen excluidos de Git.

No verificado: autenticación HF real con token del servidor, hashes remotos de esta fase, build Docker/Linux, memoria y cuotas de un proveedor gratuito, persistencia tras reinicio, permisos finales y URL pública. Estos puntos impiden afirmar que el despliegue completo esté listo.
