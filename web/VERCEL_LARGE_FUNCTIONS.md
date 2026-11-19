# Integración GitHub posterior

La raíz importable actual es **deployment/vercel-app**. Véase [guía actual de importación](../docs/VERCEL_IMPORTACION.md). El contenido siguiente conserva la preparación del Prompt18 y sus pruebas; la limitación anterior a CLI queda superada por el snapshot de código publicado para GitHub. No cambia el estado de permisos ni autoriza un despliegue.

# Prompt 18: proyecto Vercel integral preparado

Preparación local del 19/11/2026. No hay commit, push, despliegue, activación de planes ni carga adicional de datos/pesos. Streamlit y el estudio científico permanecen conservados.

## Paquete concreto

`artifacts/local/vercel-large-ready/` contiene un proyecto autónomo de **43 archivos y 5.249.826 bytes antes del build**, generado por `scripts/prepare_vercel.py`. Incluye solo módulos necesarios, seis assets fuente del frontend, configuraciones, métricas agregadas ya publicables y la plantilla `deployment/vercel/`. Un manifiesto registra hashes de los archivos copiados. No contiene datasets, ventanas privadas, predicciones individuales, ZIP, pesos, experimentos ni secretos.

Las primeras carpetas vercel-large-v1/v2 son instantáneas previas de empaquetado local, conservadas y excluidas de Git; utilizar **vercel-large-ready** para revisar esta entrega. El generador rechaza destinos existentes. Para regenerar después de otro cambio usar un destino nuevo, no modificar la copia generada manualmente.

### Cómo se ejecutará

- Una función Python, `api/forecast.py`, expone el callable WSGI de `web/vercel_runtime.py`. No abre sockets ni ejecuta Waitress/Streamlit.
- Las nueve rutas /api se reescriben a esa función con selección de endpoint validada; se rechazan rutas ambiguas y métodos inválidos.
- HTML/CSS/JS/Plotly se compilan a `public/` y se sirven por CDN. Plotly supera 4,5 MB y **no** se envía como respuesta de la función. La API es del mismo origen: no requiere configurar otro dominio/backend.
- vercel.json habilita Fluid y fija maxDuration=300; no intenta modificar la memoria Hobby. El build exige VERCEL_SUPPORT_LARGE_FUNCTIONS=1. La aceptación efectiva de la beta solo puede acreditarse en la cuenta/build de Vercel.
- La configuración científica raíz no se copia ni se edita. El proyecto autónomo tiene requirements.txt y .python-version=3.13; no puede seleccionar accidentalmente las dependencias completas del Poetry científico.

Se mantienen las cinco dependencias directas con versiones exactas: torch2.15.0+cpu, numpy2.5.4, pandas3.1.0, plotly7.1.0 y yfinance1.8.0. Yfinance conserva las actualizaciones autorizadas. No incluye Streamlit, Waitress, ONNX ni herramientas de entrenamiento como dependencias adicionales. Las funciones de inferencia siguen usando el código original; no se cambiaron los parámetros ni las transformaciones científicas.

## Models y datos: dos aprovisionamientos

La descarga mínima de modelos usa el endpoint oficial HTTPS de Models, revisión completa y manifiesto SHA256 independiente; después verifica tamaños/hashes de los seis archivos antes de importar/cargar. CPU, sin fallback a otras redes. La caché por revisión evita descargar de nuevo en la misma instancia. Los errores son sanitizados y las tentativas fallidas se espacian30s.

El histórico y la evidencia completa se instalan con los verificadores originales y los registros de configs/serving_bundle.json y dashboard_bundle.json. Se comprueban SHA256 de ZIP, manifiesto y contenido; las instalaciones son nuevas y no sobrescriben originales. Pueden provenir de otro repositorio/revisión privada de Models, **solo cuando se autorice distribuirlos**. El token no se incluye en frontend ni URLs y no se reenvía a hosts de CDN ajenos.

Los seis archivos del repositorio actual no proporcionan los dos paquetes de evidencia. Por tanto, configurar únicamente HF_TOKEN no completa la aplicación. Si faltan datos/histórico, hay corrupción o falla Models, la API devuelve503 sin inventar resultados. Una caché válida de modelos puede reutilizarse; no se finge una descarga autenticada por haber probado esa caché.

El arranque carga la última observación válida sin adquirir proveedores. Visión general puede comprobar después la política de24horas si BRENT_ALLOW_PROVIDER_ACCESS=1. El botón manual público está bloqueado. Un proveedor fallido conserva ventana, referencia, fecha y predicciones previas. El fin histórico del estudio permanece inmutable; las nuevas observaciones usan otra carpeta y disponibilidad temporal validada. No hay reentrenamiento.

Todos los archivos escribibles están bajo /tmp/brent-vercel-v1. La caché/snapshot no se comparte ni persiste entre instancias: no se garantiza recuperar la adquisición más reciente tras reinicios. Sin una copia reciente recuperable se parte del snapshot congelado y se informa de sus fechas. Para continuidad durable falta un origen versionado autorizado; no se ha contratado almacenamiento ni creado un cron.

## Qué configurar en tu cuenta

**No pulsar Deploy todavía.** Primero resolver los bloqueos siguientes. La raíz científica y `deployment/vercel/` no son proyectos integrales desplegables por sí solos: la raíz correcta es el paquete **generado**.

1. Mantener cuenta/proyecto en Hobby; no activar pruebas Pro, tarjetas ni add-ons. Confirmar que Large Functions está disponible y Fluid/Active CPU habilitados.
2. Una vez autorizada la creación/publicación, enlazar por CLI desde `artifacts/local/vercel-large-ready/` con `npx vercel link`. Elegir Other, build `node build.mjs`, output `public`, Python3.13; usar los archivos del paquete, no un import directo del repositorio científico.
3. En Project Settings → Environment Variables configurar las variables de la tabla para Preview y Production. HF_TOKEN debe ser un secreto de lectura con acceso mínimo a los repositorios necesarios; no enviarlo al chat ni usar un prefijo de variable pública.
4. La primera verificación remota debe ser una Preview protegida, con consultas a proveedores desactivadas. Crear incluso esa Preview requiere la autorización posterior; no se ha ejecutado `npx vercel` ni `--prod`.
5. Comprobar en build/logs la ruta Large Functions, tamaño total real, instalación Linux, health/ready, cinco vistas, tres redes/persistencia, tres horizontes, banda, fallos y cold start. Revisar consumo de memoria/CPU/cuotas y que no haya errores de ruta/404. Solo después aprobar publicación.

| Variable | Valor |
|---|---|
| VERCEL_SUPPORT_LARGE_FUNCTIONS | `1`, build y runtime |
| BRENT_PUBLIC_ACCESS_AUTHORIZED | `1` únicamente tras resolver permisos y autorizar exposición; no otorga licencias |
| BRENT_MODEL_SOURCE | `hf` |
| BRENT_HF_REPO_ID | `Juaandeleoon/caso-2-prediccion-petroleo` |
| BRENT_HF_REVISION | `b9ad781cd90e1adcb080009358fca143e0bf0931` |
| BRENT_MANIFEST_SHA256 | `0a087144d49fe504ba54d5498688535827f7c83b57913748f8f5f8a87dcf7570` |
| HF_TOKEN | Secreto oficial de lectura, pendiente de configurar y probar |
| BRENT_DATA_REPO_ID | Pendiente: origen autorizado de brent-frozen-v1.zip y brent-dashboard-v1.zip |
| BRENT_DATA_REVISION | Pendiente: commit completo fijo con esos ZIP; no tiene que coincidir con el de pesos |
| BRENT_RUNTIME_DIR | `/tmp/brent-vercel-v1` |
| BRENT_ALLOW_PROVIDER_ACCESS | `0` en primera validación; `1` después de acreditar permiso y presupuesto de tiempo |

El hash del manifiesto de Models es el de la copia local preparada; falta comprobar la coincidencia con la descarga remota autenticada. Los hashes de ambos ZIP se fijan en los registros originales; no se inventan URL/revisión de recursos aún no publicados. Publicar respuestas individuales de la API también requiere revisar las condiciones de FRED/Yahoo/derivados. Privado no equivale a autorización de redistribución.

No se ha añadido un workflow GitHub ni copias de runtime al Git científico. Para conectar este paquete a GitHub habría que acordar una estrategia de generación/publicación posterior, sin duplicar o desincronizar el código. La vía CLI desde la salida generada permite revisar primero el paquete exacto que se transmitiría.

## Pruebas ejecutadas y preparadas

Nueve pruebas nuevas cubren cierre de imports y hashes del paquete, ausencia de artefactos privados, no sobrescritura, build CDN de siete archivos, configuración beta/pins/credenciales, routing, paridad de las tres redes con Streamlit en los tres horizontes, rechazo de histórico ausente/corrupto, Models fallido sin fallback y fallo de proveedor conservando resultados válidos. Las credenciales de esas pruebas son ficticias y no se realiza red.

La prueba independiente del **paquete generado**, en un proceso nuevo y con copias locales verificadas de los artefactos fuera del paquete, pasó16consultas WSGI. Importación/carga en frío:5,896s; CPU acumulada:6,719s; pico working set:332.181.504bytes; pico memoria comprometida:1.064.599.552bytes. Son mediciones Windows con caché de prueba, no Linux, Vercel, autenticación remota ni latencia de descarga. El frontend se compiló con API del mismo origen y sin secretos. La metodología es contenido estático conservado; las otras vistas consultan la API original.

Se resolvieron34candidatos wheel para CPython3.13/manylinux x86_64, con versiones guardadas en constraints-linux.txt. **No se ejecutó una instalación Linux:** no hay Docker ni WSL instalado. La resolución cruzada puede evaluar marcadores del anfitrión; no sustituye instalar/importar dentro de Linux. La prueba --require-linux rechaza expresamente este Windows.

El paquete contiene Dockerfile.check y verify_runtime.py. En una máquina Linux/Docker, construir con `docker build -f Dockerfile.check -t brent-vercel-check .` desde la raíz generada. Para ejecutar, montar una carpeta de resultados en /results y un runtime de prueba con los paquetes/modelos verificados en /tmp/brent-vercel-v1; pasar las variables necesarias y usar `--network=none` si todos los artefactos ya están provisionados. No montar los originales científicos como carpetas de escritura. El comando incluido exige Linux, ejecuta pip check durante build y registra RSS, arranque y respuestas. Es validación local, no despliegue Docker en Vercel.

Logs e informes permanecen en artifacts/local/vercel18-checks/, excluidos de Git. No se confunde disponibilidad de wheels con un build de Vercel aceptado ni la memoria de Windows con RSS Linux. Las medidas no prueban todos los picos de concurrencia. Antes de activar proveedores queda pendiente medir adquisición externa dentro del máximo300s y garantizar la copia de datos ante reinicios.

## Resultado final de pruebas

**235/235 pruebas Python (98,786 s)**, incluidas nueve nuevas; **6/6 pruebas JavaScript**. pip check y git diff --check sin errores. Los 42 hashes del manifiesto de código del paquete coinciden. Ningún artefacto privado ni coincidencia con patrones de credenciales/rutas personales en el paquete transmitible. Se conservan intactos los 163 archivos protegidos, 92 entradas del inventario y 21 archivos de presentación. Logs y copias privadas de prueba: artifacts/local/vercel18-checks, fuera del paquete y de Git. Las pruebas generales pueden usar ejemplos temporales de entrenamiento propios; nunca reajustan las redes científicas congeladas.

## Estado de salida

El empaquetado WSGI integral está preparado y probado localmente. Persisten tres grupos de bloqueos: permisos y origen de datos/evidencia; autenticación HF real; instalación/memoria/build y aceptación efectiva de la beta en Vercel. No se ha demostrado un bloqueo nuevo de PyTorch que requiera cambiar a ONNX; el experimento ONNX del Prompt17 permanece conservado como alternativa.

Referencias oficiales: [Python/WSGI](https://vercel.com/docs/functions/runtimes/python), [funciones en /api](https://vercel.com/docs/functions/runtimes/python/api-directory), [Fluid/configuración](https://vercel.com/docs/project-configuration/vercel-json), [Large Functions y límites](https://vercel.com/docs/functions/limitations).
