# Plan de cierre: inferencia real de Brent Forecast

**Ampliación posterior autorizada:** la web ya responde con inferencia HF real. La actualización diaria independiente de pesos se documenta en [ACTUALIZACION_DIARIA.md](ACTUALIZACION_DIARIA.md). La exclusión de Actions en el plan original siguiente correspondía al alcance de aquella fase; la nueva petición la sustituye únicamente para adquisición/publicación de observaciones, sin entrenamiento automático.

Decisión del 19/11/2026. Este documento fija el camino de cierre; la [guía de importación y diagnóstico](VERCEL_IMPORTACION.md) conserva inventarios, hashes, evidencias y el borrador de solicitud de permisos. No se autorizan publicaciones de artefactos ni se activa la exposición pública mediante este plan.

## Arquitectura elegida

Mantener una sola aplicación Vercel: frontend HTML/CSS/JavaScript por CDN → API Python WSGI del mismo origen → PyTorch CPU → modelos y datos verificados desde Hugging Face Models privado. No añadir FastAPI, Render, ONNX, nuevos modelos, bases de datos ni GitHub Actions para resolver este cierre. Streamlit queda como respaldo local íntegro.

La raíz es deployment/vercel-app, preset Other, build node build.mjs, salida public e instalación automática sin override. Python3.13 y torch2.15.0+cpu conservan su lock y wheel/SHA256. requirements-linux.txt no debe renombrarse a requirements.txt: dispararía la instalación estática anterior al selector Python. El build ya ha sido publicado por el propietario; health=200 acredita la función, no la inferencia.

| Componente | Función y control |
|---|---|
| Vercel CDN | Cinco secciones y Plotly; ningún token, peso ni CSV privado en assets estáticos |
| api/forecast.py y web/vercel_runtime.py | Configuración/autorización, descarga fija, diagnóstico seguro, caché temporal y servicio original |
| PyTorch CPU | Ejecutar RNN-D/42, LSTM-B/42 y GRU-C/42; mismos transformadores, protocolo, cuantiles y60pasos |
| Models privado | Seis archivos mínimos, revisión fija y SHA independiente; token solo servidor |
| Repositorio autorizado de datos | Los dos ZIP originales, revisión fija separada y hashes de registros versionados |
| /tmp/brent-vercel-v1 | Caché efímera; no es una fuente durable de datos ni modifica originales |

Vercel documenta Hobby con2GB/1vCPU y300s por función, y Large Functions Python hasta5GB en beta. Es una candidata razonable para este proyecto; que el build pase no certifica memoria, descarga privada o arranque de PyTorch Linux. El uso gratuito queda sujeto a cuotas y condiciones del plan, sin garantía de disponibilidad permanente. [Límites oficiales](https://vercel.com/docs/functions/limitations), [runtime Python](https://vercel.com/docs/functions/runtimes/python).

Render Free dispone de512MB y0,1CPU y suspende tras15min sin tráfico. No hay evidencia de que mover allí este backend completo mejore la estabilidad; la memoria Windows no es RSS Linux y no permite declarar que512MB sean suficientes o insuficientes. Añadiría otro despliegue, secretos, origen y arranque. [Compute](https://render.com/docs/compute-plans), [servicios gratuitos](https://render.com/docs/free).

## Recursos y permisos que deben cerrarse

No se necesita publicar el CSV original, todos los experimentos ni pesos adicionales. Sí se necesita derecho suficiente para el uso remoto de los checkpoints/transformadores/cuantiles existentes, las entradas temporales y las cotizaciones/pares que muestran las gráficas. Un token o un repositorio privado no lo concede. Los términos actuales y la versión aplicable al estudio deben aclararse; no se presupone que los pesos sean copias del dataset ni que estén automáticamente libres de restricciones.

La UAX puede apoyar la consulta a FRED y titulares aplicables (Cboe/VIX, Yahoo y proveedores de sus instrumentos). Solicitar confirmación de alojamiento en Vercel/HF privado, inferencia académica no comercial, transferencia de ventanas, publicación de salidas/bandas y visualización histórica. EIA y BCE permiten determinados usos directos con condiciones; eso no cubre por sí solo las diez entradas ni certifica el uso remoto de los derivados ya entrenados. El detalle y texto de solicitud están en la guía enlazada. No se contacta a terceros desde esta tarea.

La opción más simple, si se autoriza el contenido, es reutilizar exactamente brent-frozen-v1.zip (176139bytes) y brent-dashboard-v1.zip (2594014bytes). Reducirlos no elimina los permisos de su contenido y exige otro contrato de integridad; no se prepara otra variante ahora. Si solo se autoriza parte, comprobar primero si alcanza para todas las funciones. Si no se acredita permiso de pesos o entradas esenciales, el requisito de inferencia pública con estos modelos sigue bloqueado, independientemente del proveedor cloud.

## Secuencia manual única

1. Resolver/documentar los permisos del contenido descrito. Mantener BRENT_PUBLIC_ACCESS_AUTHORIZED=0 entretanto. La exigencia docente no equivale a una autorización de los proveedores.
2. Crear un token HF fine-grained de lectura sobre los repositorios necesarios e introducirlo como HF_TOKEN en Vercel, Production, solo en servidor. No enviarlo al chat ni añadirlo a Git.
3. Solo con permiso acreditado y autorización de carga, subir los dos ZIP originales al Models privado ya existente (o al destino autorizado). Conservar nombres y hashes. Registrar el commit completo que contiene ambos como revisión de datos. No se ha realizado esta carga.
4. Completar las variables siguientes. La revisión de modelos puede seguir siendo anterior a la nueva revisión de datos; no cambiarla por main ni recalcular hashes para aceptar contenido diferente.
5. Verificar descarga autenticada, SHA256 y preparación con los recursos remotos. Solo tras resolver exposición pública, habilitar la autorización. Mantener inicialmente proveedores=0: las peticiones ejecutan realmente las redes sobre el snapshot, aunque entradas idénticas produzcan salidas idénticas. Se deben mostrar su origen/fecha de adquisición y antigüedad de pesos, sin presentarlas como cotizaciones de hoy.
6. Autorizar commit/push del código seguro pendiente y desplegar expresamente su nuevo commit. No pulsar Redeploy sobre uno anterior. Probar readiness, las cinco secciones y las doce combinaciones modelo/horizonte. Registrar consumo y arranque Linux en Vercel.
7. Después de esa aceptación, habilitar adquisiciones nuevas solo si sus permisos están cubiertos y se verifica la política de disponibilidad. Actualizar observaciones no actualiza pesos. /tmp puede perder adquisiciones tras reinicio: antes de prometer actualización durable se necesita un origen autorizado recuperable de esas observaciones.

| Variable Production | Valor |
|---|---|
| VERCEL_SUPPORT_LARGE_FUNCTIONS | 1, build y runtime |
| BRENT_MODEL_SOURCE | hf |
| BRENT_HF_REPO_ID | Juaandeleoon/caso-2-prediccion-petroleo |
| BRENT_HF_REVISION | b9ad781cd90e1adcb080009358fca143e0bf0931; coincidencia autenticada pendiente |
| BRENT_MANIFEST_SHA256 | 0a087144d49fe504ba54d5498688535827f7c83b57913748f8f5f8a87dcf7570 |
| HF_TOKEN | Secreto de lectura, sin valor en documentación |
| BRENT_DATA_REPO_ID | Namespace/repositorio autorizado que realmente contenga ambos ZIP; puede ser el mismo de modelos |
| BRENT_DATA_REVISION | 6b5f190dc22393ee91d9a1bffe9c17c27672bab1, comunicado después por el propietario; contenido/hashes remotos autenticados pendientes |
| BRENT_RUNTIME_DIR | /tmp/brent-vercel-v1 |
| BRENT_ALLOW_PROVIDER_ACCESS | 0 en aceptación inicial |
| BRENT_PUBLIC_ACCESS_AUTHORIZED | 0 mientras no se acredite permiso; 1 solo tras revisión y autorización de exposición |

## Criterios de aceptación y de cambio de backend

- /api/health devuelve200 y /api/ready devuelve200 con source=hf; health por sí solo no basta.
- Las tres redes ejecutan su forward de PyTorch; horizontes5/63/252 y persistencia funcionan, con fechas explícitas y bandas originales. No se sirven predicciones archivadas como inferencia nueva.
- Visión general, predicción, comparación y evaluación reciben datos/gráficos auténticos; metodología conserva el contenido propio. MAE/MSE/RMSE/R², errores y resultados negativos no cambian.
- Una instancia sin caché recupera los recursos fijos; una instancia caliente no vuelve a descargarlos. Fallos de acceso/integridad producen503sin resultados inventados, logs ni respuestas con secretos. Refresh manual público sigue prohibido.
- Medir arranque con descarga en Linux: objetivo de aceptación del probe <240s y respuestas <4500000bytes, bajo el máximo300s. RSS inferior al límite Hobby con margen para atender peticiones. Son criterios técnicos, no requisitos añadidos a la rúbrica del profesor.
- Si el error es de permiso, token, revisión, archivo ausente o SHA, resolver ese recurso: no migrar por ese motivo. Cambiar a backend independiente solo ante OOM, incompatibilidad nativa, duración o cuotas demostradas después de aprovisionar recursos autorizados.

Si se activa el último caso, mantener el frontend y los contratos API; reutilizar el backend Python existente en un servidor Linux con capacidad medida, preferiblemente institucional si la UAX lo ofrece. Su gratuidad/disponibilidad no está confirmada. web/production.py y Dockerfile son preparaciones existentes, no un servicio ya validado: necesitarían aprovisionamiento autorizado, prueba Linux, origen HTTPS/CORS y revisión de dependencias. No contratar ni asumir que Render Free es suficiente. No hay razón técnica demostrada ahora para implementar esa migración.

## Estado de verificación

Preparación local con recursos auténticos y controles activos; no hay token HF real ni permisos nuevos. La prueba de runtime vacío intercepta solo HTTPS, usa los bytes originales de los ocho archivos, exige bearer en servidor y revisiones distintas de pesos/datos, valida hashes y compara las doce predicciones con el servicio original. Espía forward sin sustituir su ejecución para acreditar inferencia neuronal. El primer fixture alcanzaba los ZIP locales del repositorio; se corrigió para usar la raíz autónoma sin archivos privados y probar de verdad el aprovisionamiento.

El paquete autónomo de45archivos se genera y compila localmente. verify_runtime.py pasó16consultas WSGI con caché de prueba ya aprovisionada:6,879s de carga,328638464bytes de pico working set y554946560bytes de pico commit, en Windows. No son RSS Linux, descarga HF real, cold start de Vercel ni una prueba de concurrencia completa. Los logs quedan fuera de Git, en artifacts/local/final-serving-plan/. El comando preparado en Linux es python verify_runtime.py --require-linux --output RUTA_NUEVA.json, desde la raíz autónoma y con secretos/recursos autorizados; no ejecutar como si se hubieran obtenido permisos cuando no existen.

Quedan por acreditar permisos, autenticación y descargas remotas, mediciones Linux y aceptación de la URL con inferencia. No se ha reentrenado, publicado otro recurso, activado autorización pública ni realizado commit/push/despliegue en esta fase.

Pruebas finales de esta preparación:14/14contratos Vercel en16,246s,6/6JavaScript y pip check/diff correctos. La prueba nueva se añade a los controles de autorización, fallo de proveedor, integridad y paridad existentes; no cambia la app ni sus predicciones. Los163archivos protegidos,92entradas del inventario y21archivos de presentación se cotejan por SHA256 antes del cierre. Los cambios técnicos de diagnóstico pendientes de las fases anteriores se conservan; este cierre añade el plan y amplía tests/test_vercel.py.
