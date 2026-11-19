# Prompt 17: comparación final de opciones gratuitas

Comprobación del 19/11/2026. Análisis y pruebas locales; no se ha activado ninguna beta en una cuenta, contratado servicios, cambiado visibilidades, hecho commit/push ni desplegado. La aplicación y su configuración de producción no se han sustituido. Los experimentos ONNX permanecen fuera de Git.

## Recomendación revisada

**Primera opción: frontend y API Python/PyTorch en Vercel Hobby con Large Functions.** Es la arquitectura con menos cambios y conserva pesos privados e inferencia original. Es un candidato técnicamente razonable, **todavía no un despliegue validado**. La conclusión anterior sobre 500 MB solo se aplica al límite estándar; no descarta esta beta de 5 GB.

Orden recomendado: validar Large Functions en el proyecto concreto; mantener ONNX como alternativa si la beta no está disponible o falla el arranque; recurrir a un backend separado únicamente si ambas opciones quedan descartadas. No contratar Render para resolver un problema de tamaño sin antes probar esta vía.

## 1. Disponibilidad y coste de Large Functions

La documentación oficial describe paquetes sin comprimir de hasta 5 GB para Node.js, Bun y Python. Requiere Fluid Compute con Active CPU habilitado. Los proyectos nuevos son elegibles por defecto; los existentes pueden solicitar su uso con la variable de proyecto `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`. Solo las funciones que exceden el límite estándar siguen la ruta beta. No se aplica a Edge.

La guía oficial mantenida por Vercel incluye expresamente la beta de 5 GB en su tabla de Hobby. Las condiciones publicadas no la reservan a Pro. **No se ha comprobado la elegibilidad efectiva de la cuenta/proyecto ni un build remoto**; añadir la variable no demuestra por sí solo que un despliegue haya sido aceptado.

Hobby mantiene 2 GB de RAM, 1 vCPU y un máximo de 300 segundos por petición con Fluid Compute. Las funciones grandes no aumentan esos recursos ni convierten el proceso en un servidor persistente. El límite normal de respuesta/petición de función continúa en 4,5 MB.

Las asignaciones mensuales Hobby publicadas son 4 horas de CPU activa, 360 GB-h de memoria provisionada y un millón de invocaciones. La página de precios no publica un recargo específico por Large Functions: se consumen las cuotas normales. Hobby no ofrece consumo adicional bajo demanda como Pro; superar cuotas puede suspender el servicio. No se promete gratuidad ilimitada ni que una beta mantenga indefinidamente sus condiciones. No activar prueba Pro, add-ons ni cambios de plan.

El anuncio inicial de junio decía que no se admitían Secure Compute/Static IPs; la documentación actual ya los marca compatibles y excluye BYOC. Se prioriza la documentación vigente. Esas funciones no son necesarias para este caso ni se propone contratarlas.

Fuentes primarias: [límites vigentes](https://vercel.com/docs/functions/limitations), [tabla oficial mantenida por Vercel](https://github.com/vercel/vercel-plugin/blob/main/skills/vercel-functions/SKILL.md), [Hobby](https://vercel.com/docs/plans/hobby), [cuotas y facturación Fluid](https://vercel.com/docs/functions/usage-and-pricing), [anuncio de Large Functions](https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size).

## 2. Encaje del backend actual

El wheel oficial Linux CPU cp313 de PyTorch 2.15 ocupa 752.627.128 bytes sin comprimir: excede 500 MB pero es menor de 5 GB. Esto elimina la objeción de tamaño de PyTorch por sí solo. Falta construir y medir **todas** las dependencias Linux, código y recursos; no se presenta el tamaño de un wheel o del entorno Windows como el tamaño final del bundle.

Se midió un proceso nuevo con el backend original, un hilo CPU y proveedores deshabilitados, usando los paquetes locales íntegros:

| Medida local Windows, Python 3.13.7 | Valor observado |
|---|---:|
| Importación en frío y carga de recursos | 5,709 s |
| CPU acumulada, incluida carga y 16 consultas | 6,781 s |
| Máximo working set | 338.026.496 bytes |
| Máximo de memoria comprometida del proceso | 1.096.568.832 bytes |
| Mayor respuesta entre las probadas | 285.680 bytes |
| Consultas API correctas | 16/16 |

Las consultas incluyen histórico máximo, comparación, evaluación y cuatro modelos por tres horizontes. La generación de predicción con gráfico tardó aproximadamente 0,065–0,073 s en este proceso. Son mediciones de una ejecución Windows con datos ya locales, **no latencias Vercel ni medidas de Linux/cgroups**. El working set no es toda la memoria comprometida; no debe utilizarse solo para afirmar que Render 512 MB basta. El margen aparente frente a 2 GB es favorable, pero exige medir arranque, carga remota y concurrencia en el destino.

La descarga HF de seis archivos tiene hasta 30 s por archivo: en condiciones adversas, 180 s teóricos más importación, validación y adquisición de datos pueden acercarse o superar 300 s. Las esperas, redirecciones y reintentos impiden convertir ese cálculo en una cota garantizada. Una actualización completa no debería bloquear cada primera predicción. El arranque y los fallos de red deben medirse por separado, con un presupuesto total y una copia de datos válida disponible.

El sistema de archivos de funciones es de solo lectura salvo `/tmp` (500 MB de scratch por instancia). La caché mínima de modelos, de unos 89 kB, cabe con holgura; las dependencias grandes deben formar parte del bundle de build, no descargarse/instalarse en `/tmp` al arrancar. Configurar `BRENT_MODEL_CACHE_DIR` y `BRENT_LIVE_DIR` allí. No hay garantía de persistencia ni de que dos instancias compartan la última actualización. Véase [filesystem oficial](https://vercel.com/docs/functions/runtimes#file-system-support).

La siguiente preparación, si se acuerda esta arquitectura, requiere un entrypoint WSGI para Vercel que reutilice Application; no lanzar Waitress ni un bucle de servidor dentro de una función. Servir HTML estático y API desde el mismo proyecto/origen evita un proveedor adicional y CORS entre servicios. **La configuración existente sigue siendo la del frontend estático; no se ha cambiado por una configuración integral aún no probada.**

## 3. Prueba experimental ONNX real

Se exportaron copias locales de RNN-D/42, LSTM-B/42 y GRU-C/42 con opset 17, lote fijo 1, ventana fija 60 y 10 entradas originales. No se reentrenó, cuantizó ni alteró ningún checkpoint.

Cada grafo integra mediana guardada, normalización float64, diez máscaras de ausencia, conversión a float32 antes de la recurrencia, salida de tres cambios normalizados, inversión float64, suma del Brent de referencia y bandas con cuantiles originales. La envoltura PyTorch reproduce exactamente las salidas científicas del servicio en los casos probados. Los parámetros de normalización y los cuantiles son diferentes según modelo y se conservan.

Se seleccionaron 48 ventanas del histórico procesado distribuidas entre orígenes posteriores al reajuste de julio de 2024 y el final disponible, más la ventana empaquetada: **49 ventanas por modelo**. ForecastService validó previamente fechas, disponibilidad, esquema y procedencia. No se utilizaron valores reales futuros para ajustar nada ni se recalcularon métricas de test.

Tres comprobaciones independientes de runtime: ONNX Runtime CPU 1.23.2, ONNX Runtime Web 1.23.2 WASM en Node y el mismo WASM en Chrome. Cada runtime produjo 147 inferencias, 441 precios y 882 extremos de bandas comparados con PyTorch. Tolerancia absoluta predefinida: **0,0001 USD/barril**, relativa cero; cien veces menor que un céntimo. No es igualdad bit a bit ni una prueba para todas las entradas posibles.

| Modelo | ONNX, bytes | Máximo CPU, precio USD/barril | Máximo Chrome WASM, precio/bandas USD/barril |
|---|---:|---:|---:|
| RNN-D/42 | 11.684 | 0,000004641832 | 0,000004641832 |
| LSTM-B/42 | 33.549 | 0,000004641832 | 0,000007736387 |
| GRU-C/42 | 25.522 | 0,000006189110 | 0,000006189110 |

Los tres ficheros suman 70.755 bytes. El motor WASM utilizado necesita además su binario de 11.905.541 bytes y módulos JavaScript; los modelos pequeños no implican una descarga total de solo 71 kB. Se probó WASM CPU con un hilo; **no** WebGPU, Safari, Firefox ni un teléfono físico. Los avisos del exportador sobre formas y estados se conservan; estos grafos no se validaron para otro lote, otra longitud o estados recurrentes externos.

Estos resultados demuestran que el cálculo neuronal puede ejecutarse realmente en Chrome. **No demuestran que toda la aplicación se haya migrado a ONNX ni que conserve todas las funciones sin trabajo adicional.** El grafo no valida timestamps de Madrid, días hábiles, procedencia ni antigüedad, ni descarga proveedores, construye ventanas o calcula fechas objetivo. Esas reglas siguen en Python; habría que conservar una API de datos o portar y probar las reglas restantes. Los gráficos históricos y métricas continuarían usando los resultados científicos originales.

El navegador necesita recibir pesos, estadísticas y cuantiles, aunque el origen HF sea privado. Un proxy con token oculto que entrega el ONNX también distribuye el artefacto. No se autoriza esa redistribución por esta prueba local; no incluir HF_TOKEN en cliente, no hacer público Models y no publicar estos derivados. La actualización de proveedores con CORS/autenticación y la persistencia siguen pendientes de resolver también en esta opción.

Fuentes: [exportación PyTorch](https://docs.pytorch.org/docs/2.15/onnx.html), [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), [despliegue y assets del motor](https://onnxruntime.ai/docs/tutorials/web/deploy.html).

## 4. Comparación de arquitecturas

| Opción | Ventaja principal | Recursos/estabilidad | Trabajo y bloqueo |
|---|---|---|---|
| Vercel Hobby + PyTorch Large Functions | Una plataforma, mismos modelos y preprocesamiento, pesos en servidor | 5 GB de bundle beta; 2 GB RAM, 1 vCPU, 300 s; cold starts y cuotas | Primera candidata. Verificar elegibilidad/build Linux/cold start y aprovisionar datos privados. No garantiza disponibilidad ilimitada. |
| Vercel + ONNX en navegador | Inferencia sin RAM/CPU de servidor para la red; cálculo demostrado en Chrome | Depende de equipo/navegador y descarga del motor; pequeña diferencia numérica | Requiere distribuir pesos y portar/conservar validación de datos/fechas. Más cambios; permisos no acreditados. |
| Vercel + Render Free PyTorch | Evita la beta y conserva la API actual | 512 MB RAM, 0,1 CPU; suspensión tras 15 min, arranque de aproximadamente un minuto y disco efímero | Dos plataformas y CORS; memoria/CPU más ajustadas, Linux aún no probado. No se considera más estable por defecto. |

Render concede 750 horas gratuitas por workspace/mes con cuotas de tráfico/build. Sin persistencia gratuita de disco; reinicios pueden perder snapshots. No se debe activar una tarjeta o recursos de pago para compensar límites sin autorización. Fuentes: [planes de cómputo Render](https://render.com/docs/compute-plans), [condiciones gratuitas](https://render.com/docs/free).

## 5. Lo que falta, común a las opciones

1. Comprobar permisos de uso y exposición pública de los pesos, transformaciones, series y derivados. El repositorio privado no demuestra ese permiso. Las observaciones y predicciones devueltas por la API también son contenido expuesto.
2. Probar el acceso HF con secreto oficial del servidor, revisión fija y SHA256. Sigue sin haber token configurado para esa prueba; las pruebas ONNX usaron copias locales auténticas, no una descarga privada nueva.
3. Aprovisionar de forma autorizada los paquetes frozen/dashboard y una ventana actual. Los seis archivos Models no contienen toda esa evidencia. No añadir datos privados al frontend ni publicar ZIP por resolver un problema de despliegue.
4. Preservar la última adquisición válida a través de reinicios/instancias o declarar claramente que se vuelve al snapshot congelado. `/tmp` por sí solo no proporciona continuidad; diseñar una copia versionada autorizada sin introducir servicios de pago.
5. Para Vercel: preparar el empaquetado WSGI completo y runtime CPU separado del entorno científico; medir bundle y memoria Linux. Después, solo con autorización, validar una Preview de Hobby con la beta visible y las cuotas revisadas, modelos/fechas/bandas, fallo HF/proveedor y arranque en frío. No iniciar Render antes de comprobar esta opción.

No se puede garantizar una web pública estable en un proveedor sin desplegar y medir allí. La evidencia actual favorece Vercel Large Functions como **próxima prueba**, no como resultado ya conseguido.

## 6. Evidencias y conservación

Los exports, herramientas y datos del verificador están exclusivamente en `artifacts/local/onnx-analysis/` (ignorado). Archivos: `check_parity.py`, `experiment-v1/parity.json`, tres ONNX, `check_wasm.mjs`, `wasm-node-report.json`, `browser-check.html`, `browser-result.json` y casos numéricos privados. Dependencias aisladas: onnx 1.20.0, onnxruntime 1.23.2 y auxiliares; no se modificó .venv/pyproject/poetry.lock. El servidor temporal del navegador solo escuchó en loopback y se cerró.

La medición del backend está en `artifacts/local/deployment-validation/runtime-v17.json`. Los cuatro artefactos originales directamente leídos por la conversión conservaron sus SHA256 antes/después. Las pruebas generales 226/226 y Node 6/6 corresponden al Prompt 16; no se repitieron sin cambios de código de aplicación. Esta fase añade comprobaciones numéricas CPU/WASM/Chrome y 16 consultas medidas al backend. No presenta las tolerancias de conversión como nuevas métricas científicas de evaluación.

Comprobación final: SHA256 de 163 archivos protegidos y 92 entradas del inventario sin diferencias; los 21 archivos de presentación también intactos. `pip check` sin conflictos y `git diff --check` sin errores. Los ONNX (70.755 bytes en total) y los casos privados siguen excluidos de Git; índice vacío y HEAD original `45929b3952c293d974b971d84f2dc751bb396020` sin nuevos commits. Solo se actualizaron documentos de preparación/comparación en esta fase; no se modificaron los módulos activos de aplicación.
