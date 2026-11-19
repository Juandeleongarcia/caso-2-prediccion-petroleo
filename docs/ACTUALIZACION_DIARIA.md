# Observaciones diarias sin reentrenamiento

## Auditoría inicial y decisión

El 19/11/2026 no había .github/workflows ni workflows/ejecuciones en la API pública de GitHub (ambos recuentos0). Vercel devolvía readiness200/source=hf y adquisición2026-11-11T21:08:21.056273+00:00, con automatic_refresh=false. El actualizador local RefreshController ya validaba entradas y conservaba copias buenas, pero su caché /tmp era efímera y necesitaba visitas/proveedores habilitados. No era un cron durable.

El propietario ha confirmado autorización para adquisición periódica, almacenamiento Actions/HF y uso público académico, y ha elegido live/ en el Models privado existente. Esta confirmación no amplía licencias a terceros ni concede reutilización general. Los interruptores explícitos y secretos siguen siendo necesarios. No se han activado variables remotas ni publicado datos desde esta preparación.

## Diseño

GitHub Actions adquiere diez series y valida un snapshot con las tres redes congeladas. Solo después publica, en un único commit HF, cuatro archivos nuevos y el puntero live/latest.json. Las rutas de pesos, los dos ZIP y resultados científicos no son operaciones de subida. El destino debe ser privado y no haber cambiado desde la lectura inicial (parent_commit/comparación y sustitución). Si falla una fuente, validación o conflicto de commit, no se publica un nuevo puntero. No hay datasets en commits GitHub, artifacts de Actions ni cachés de datos.

Vercel conserva sus pins originales BRENT_HF_REVISION y BRENT_DATA_REVISION. Cuando se habilita el modo live, resuelve main del repositorio de datos cada cinco minutos **por instancia y bajo demanda**, obtiene un SHA completo y descarga puntero/manifiesto/CSV exclusivamente de ese commit. Verifica SHA256, esquema, política temporal y vinculación al manifiesto del modelo. Puede cambiar datos sin redeploy diario, pero nunca pesos ni evaluación histórica. El SHA ancla una lectura coherente; el canal móvil confía en el repositorio privado y sus publicadores autorizados, no proporciona una firma independiente frente a un publicador malicioso.

La pestaña abierta de visión general consulta cada cinco minutos; cualquier petición API comprueba nuevas versiones si vence su intervalo. Actions sí se ejecuta sin visitantes. Una instancia fallida conserva su última copia validada; si reinicia durante una caída de HF, puede volver al bundle congelado. La web indica la fecha realmente disponible, sin garantizar que /tmp sobreviva o se comparta entre instancias.

## Datos y protocolo temporal

- El CSV original data/petroleo_dataset.csv, snapshots científicos, checkpoints, escaladores, cuantiles, métricas y predicciones archivadas son inmutables. El dataset operativo nuevo usa features.csv (mismas92columnas), target.csv (mismas5columnas), observations.csv con procedencia/primera adquisición y manifest.json. No sustituye el CSV original del profesor.
- Se actualizan las diez series de configs/data.json usadas por las redes. OPEP no es una entrada del modelo y no se descarga en este flujo. Se mantienen calendario L–V, decisión08:00Madrid, lags, antigüedad máxima, orden de entradas y horizontes5/63/252. No se cambian instrumentos ni fuentes para esconder fallos.
- La ventana inicial y el histórico provienen del bundle congelado verificado. Las decisiones y objetivos anteriores se conservan; no se recalculan hacia atrás con precios revisados. Se retiene la primera observación no ausente por variable/fecha; correcciones posteriores se cuentan en ignored_revision_count, sin sobrescribir vintages anteriores. Esta política conservadora puede diferir de las últimas revisiones de un proveedor.
- Las observaciones nuevas se conocen a partir del máximo entre su lag original y su primera adquisición efectiva. La procedencia histórica del seed sigue siendo retrospectiva y no se convierte en point-in-time. Los timestamps se serializan uniformemente en UTC para evitar mezclar formatos al concatenar días.
- Si aún no hay Brent admisible en la última decisión (por ejemplo, bootstrap después de08:00), se conserva el origen de predicción anterior y se marca origin_advance_deferred. La adquisición queda guardada para decisiones posteriores; no se antedatan valores para simular actualidad. La UI distingue adquisición, observación, origen y reajuste lógico de pesos.
- Cualquier fuente de entrada ausente, duplicado, infinito, futuro indebido, política/esquema/hash incorrectos o inferencia no válida rechaza la actualización. WTI negativo no se elimina automáticamente: no se inventan límites de precio que descarten acontecimientos reales. Los errores no contienen tokens, URLs firmadas ni cotizaciones en los logs públicos.

## Configuración manual: GitHub

1. Publicar el código y workflow revisados en main. La preparación local por sí sola no habilita Actions. Si GitHub pide habilitar Actions, hacerlo en el repositorio del proyecto.
2. En Settings → Secrets and variables → Actions, crear **HF_LIVE_WRITE_TOKEN** como secreto. Usar un token HF fine-grained con lectura y escritura limitadas a Juaandeleoon/caso-2-prediccion-petroleo. No reutilizar el secreto de solo lectura de Vercel, pegarlo en chat, YAML o README. El publicador usa el SDK oficial2.4.0, sin login persistente.
3. Crear la variable **BRENT_LIVE_UPDATE_AUTHORIZED=1** (el alcance ha sido confirmado) y dejar **BRENT_DAILY_ENABLED=0** durante la primera prueba.
4. Actions → **Brent daily observations** → Run workflow → main → publish desmarcado. Debe terminar con status=validated en el resumen: adquirió fuentes y validó las tres redes, pero no subió nada.
5. Repetir manualmente con publish marcado. Debe terminar con status=published_and_verified y published_commit. Comprobar live/latest.json y live/snapshots/ID/ en HF privado; los modelos y ZIP originales conservan sus pins. Un fallo después del commit durante la verificación de descarga puede dejar un commit publicado válido pero el job fallido: revisar ese SHA; no se afirma rollback remoto ni se deshabilitan hashes.
6. Activar **BRENT_DAILY_ENABLED=1**. Cron:37 6 * * *, es decir06:37UTC (07:37Madrid en invierno/08:37en verano). La hora evita el inicio de hora, pero GitHub puede retrasar ejecuciones; no es un SLA. Schedule se ejecuta desde la rama por defecto y puede deshabilitarse por inactividad de un repo público. Revisar que permanezca activo.

El workflow usa Ubuntu24.04, Python3.13 y un lock de44paquetes con hashes separado del entorno científico/Vercel. checkout/setup-python se fijan por SHA oficial. Solo tiene contents:read y concurrency sin cancelar un job en curso. No se ejecuta al abrir PRs y exige main. Instala y prueba sin acceso a datos privados; el secreto se entrega solo al paso de adquisición/publicación. No hace push a GitHub.

## Configuración manual: Vercel (una vez)

Primero comprobar la primera publicación manual correcta en HF; después añadir en Production:

```text
BRENT_LIVE_ENABLED=1
BRENT_LIVE_ACCESS_AUTHORIZED=1
BRENT_LIVE_REPO_ID=Juaandeleoon/caso-2-prediccion-petroleo
BRENT_LIVE_POLL_SECONDS=300
BRENT_ALLOW_PROVIDER_ACCESS=0
```

Conservar HF_TOKEN de **lectura**, BRENT_PUBLIC_ACCESS_AUTHORIZED=1 y todos los pins/hashes válidos actuales de modelos y ZIP. Este modo desactiva la adquisición directa del backend: la centraliza en Actions. La autorización live es independiente y su ausencia bloquea activar el nuevo modo, no modifica el comportamiento por defecto.

Crear **un despliegue inicial** del código nuevo para aplicar módulo/variables. Después, los datos cambian sin redeploy. /api/models debe indicar data_refresh_mode=github_actions, automatic_refresh=true, refresh.last_success_at, snapshot_id y published_commit. Abrir visión general y verificar que adquisición coincide con live/latest.json y el reajuste lógico sigue01/07/2024. La fecha real de ejecución del entrenamiento no se inventa: no quedó registrada. Probar RNN/LSTM/GRU y los tres horizontes. Comparación y evaluación conservan las evidencias científicas congeladas.

Para verificar cada día: Actions muestra la última ejecución schedule, pasos y Summary; status, snapshot_id, data_updated_at, latest_decision_date, origin_advance_deferred y weights_updated=false. En la web, consultar /api/models tras el intervalo de poll y comparar published_commit con HF. Revisar también observation_date/origin_date: una adquisición correcta en fin de semana no implica una cotización nueva ni una nueva decisión.

Para desactivar sin perder la web: BRENT_DAILY_ENABLED=0 detiene el cron; BRENT_LIVE_ENABLED=0 más un único redeploy devuelve el comportamiento anterior. No borrar modelos ni snapshots. Si HF falla mientras está activo, el código conserva la copia válida de la instancia y comunica el fallo sin presentar el intento como éxito.

## Reentrenamiento manual: procedimiento separado

Este workflow **no entrena** ni pretende ser un botón de entrenamiento completo. Los scripts académicos existentes verifican un protocolo cerrado; no deben reutilizarse sobre datos nuevos cambiando sus resultados. Para actualizar pesos habrá que:

1. Fijar un snapshot de entrenamiento autorizado, sus hashes, cutoff y política de disponibilidad; los datos live conservan primeras adquisiciones y no deben reinterpretarse retrospectivamente.
2. Crear un protocolo/run_id nuevo bajo experiments/ con destino independiente y una separación cronológica nueva, etiquetas maduras para252días, escaladores ajustados solo en train y validación interior. La carpeta live reciente por sí sola puede no tener suficiente historial/etiquetas maduras; no se simula esa suficiencia.
3. Ejecutar entrenamiento solo mediante orden manual explícita, con las arquitecturas y comparación con persistencia, registrar épocas/semillas/curvas y evaluar sin optimizar sobre test. Los originales no se abren en modo escritura ni se usa --resume sobre runs cerrados.
4. Crear un paquete candidato/versionado en otra carpeta/HFprefix o repo, verificar inferencia CPU, hashes, paridad de preprocesamiento, incertidumbre y permisos. No publicar candidatos desde el cron de datos.
5. Tras revisión y autorización, actualizar explícitamente pins/hashes de aplicación, su evidencia/model card y la vinculación del canal live al nuevo manifiesto; desplegar y conservar rollback al pin original. Un modelo nuevo no sustituye silenciosamente las métricas de los modelos antiguos.

La implementación ejecutable de ese protocolo nuevo de entrenamiento es una fase posterior, no un reentrenamiento ejecutado ni una automatización ficticia en esta entrega.

## Evidencia y límites de esta preparación

La descarga local real de las diez fuentes produjo un snapshot válido con adquisición2026-11-19T14:37:07.570534+00:00. Las tres redes CPU validaron su ventana sin ajuste de parámetros. Un ensayo local de la app con ese snapshot y transporte HF simulado superó17consultas, incluidas las doce combinaciones modelo/horizonte. No se publicó ese snapshot. Las pruebas offline cubren append-only, primera adquisición, bootstrap, fuentes ausentes, corrupción, rollback, política y publicación atómica restringida a live/. No prueban credenciales remotas ni una ejecución Actions real.

Logs/datos de prueba: artifacts/local/daily-update-checks/, fuera de Git. El lock de Actions se resolvió para Linux; su ejecución/imports nativos y la primera publicación real deben comprobarse mediante Run workflow. La web actual no se ha desplegado ni se han cambiado secretos desde esta sesión. Sin configurar secretos/variables y publicar el workflow, la actualización diaria no está activa.

Regresión completa:249/249pruebas Python en143,989s y7/7JavaScript; pip check y git diff --check correctos. Una copia limpia de los archivos publicables, sin ZIP ni checkpoints ni artifacts/local preexistente, pasó los7contratos offline del workflow en8,860s. Se corrigió el fixture para crear su directorio temporal en un clon nuevo. El SDK oficial2.4.0 y el YAML se inspeccionaron desde herramientas locales aisladas; no se instalaron sobre el entorno científico. Se cotejan los163recursos protegidos,92entradas del inventario y21archivos de presentación antes de publicar el código. No se ejecuta ningún entrenamiento científico nuevo.

Referencias oficiales: [schedule y dispatch de GitHub](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows), [API de commits HF](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#create_commit), [tokens HF](https://huggingface.co/docs/hub/security-tokens). GitHub no es un reloj exacto y los hashes no sustituyen autorización de los proveedores.
