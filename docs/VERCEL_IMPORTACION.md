# Importación de Brent Forecast desde GitHub

**Actualización diaria de observaciones:** [ACTUALIZACION_DIARIA.md](ACTUALIZACION_DIARIA.md) describe el nuevo canal live/ autorizado, workflow, secretos, control de revisiones y comprobación en la web. Los pins de modelos y paquetes científicos originales se conservan.

**Plan de cierre vigente:** [PLAN_DESPLIEGUE.md](PLAN_DESPLIEGUE.md). Mantener frontend y PyTorch en Vercel; cerrar recursos/permisos y ejecutar la aceptación antes de plantear otro backend. Las secciones siguientes conservan el diagnóstico y evidencias de las revisiones previas.

## Recuperar el despliegue del canal live (19/11/2026)

Causa comprobada en GitHub main7f21279: deployment/vercel-app/vercel.json y su plantilla deployment/vercel/vercel.json contienen git.deploymentEnabled=false. Según la [configuración oficial de Vercel](https://vercel.com/docs/project-configuration/git-configuration), esto desactiva todos los despliegues automáticos por Git. Ese bloqueo se introdujo durante la preparación para impedir despliegues sin autorización; no es un fallo de Hugging Face. GitHub tiene los commits recientes y checks de Actions correctos, sin checks de Vercel en7f21279. La API pública todavía informa automatic_refresh=false y adquisición2026-11-11T21:08:21.056273+00:00.

La corrección local cambia plantilla y snapshot a `{"git":{"deploymentEnabled":{"main":true,"**":false}}}`, sincroniza los hashes y actualiza las pruebas. Vercel permite una rama si al menos una regla coincidente es true, por lo que main se permite aunque también coincida con **. No se han inspeccionado ajustes privados de la cuenta Vercel, cambiado secretos ni iniciado despliegues. Este cambio requiere commit/push autorizado; hacerlo puede disparar un despliegue Production.

### Desplegar ahora el código que ya existe en GitHub

1. En brent-forecast → Settings → Git, confirmar el repositorio Juandeleongarcia/caso-2-prediccion-petroleo conectado. Revisar que la integración Vercel GitHub App mantiene acceso a ese repositorio. No desconectar/reimportar por defecto.
2. En Settings → Environments → Production, confirmar la rama de seguimiento main (según el panel, puede aparecer como Production Branch en Git).
3. Settings → Build and Deployment: Root Directory deployment/vercel-app; preset Other; Build Command node build.mjs; Output Directory public. Mantener detección automática de instalación; no añadir requirements.txt ni un comando pip personalizado. Revisar Ignored Build Step: dejar comportamiento automático, sin un comando personalizado que devuelva0 e ignore siempre. Si existe Skip deployment para proyectos sin cambios, desactivarlo durante este intento.
4. Confirmar que las cuatro variables live ya guardadas pertenecen a Production: BRENT_LIVE_ENABLED=1, BRENT_LIVE_ACCESS_AUTHORIZED=1, BRENT_LIVE_REPO_ID=Juaandeleoon/caso-2-prediccion-petroleo, BRENT_LIVE_POLL_SECONDS=300. Mantener HF_TOKEN de lectura, autorizaciones públicas acreditadas, pins y hashes originales. BRENT_ALLOW_PROVIDER_ACCESS=0 mantiene la adquisición en Actions. No cambiar BRENT_DATA_REVISION para perseguir cada snapshot: el canal live resuelve main de HF y ancla cada lectura a un SHA completo con integridad.
5. Deployments → **Create Deployment** → introducir **7f21279a8bf15b937e576a0d357dd615980be3ba**, rama main/entorno Production. Alternativamente introducir main o su URL para seleccionar la punta vigente. Verificar Source antes de confirmar **Create Deployment**. Este mecanismo manual oficial permite desplegar un Git reference aunque los despliegues automáticos estén interrumpidos. No usar Redeploy del32737f0: reutiliza el código antiguo. Si solo se ofrece Preview, comprobar la rama Production y promover el nuevo despliegue correcto; no promover uno que carezca de las variables Production.
6. Esperar Ready y comprobar que el despliegue nuevo es el que sirve brent-forecast.vercel.app; en los logs verificar Python3.13 y Source correcto. Si la interfaz ofrece caché de build, desmarcar su reutilización para este intento de diagnóstico.

### Comprobar que está leyendo los snapshots publicados

Abrir /api/ready:200 e inference_loaded=true donde se exponga en /api/health. Abrir /api/models: automatic_refresh=true, data_refresh_mode=github_actions y refresh.status=current tras una comprobación satisfactoria. Comparar refresh.published_commit, refresh.snapshot_id y data_updated_at con el SHA/snapshot de la ejecución published_and_verified de Actions y el puntero privado live/latest.json. Si hay un fallo, conservar los controles y revisar refresh.status, refresh.last_attempt_at y los logs del servidor sin imprimir secretos: Ready por sí solo puede indicar inferencia con el bundle anterior.

La comprobación es bajo demanda, como máximo una vez cada300s por instancia; volver a consultar tras ese intervalo y recargar la web. Una adquisición nueva no garantiza una cotización/origen nuevos si no hay observaciones admisibles. Las cinco secciones conservan evaluación histórica congelada; solo la ventana operativa y la fecha de adquisición cambian. No hay que redesplegar cada día ni actualizar pins de pesos para consumir live.

Una vez autorizada/publicada la corrección de git.deploymentEnabled, los siguientes pushes a main podrán generar despliegues automáticos. La actualización diaria de datos es independiente y no hace pushes a GitHub. Las restricciones y recuentos descritos a continuación son evidencias históricas de fases previas, no el estado remoto actual acreditado.

## Auditoría tras configurar Models y subir los ZIP (19/11/2026)

El propietario confirma ambos ZIP subidos al Models privado, HF_TOKEN/BRENT_DATA_REPO_ID configurados en Production y BRENT_DATA_REVISION=6b5f190dc22393ee91d9a1bffe9c17c27672bab1, además de un despliegue nuevo. Esa revisión tiene40caracteres hexadecimales; el contenido remoto y sus hashes no se han comprobado por descarga autenticada, porque esta sesión no tiene HF_TOKEN ni caché oficial de credenciales. No se solicitan tokens por chat. El usuario confirma que los permisos públicos siguen pendientes y BRENT_PUBLIC_ACCESS_AUTHORIZED=0.

Dos bloqueos reproducidos: primero public_access_not_authorized (con Large Functions habilitado), por lo que no se intenta descargar nada. Después, el valor comunicado BRENT_HF_REVISION=b9ad781cd90e1adcb08009358fca143e0bf0931 tiene39caracteres y falla la validación de revisión inmutable. Corregirlo copiando el pin completo de40caracteres: **b9ad781cd90e1adcb080009358fca143e0bf0931**. Si el valor de Production difiere del pegado en el chat, verificarlo allí; no se han leído sus secretos o ajustes desde el panel remoto.

Se puede corregir hoy la revisión de modelos, confirmar el token de lectura y los identificadores/hashes, conservando autorización=0 y proveedores=0. No hay una corrección legítima de código que habilite exposición pública mientras los permisos sigan pendientes. Tras acreditar el alcance requerido, autorización=1 y un nuevo despliegue Production permiten intentar preparación real, siempre sujeta a autenticación/integridad/recursos. [Vercel aplica cambios de variables solo a despliegues nuevos](https://vercel.com/docs/environment-variables).

| Variable | Valor/acción exacta |
|---|---|
| VERCEL_SUPPORT_LARGE_FUNCTIONS | 1, disponible en build y runtime |
| BRENT_PUBLIC_ACCESS_AUTHORIZED | Mantener0: permiso público pendiente |
| BRENT_ALLOW_PROVIDER_ACCESS | Mantener0; no impide inferencia con snapshot verificado |
| BRENT_MODEL_SOURCE | hf |
| BRENT_HF_REPO_ID | Juaandeleoon/caso-2-prediccion-petroleo |
| BRENT_HF_REVISION | b9ad781cd90e1adcb080009358fca143e0bf0931 |
| BRENT_MANIFEST_SHA256 | 0a087144d49fe504ba54d5498688535827f7c83b57913748f8f5f8a87dcf7570 |
| HF_TOKEN | Secreto válido, sin comillas/espacios añadidos; lectura del contenido del Models privado, no solo permiso de Inference Providers |
| BRENT_DATA_REPO_ID | Juaandeleoon/caso-2-prediccion-petroleo si es el repositorio donde se subieron ambos ZIP; sin URL ni /tree/ |
| BRENT_DATA_REVISION | 6b5f190dc22393ee91d9a1bffe9c17c27672bab1; confirmar que ese commit contiene los dos ZIP en la raíz |
| BRENT_RUNTIME_DIR | /tmp/brent-vercel-v1 |

Las URL generadas para los históricos son https://huggingface.co/Juaandeleoon/caso-2-prediccion-petroleo/resolve/6b5f190dc22393ee91d9a1bffe9c17c27672bab1/brent-frozen-v1.zip y la misma ruta con brent-dashboard-v1.zip. No son enlaces públicos confirmados ni incluyen credenciales; una consulta anónima fallida no probaría que falten archivos en el repo privado. Deben descargarse mediante el token solo en el servidor. El commit de datos puede ser posterior y diferente del de los seis modelos. Los hashes esperados, sin modificarlos para aceptar otro ZIP, son2ccf6f181772420dfbd1650c726b72ac6c936f1e2eb1fed82b3ef99c049f7f2a y2b26b26d90383c3ab0c27a563714cae0ca668f94538d7e1bfcdbcd2542a45620, respectivamente.

Orden auditado: configuración/autorización → dependencias Python → ZIP verificados en /tmp/brent-vercel-v1/brent-frozen-v1 y brent-dashboard-v1 → seis archivos verificados en /tmp/brent-vercel-v1/models/REVISION_MODELOS → Study e inferencia. No usar rutas Windows, ZIP extraídos manualmente en Git, ramas main ni otra revisión para eludir un hash. No se requieren BRENT_BUNDLE_DIR, BRENT_DASHBOARD_DIR, BRENT_MODEL_CACHE_DIR o BRENT_LIVE_DIR manuales: el adaptador los deriva del runtime.

Los endpoints públicos siguieron devolviendo health=200/inference_loaded=false y ready/models=503 con el mensaje genérico anterior. No se puede afirmar que fallase el token o el ZIP: con autorización=0 no se alcanza esa etapa. Se publica únicamente el diagnóstico seguro pendiente, sus pruebas y documentación; no se publica ni activa ningún recurso restringido. La regresión rechaza también la revisión mal copiada de39caracteres. Para ver esos códigos, crear un despliegue Production del nuevo commit seguro; con autorización=0 el503public_access_not_authorized seguirá siendo el resultado correcto.

Comprobaciones de esta auditoría:14/14pruebas Vercel en18,627s,6/6JavaScript y regresión focal del SHA mal copiado superadas; pip check y diff correctos. Coinciden44hashes del payload de despliegue,163recursos protegidos,92entradas del inventario y21archivos de presentación. Los dos ZIP locales siguen intactos. Los logs quedan en artifacts/local/vercel-activation-audit, excluido de Git. No se modifican valores remotos de Vercel ni se crean despliegues desde esta sesión.

## Viabilidad de una alternativa sin los ZIP completos (19/11/2026)

**Conclusión: no se ha acreditado una alternativa completa, con las cinco secciones equivalentes y las predicciones originales, que use exclusivamente materiales con permiso remoto confirmado.** Prescindir del contenedor ZIP es técnicamente posible mediante otro contrato de recursos, pero no elimina la necesidad de sus entradas, resultados individuales y autorizaciones. No se implementa otra aplicación ni se sustituye la completa por una demostración. Se conservan los cambios anteriores de diagnóstico y la autorización pública permanece sin activar.

| Sección | Qué puede conservarse con el material ya revisado como publicable | Qué falta para mantener toda su funcionalidad |
|---|---|---|
| Visión general | Explicación del estudio y resumen de métricas agregadas | Histórico Brent y contexto de mercado con procedencia y uso remoto acreditados |
| Predicción | Explicación de los tres modelos y horizontes | Autorización suficiente para pesos/transformadores/cuantiles y una ventana válida de60pasos, con sus valores y disponibilidad |
| Comparación | MAE/MSE/RMSE/R²/sesgo por modelo, horizonte, fase y semilla de las1017filas agregadas | Parejas temporales originales si se muestran curvas real/predicho o evolución puntual del error |
| Evaluación | Métricas y conclusiones agregadas auténticas | Datos individuales de predicciones.csv y evidencia de curvas para reproducir todos los gráficos actuales |
| Metodología | Documentación propia, código, arquitectura, protocolo y limitaciones | No requiere cargar cotizaciones para explicar el método; enlazar el protocolo no concede derechos sobre sus payloads |

El subconjunto existente artifacts/derived/brent-academic-v1/metrics.csv y artifact_manifest.json no contiene y_true/y_pred ni ventanas o pesos. src/oil_forecast/public_study.py lo verifica sin abrir los experimentos. deployment/public_app/app.py ya contiene cinco pestañas, pero las identifica como demostración sin inferencia: tener cinco pestañas no equivale a las cinco secciones completas. No se migra ni se despliega esa alternativa aquí. Las métricas agregadas no determinan las parejas temporales originales y no permiten reconstruir legítimamente curvas ni generar predicciones.

### Dependencias científicas que no pueden sustituirse para eludir permisos

El protocolo original usa diez series, transformadas en veinte entradas (diez valores normalizados y diez indicadores de ausencia), durante60pasos. Son Brent, WTI, gas natural, tipo BCE, USD/EUR, CNY/USD, VIX, índice dólar DX-Y.NYB y futuros GC=F/HG=F. OPEP no forma parte de esas diez entradas. Los identificadores y lags están en configs/data.json. Los transformadores son train-only y permanecen dentro de checkpoints.pt.

No es válido sustituir futuros de oro/cobre por spot, DX-Y.NYB por otro índice dólar ni VIX por otro indicador y afirmar equivalencia. El imputador admite ausencias auténticas, pero retirar deliberadamente cotizaciones para evitar sus permisos cambia entradas/máscaras y puede cambiar las predicciones. Tampoco se pueden inventar fechas de disponibilidad, recuperar precios revisados y atribuirlos al pasado, ni llenar valores sintéticos para completar la ventana. Reobtener una serie de un proveedor permitido exigiría verificar instrumento, unidades, valores, revisiones y disponibilidad; no se ha demostrado que reproduzca los bytes ni las predicciones originales.

### Permisos: qué se ha verificado y qué debe aclararse

| Fuente o artefacto | Hecho verificable | Acción necesaria |
|---|---|---|
| EIA, fuente original de Brent/WTI/gas | Permite reutilizar sus productos propios con atribución y fecha, exceptuando materiales protegidos de terceros | Obtenerlos directamente de EIA sería una vía candidata para esas series, sin cambiar aquí el dataset. No cubre las otras siete ni resuelve automáticamente las condiciones del canal FRED usado en el estudio |
| BCE, tipo de depósito | Permite usos de información obtenida directamente de su web bajo condiciones de exactitud, atribución y declaración de modificaciones | Puede estudiarse como fuente directa del tipo, pero no sustituye las condiciones de todas las demás variables ni acredita igualdad histórica |
| FRED, DCOILBRENTEU/DCOILWTICO/DHHNGSP/ECBDFR/DEXUSEU/DEXCHUS/VIXCLS | Los términos actuales restringen el uso de sus servicios/contenido en desarrollo/entrenamiento ML, también en usos educativos; algunas series son de terceros | Pedir aclaración escrita sobre los términos aplicables al estudio histórico y autorización o confirmación del uso remoto de los checkpoints/derivados y del subconjunto necesario. No se afirma una infracción retrospectiva ni que todo peso sea una copia protegida |
| Cboe, VIXCLS | La ficha FRED identifica copyright de Cboe y reproducción con permiso | Aclarar el alcance para esta app académica pública, uso en inferencia, almacenamiento cloud y visualización; el permiso concedido a FRED no demuestra una sublicencia para esta app |
| Yahoo y titulares de DX-Y.NYB, GC=F y HG=F | yfinance distingue la licencia de su software de los derechos sobre cotizaciones y remite a términos de Yahoo; la página de términos devolvió999 en esta consulta | Solicitar a Yahoo/proveedores aplicables la confirmación del alcance permitido. No se certifica autorización ni se presupone que Yahoo pueda sublicenciar todos los instrumentos |
| Pesos, escaladores y cuantiles existentes | Son derivados separados del dataset; el proyecto no aporta una autorización suficiente para certificar este uso remoto | Determinar con los titulares/institución si los términos aplicables los permiten o requieren autorización. Haberlos subido a Models privado no constituye evidencia del permiso |

Fuentes consultadas: [FRED, términos y contacto](https://fred.stlouisfed.org/legal/), [EIA](https://www.eia.gov/about/copyrights_reuse.php), [BCE](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html), [VIXCLS y titular](https://fred.stlouisfed.org/series/VIXCLS), [yfinance y enlaces a términos](https://github.com/ranaroussi/yfinance). No se encontró evidencia nueva que cierre los permisos pendientes de este proyecto.

### Qué solicitar y qué archivos cubrir

Con apoyo de la UAX si procede, remitir a cada titular aplicable esta descripción, ajustando a sus series: «Proyecto académico individual, sin finalidad de inversión ni comercial, de predicción Brent con RNN/LSTM/GRU. Se solicita confirmar los términos aplicables a la adquisición y entrenamiento histórico y si está permitido conservar y ejecutar los pesos/transformadores/cuantiles congelados en Vercel, almacenarlos en Hugging Face privado, transferir al servidor una ventana de60días con fechas de disponibilidad, publicar predicciones y bandas y mostrar las cotizaciones/parejas históricas necesarias para evaluación. Indiquen atribuciones, alcance, restricciones y si se necesita autorización adicional de terceros. La URL prevista es https://brent-forecast.vercel.app. No se solicita acceso mayorista ni publicación del dataset original». FRED publica stlsFRED@stls.frb.org como contacto; no se ha enviado ningún mensaje. Para Cboe/Yahoo usar sus canales oficiales de permisos/licencias y la asesoría institucional, sin asumir una dirección no verificada.

La confirmación debe cubrir al menos: checkpoints.pt con sus transformadores; intervals.json; latest_window.csv y sus diez series; brent_history.csv para el mercado; las parejas necesarias de predictions.csv y el uso de los gráficos/evidencias históricas. protocol.json, load_models.py y manifiestos son el contrato técnico que deberá seguir verificándose. En la app actual los dos ZIP completos siguen siendo el contrato de carga; la lista anterior delimita permisos y no autoriza eliminar ficheros de esos ZIP ni relajar sus hashes.

Si se autoriza todo el contenido, la opción más simple sigue siendo usar los dos ZIP ya verificados. Si solo se autoriza un subconjunto suficiente, podría prepararse después un paquete nuevo independiente con manifiesto propio, referencias a los originales y pruebas de paridad, sin tocar originales. Si se deniega o no se aclara el permiso de pesos o entradas esenciales, no puede certificarse la misma app pública bajo estas restricciones: conservar la demostración local o acordar otro estudio con datos/modelos autorizados. Un alojamiento privado, otro proveedor cloud o un cambio de formato no resuelve por sí solo los derechos. No se habilita inferencia pública ni se publica ningún artefacto en esta revisión.

Comprobación focal del subconjunto publicable:6/6contratos AggregateStudyContracts de test_public_demo.py superados en0,382s, incluyendo hashes, ausencia de cotizaciones/pares individuales, semillas, fases y métricas reales. El intento inicial tuvo errores de acceso a temporales del sandbox y se repitió con acceso permitido; no se alteraron datos ni producción. No se repiten entrenamientos ni pruebas extensas de inferencia porque esta revisión solo amplía la documentación de viabilidad. Los cambios de código pendientes de la intervención anterior se conservan.

## Web publicada: diagnóstico de los 503 (19/11/2026)

La interfaz pública está en https://brent-forecast.vercel.app. Las consultas reales devolvieron /api/health=200 con inference_loaded=false y /api/ready, /api/models y /api/overview=503. Esto acredita que la función Python y las rutas responden, pero no que los modelos estén cargados. Ready en el panel de Vercel describe el despliegue, no la preparación de los artefactos de la aplicación.

El estudiante confirmó esta configuración Production: BRENT_PUBLIC_ACCESS_AUTHORIZED=0, BRENT_ALLOW_PROVIDER_ACCESS=0, BRENT_MODEL_SOURCE=hf y BRENT_HF_REPO_ID=Juaandeleoon/caso-2-prediccion-petroleo. HF_TOKEN, BRENT_DATA_REPO_ID y BRENT_DATA_REVISION no están configurados. El repositorio Models es privado y contiene los seis archivos mínimos; no contiene los dos ZIP. Es información confirmada por el propietario, no una lectura de sus secretos remotos.

**Causa en orden de ejecución:** settings detiene la preparación por autorización pública=0, antes de consultar Models. Si se resolviese ese permiso, faltaría HF_TOKEN; después faltaría el origen inmutable de los ZIP. No basta con cambiar una bandera ni con hacer público el repositorio. BRENT_ALLOW_PROVIDER_ACCESS=0 no impide inferir con el snapshot válido: solo desactiva consultas nuevas a proveedores y debe mantenerse durante la primera validación.

### Archivos comprobados y para qué sirven

Los originales existen, sus SHA256 coinciden con configs/serving_bundle.json y configs/dashboard_bundle.json, y la comprobación CRC de ambos ZIP pasó. No se reconstruyeron ni modificaron.

| Paquete local | Bytes comprimidos | Contenido y necesidad |
|---|---:|---|
| artifacts/releases/brent-frozen-v1.zip | 176139 | artifact_manifest.json, brent_history.csv, checkpoints.pt, experiment.json, interval_metrics.csv, intervals.json, latest_window.csv, metrics.csv, protocol.json y reference_predictions.csv |
| artifacts/releases/brent-dashboard-v1.zip | 2594014 | artifact_manifest.json, experiment_catalog.json, history.csv, interval_metrics.csv, metrics.csv y predictions.csv |

Total: 2770153 bytes. El primero contiene 7008 filas de histórico Brent y una ventana de60filas con92columnas de valores/metadatos/disponibilidad, de la que el protocolo selecciona20entradas. El segundo contiene1017filas de métricas,1668de curvas y54762predicciones archivadas. history.csv del segundo paquete son curvas de entrenamiento, no nuevas cotizaciones. Estas cantidades describen los paquetes congelados, no datos descargados hoy.

Para inferir se necesitan los pesos auténticos, sus transformadores ajustados, protocolo/orden de entradas, cuantiles y una ventana válida de60pasos con fechas de disponibilidad. Los transformadores ya están dentro de checkpoints.pt; no falta un scaler.pkl externo. Los seis archivos mínimos de Models son checkpoints.pt, protocol.json, intervals.json, load_models.py, README.md y manifest.json. El checkpoint mínimo es idéntico byte a byte al del paquete frozen.

Esos seis archivos no contienen la ventana ni el histórico, ni las parejas real/predicho y curvas necesarias para las otras páginas. Para conservar la aplicación actual completa, la vía mínima de integración es proporcionar los dos ZIP originales además de los seis archivos mínimos; el cargador exige su inventario íntegro verificado. No hay que subir el CSV original, experiments/, .venv/ ni credenciales. Recortar o cambiar un ZIP rompe su hash; un nuevo paquete exigiría otra preparación explícita y no resolvería los derechos de sus datos.

### Derechos y acciones manuales, por orden

1. **Mantener BRENT_PUBLIC_ACCESS_AUTHORIZED=0 y no subir los ZIP por ahora.** Sigue sin acreditarse el alcance de alojamiento, uso remoto de pesos/transformadores, redistribución de cotizaciones y presentación pública de resultados. Los términos actuales de FRED incluyen una restricción de desarrollo/entrenamiento ML; no se afirma retrospectivamente qué versión rigió el estudio. EIA permite sus productos propios con atribución, pero no concede derechos sobre todas las series de terceros. La licencia de yfinance no autoriza por sí sola los datos Yahoo. Solicitar aclaración documentada a los proveedores/titulares aplicables, con apoyo de la UAX si procede. Un repo privado no concede ese permiso. Referencias: [FRED](https://fred.stlouisfed.org/legal/), [EIA](https://www.eia.gov/about/copyrights_reuse.php), [yfinance](https://github.com/ranaroussi/yfinance).
2. Crear en Hugging Face → Settings → Access Tokens un token fine-grained de **lectura**, limitado al repositorio requerido. Guardarlo como HF_TOKEN en Vercel → Settings → Environment Variables → Production, solo en servidor; no copiarlo al chat, al frontend ni a Git. Esto prepara autenticación y no autoriza servir contenidos. [Guía oficial de tokens](https://huggingface.co/docs/hub/security-tokens).
3. **Solo si se acredita el permiso y se aprueba la carga**, subir los dos ZIP originales sin renombrarlos al mismo Models privado (o a otro repositorio de tipo Model autorizado). No se ha ejecutado esa carga. Registrar el commit completo que contiene ambos ZIP como BRENT_DATA_REVISION; no inventarlo ni usar main. Configurar BRENT_DATA_REPO_ID con namespace/repositorio real. Si se usa el actual, será Juaandeleoon/caso-2-prediccion-petroleo. Esa revisión puede ser distinta de BRENT_HF_REVISION: el pin de los seis modelos debe seguir apuntando a sus bytes originales.
4. Verificar descargas autenticadas y hashes de ambos ZIP contra los registros ya versionados, y los seis archivos contra BRENT_MANIFEST_SHA256. Mantener los valores de pesos/revisión/hash de la tabla siguiente hasta validar otra revisión real. Solo después de resolver permisos de exposición, establecer BRENT_PUBLIC_ACCESS_AUTHORIZED=1. Mantener BRENT_ALLOW_PROVIDER_ACCESS=0 hasta comprobar inferencia y aclarar también la adquisición de nuevos datos.
5. Publicar los cambios de diagnóstico únicamente tras autorización de commit/push, crear un despliegue del nuevo commit para que reciba las variables Production y consultar /api/ready. HTTP200 es requisito para continuar: después comprobar las cinco secciones y /api/predict?model=GRU&horizon=5, además de RNN/LSTM/persistencia y63/252. No aceptar health=200 como prueba de inferencia.

Si no se obtiene permiso, la alternativa inmediata es mostrar la aplicación completa localmente a la profesora, conservando la web pública con su aviso honesto de indisponibilidad. Una demostración pública de agregados autorizados sería otra opción, identificada expresamente sin inferencia y sin sustituir aquí la aplicación completa. Cambiar Vercel por otro alojamiento no soluciona derechos; usar datos/modelos con permisos distintos sería una fase nueva y no se realiza en esta intervención.

### Diagnóstico técnico preparado, sin reducir controles

La revisión local añade al503 un code estable y un log de etapa/tipo de excepción. No publica valores de variables, tokens, URLs firmadas, datos ni texto/traceback de la excepción. El cooldown conserva el mismo código y los recursos válidos se siguen verificando. El código no cambia pesos, cuantiles, transformadores ni predicciones.

| code | Qué revisar |
|---|---|
| large_functions_disabled | VERCEL_SUPPORT_LARGE_FUNCTIONS=1 |
| public_access_not_authorized | Permiso de exposición todavía no confirmado; no saltarlo |
| model_authentication_missing | HF_TOKEN presente en servidor y BRENT_MODEL_SOURCE=hf |
| model_configuration_invalid | Repo, commit completo y SHA256 del manifiesto válidos |
| runtime_directory_invalid | Runtime absoluto bajo /tmp en Vercel |
| python_dependencies | Instalación/importación de PyTorch y dependencias; consultar logs de Vercel |
| history_source_missing | BRENT_DATA_REPO_ID y BRENT_DATA_REVISION faltan o no son válidos |
| history_artifacts | Acceso/descarga de ZIP, contenido y hashes; no acredita por sí solo si es token,404o corrupción |
| application_imports | Importación de módulos de la aplicación |
| model_artifacts | Acceso a los seis archivos y verificación de manifiesto/payload; revisar permisos del token, revisión y hashes |
| study_initialization | Carga de redes, transformadores y datos del estudio verificado |

El despliegue público existente conserva el503genérico hasta publicar esta revisión; no se afirma haber aplicado ni probado estos códigos en remoto. La autenticación y descarga privada reales siguen pendientes de configurar de forma segura.

Verificación local de esta revisión: **13/13 pruebas de Vercel (12,754 s)**, **6/6 de descarga/inferencia (1,866 s, un hilo CPU)** y **6/6 JavaScript**; pip check y git diff --check correctos. La simulación de transporte HF emplea los archivos auténticos locales: verifica lógica de descarga/paridad, no credenciales remotas. La primera ejecución de descarga/inferencia sin limitar hilos quedó detenida y se repitió con un hilo, sin modificar modelos. Un fixture de diagnóstico se corrigió para declarar Large Functions antes de comprobar el bloqueo de autorización. Se preservan los163archivos protegidos,92entradas del inventario y21archivos de presentación. Los cinco archivos cambiados son web/vercel_runtime.py, su copia autónoma, source_manifest.json, tests/test_vercel.py y esta guía. No se han hecho commit, push, despliegue ni cargas de artefactos en esta intervención.

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
| Install Command | Automático/sin override; backend desde pyproject.toml + uv.lock. No ejecutar pip/uv desde el instalador estático |
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

### Corrección de la instalación estática previa (posterior a d224d3a)

El segundo fallo no era el mismo conflicto de índices. CLI65.17.0 incluye @vercel/static-build28.0.1 y @vercel/build-utils15.4.1. Con preset Other y buildCommand, el builder estático detecta requirements.txt en la raíz y llama a runPipInstall. Ese helper ejecuta uv pip install --target sin seleccionar --python: puede usar3.14 antes de que intervenga @vercel/python32.0.1, aunque el proyecto restrinja3.13. Añadir otra restricción al mismo pyproject no corrige esa instalación separada.

Se renombra requirements.txt a requirements-linux.txt en plantilla y snapshot. El archivo conserva exactamente las dependencias y el SHA256 del wheel CPU; queda para Docker/comprobaciones manuales. La instalación de la función sigue usando pyproject.toml y uv.lock, sin installCommand personalizado, conservando el empaquetado Large Functions. No se cambia PyTorch, las redes ni la web.

Evidencia: con el builder estático oficial de esa CLI se interceptó la llamada a pip para impedir instalaciones: el paquete anterior la invoca; el corregido compila los siete assets sin invocarla. En Windows se adaptó exclusivamente el lanzador del comando Node, ejecutando el build.mjs real. El analizador oficial0.15.3 descubre los archivos del paquete desde api/ y selecciona3.13, tanto con la raíz del repositorio como con la raíz autónoma. Las pruebas y sus paquetes de diagnóstico quedan en artifacts/local/vercel-python-selection, excluido de Git. No equivalen a ejecutar un build remoto ni a medir memoria Linux.

Verificación de esta corrección: 11/11 pruebas Python de Vercel (14,979 s), 6/6 JavaScript, pip check y diff sin errores. Incluye arranque API e inferencia CPU idéntica a Streamlit en las tres redes, persistencia y horizontes5/63/252. Se comprobó uv sync --locked --dry-run para Linux y la instalación cruzada existente: sus34versiones coinciden con el lock sin cambios. Los163archivos protegidos,92entradas del inventario y21archivos de presentación conservan sus hashes. No se repite la batería científica completa de237pruebas de la corrección previa porque no cambia su código ni sus dependencias. Quedan sin verificar instalación/imports/memoria nativos Linux, autenticación remota de artefactos y despliegue real.

Para el siguiente intento: Root Directory deployment/vercel-app, Other, build node build.mjs, output public; eliminar overrides de Install Command (no configurar uv pip install), guardar y crear un despliegue del nuevo commit de main sin reutilizar la caché. La opción Redeploy de d224d3a reutiliza ese commit y no incorpora la corrección. Verificar en Source el nuevo SHA y en los logs la selección Python3.13 desde .python-version; no continuar dando por válida la inferencia si /api/ready devuelve503. Los permisos, origen de ZIP y credenciales de la tabla siguen siendo requisitos independientes.

Código oficial: [rama del builder estático](https://github.com/vercel/vercel/blob/main/packages/static-build/src/index.ts), [helper runPipInstall](https://github.com/vercel/vercel/blob/main/packages/build-utils/src/fs/run-user-scripts.ts) y [selección del runtime Python](https://vercel.com/docs/functions/runtimes/python/python-version). La reproducción local utilizó las versiones npm exactas indicadas, no únicamente la rama main de las referencias.

### Corrección del instalador de Vercel

Se reprodujo el conflicto torch/filelock con uv0.13.9 y el requirements anterior. Su causa es la prioridad del índice CPU bajo first-index: ocultaba filelock de PyPI. Se conserva torch==2.15.0+cpu mediante su wheel oficial Linux CPython3.13 con el SHA256 original; las demás dependencias se obtienen de PyPI. No se usa unsafe-best-match ni se modifican los pesos. Se añaden pyproject.toml con Python >=3.13,<3.14 y uv.lock para fijar también el instalador basado en proyectos. El motivo exacto por el que el build anterior ignoró .python-version requiere el log completo; ahora ambos mecanismos declaran 3.13.

La instalación cruzada de los 34 paquetes Linux funciona. No acredita imports ni consumo de memoria en Linux: este equipo no tiene Docker/WSL operativo. Dockerfile.check conserva esa comprobación pendiente. Para el siguiente intento, seleccionar el commit corregido, mantener la configuración de la tabla y reconstruir sin reutilizar la caché anterior. Confirmar CPython3.13 en los logs; no desplegar de nuevo el commit fallido. No se requiere ninguna credencial adicional para esta corrección del instalador.

La corrección supera **237/237 pruebas Python (94,494 s)**, **6/6 JavaScript**, pip check y git diff --check. Incluye build CDN, arranque WSGI y paridad con Streamlit de tres redes y persistencia en tres horizontes. Los 163 archivos protegidos, 92 entradas del inventario y 21 archivos de presentación conservan sus hashes. El snapshot importable contiene 45 archivos y 44 hashes de payload, sin artefactos privados. Las 34 versiones instaladas coinciden con uv.lock; se comprobaron 90 bibliotecas ELF y uv sync --locked --dry-run. Los logs permanecen en artifacts/local/vercel-dependency-fix, excluido de Git.

Referencias oficiales: [selección de Python en Vercel](https://vercel.com/docs/functions/runtimes/python/python-version) y [prioridad de índices de uv](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).

Las cifras siguientes pertenecen a la integración previa.

236/236 pruebas Python (101,212 s) y 6/6 JavaScript superadas. El nuevo test comprueba que el snapshot coincide con las fuentes normalizadas y que sus 42 hashes y la desactivación de despliegues Git se conservan. Los 163 archivos protegidos, 92 entradas del inventario y 21 archivos de presentación mantienen sus SHA256. No hay nuevos entrenamientos ni cambios en los experimentos. Se publica solo código y documentación segura; los pesos, ZIP, históricos y credenciales continúan excluidos.
