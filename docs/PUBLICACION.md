# Inventario y preparación de publicación

El código del Prompt 11 está publicado en [GitHub/main, commit 389530b](https://github.com/Juandeleongarcia/caso-2-prediccion-petroleo/tree/389530b78d42a1f76273c8f2608c08cc5c6805ce). Tras autorización expresa se verificó el contenido remoto: 75 rutas cambiadas, 80 archivos totales, sin datos ni pesos. Las secciones de preparación del Prompt 11 se conservan como registro de esa revisión.

**Estado vigente — cierre del Prompt 17:** el estudiante ha autorizado el commit y el push a `origin/main` de los 23 cambios revisados de código, documentación, configuración y pruebas. La aplicación principal sigue siendo `app.py`, con inferencia neuronal real local. Los datos, experimentos, pesos y paquetes restringidos quedan excluidos; su publicación y cualquier despliegue siguen sin autorización.

La **sección I** describe esta revisión final. Las secciones A–H conservan el registro histórico de decisiones y comprobaciones; sus referencias a autorizaciones pendientes corresponden a aquellas fases. El contenido público actual se consulta en [GitHub/main](https://github.com/Juandeleongarcia/caso-2-prediccion-petroleo/tree/main).

## Clasificación y conservación

| Grupo | Contenido | Decisión |
|---|---|---|
| Ejecución | `app.py`, `src/oil_forecast/`, `.streamlit/config.toml` | Publicar código; conservar el diseño. |
| Configuración | `configs/`, `pyproject.toml`, `poetry.lock`, `.gitattributes`, `.gitignore` | Publicar. Los registros de paquetes contienen hashes y URL todavía no provisionadas. |
| Reproducción y pruebas | `scripts/`, `tests/` | Publicar todo el código. Algunas pruebas científicas necesitan los snapshots locales originales. |
| Documentación | README, esta guía y materiales originales del profesor | Publicar los archivos previstos, manteniendo atribución y autoría individual. |
| R² derivado | `artifacts/derived/brent-academic-v1/` | Publicar sus dos archivos: métricas agregadas y manifiesto. Es necesario para las cuatro métricas de la aplicación. |
| Inferencia | `artifacts/local/brent-frozen-v1/` y su ZIP | Conservar íntegros, fuera de GitHub. Destino Models cuando se autorice su contenido. |
| Evidencia portable | `artifacts/local/brent-dashboard-v1/` y su ZIP | Conservar íntegros, fuera de GitHub. Necesarios para la evaluación completa de la interfaz. |
| Investigación | `experiments/`, `data/`, `artifacts/local/research-environment/` | Conservar localmente e ignorar en Git; no confundir con residuos. |
| Evidencias de auditoría | `docs/auditoria/`, capturas y figuras locales | Conservar; permanecen fuera de Git. |
| Entorno | `.venv/`, `poetry.toml` | Conservar localmente; recrear con Poetry en otro equipo. |
| Caché | Bases y auxiliares SQLite de Yahoo | Regenerables; se eliminan solo después de comprobar contenido y alcance. |

La limpieza del Prompt 10 retiró `data/live/.cache/cookies.db`, `tkr-tz.db` y sus cuatro auxiliares `-shm`/`-wal`: seis archivos, 81.920 bytes. También se retiraron sus directorios de caché vacíos. Se conservaron los 25 archivos de auditoría, los cuatro experimentos, los datos y todos los manifiestos, gráficos y paquetes. No se encontraron otros temporales que justificaran eliminar archivos importantes.

## A. GitHub

Repositorio existente: [Juandeleongarcia/caso-2-prediccion-petroleo](https://github.com/Juandeleongarcia/caso-2-prediccion-petroleo). La rama revisada es `main`.

```text
app.py
.streamlit/config.toml
.gitattributes, .gitignore
pyproject.toml, poetry.lock, README.md
src/oil_forecast/                 # código y CSS
scripts/                          # gestión de artefactos y reproducción
tests/                            # contratos científicos y de interfaz
configs/                          # nueve configuraciones y registros públicos
artifacts/derived/brent-academic-v1/
  metrics.csv
  artifact_manifest.json
docs/
  README_profesor.md
  assignment/ENUNCIADO.pdf
  PUBLICACION.md
```

`.gitignore` excluye todo `data/`, experimentos, entornos, cachés, ZIP, pesos, secretos, claves privadas, paquetes instalados y evidencias locales. Ninguna de esas exclusiones elimina sus archivos. El índice también se revisa: ignorar un archivo no retira automáticamente uno ya versionado.

Los marcadores `Your Name`/`you@example.com` se retiraron del metadato opcional de autor; no se inventa una identidad. La identificación académica y la declaración final del estudiante siguen pendientes. No se añade una licencia sin decisión del titular. Las instrucciones de `docs/README_profesor.md` indican copiar los materiales al repositorio personal y subirlos a GitHub: se conservan para esa entrega con atribución, sin atribuirles una licencia general.

La consulta de solo lectura `git ls-remote origin refs/heads/main` confirmó el 17/11/2026 que el remoto y `HEAD` local apuntan a `3281a25ae9879ba236420b6a632596411e6afb2f`. No se ha creado otro repositorio ni alterado el remoto. La identidad configurada de Git utiliza el correo `noreply` de GitHub; no sustituye la identificación académica del estudiante.

### Comandos para el siguiente paso

El inventario siguiente fue autorizado y publicado al terminar el Prompt 11. Estos comandos documentan ese procedimiento histórico; no se vuelven a ejecutar durante el Prompt 12:

```powershell
git status --short --untracked-files=all
git fetch origin
git status --short --branch
git diff --check
git ls-files --cached --others --exclude-standard
poetry check --lock
poetry run python -m pip check
```

Para preparar el índice, añadir únicamente el contenido previsto; esto no publica archivos:

```powershell
git add -- .gitattributes .gitignore README.md pyproject.toml poetry.lock app.py .streamlit/config.toml configs scripts src tests docs/README_profesor.md docs/assignment/ENUNCIADO.pdf docs/PUBLICACION.md artifacts/derived/brent-academic-v1
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
git diff --cached
```

Tras revisar el contenido staged y confirmar identidad y permisos:

```powershell
git commit -m "Prepara Brent Forecast para entrega académica"
git push -u origin main
```

No se necesitan Git LFS ni datos privados para ese conjunto de código y documentación. Los paquetes permanecen fuera del commit.

### Verificación del Prompt 11 · 17/11/2026

La batería completa ejecutada con el intérprete del proyecto supera **178/178 pruebas**, sin fallos ni omisiones, en **87,537 segundos** de unittest (107,821 s incluyendo arranque del proceso). No se reentrenan modelos del proyecto: las pruebas sintéticas trabajan en temporales.

Además se creó una copia temporal de los 80 archivos públicos, inicialmente sin `.venv`, `data/` ni `experiments/`. Poetry 2.5.1 instaló desde el mismo lock en una **venv nueva**, sin cambiar las versiones ni el entorno original. `pip check` y los imports aislados pasan con Python 3.13.7, torch 2.15.0+cpu, NumPy 2.5.4, pandas 3.1.0, Streamlit 1.67.0, Plotly 7.1.0 y yfinance 1.8.0. Se instalaron los dos ZIP originales mediante sus registros SHA256 y las **13/13 pruebas de Streamlit** pasaron, sin omisiones, en **28,818 segundos**. No existían carpetas de investigación en esa copia.

El log de comprobación mezcló codificaciones de salida de Windows: falló su lectura final como UTF-8, después de que los cinco comandos de verificación devolvieran 0. Se recuperó y verificó el log existente sin repetir pruebas; no fue un fallo del proyecto ni de instalación.

La suite completa requiere los datos y experimentos históricos. Un clon público sin ellos no acredita 178 pruebas: algunas científicas se omiten y las pruebas de interfaz requieren ambos paquetes. El README distingue esa batería de `unittest discover -s tests -p test_app.py -v`, ejecutable con los paquetes y su ZIP local. La copia anterior del README se conserva en `docs/auditoria/README_pre_publicacion_prompt11.md`, ignorada por Git, para no perder el registro de fases.

Las rutas Windows/POSIX tienen regresiones automatizadas y el lock incluye wheel Linux, pero **no se ha ejecutado Linux ni un despliegue**. La comprobación de instalación limpia corresponde a Windows/CPU. Si cambia código después de este cierre, repetir las pruebas afectadas antes de publicar.

La comparación antes/después conserva **194/194 archivos protegidos idénticos**, incluyendo datos originales y recientes, experimentos, pesos, transformadores, código, pruebas, configuraciones y archivos del entorno. Coinciden además **24/24 salidas experimentales**, **52/52 entradas históricas** —los dos archivos de entorno contra `research-environment`— y **14/14 payloads** portables, junto a sus dos ZIP y manifiestos. El CSV del profesor mantiene SHA256 `49d80fd7ed1ad32a142d8138554d32066fbb1d82f41bef74911bfd71bb9e64fe`.

La auditoría de los 80 candidatos —79 textos UTF-8 y el PDF— y 85 blobs del historial no detectó tokens reconocibles, claves privadas ni rutas personales reales. Las rutas absolutas deliberadamente inválidas de dos pruebas son fixtures de rechazo. Se verificaron las exclusiones de datos, paquetes, pesos, entornos y secretos; no hay archivos ignorados que ya estuvieran versionados. No se ha añadido una licencia ni publicado contenido externo.

Durante la preparación del Prompt 11 se revisaron **75 rutas —71 nuevas y 4 modificadas—**; después se autorizaron commit y push como `389530b`. Se exportaron los 80 archivos del índice a la instalación temporal: el lock, los dos paquetes y el manifiesto académico también verifican con los bytes que publicará Git. La normalización de finales de línea afecta a cuatro archivos de texto y no cambia los hashes del suplemento académico. La publicación se verificó mediante una copia nueva desde GitHub, con los 80 archivos esperados. El Prompt 12 incorpora cambios locales posteriores sin prepararlos en el índice ni publicarlos.

### Recuperación de recursos conservados fuera de Git

| Recurso | Ubicación local preservada | Recuperación en otro equipo |
|---|---|---|
| Paquete de inferencia | `artifacts/releases/brent-frozen-v1.zip` | Copiar desde un respaldo autorizado; instalar con `manage_artifacts.py install --registry configs/serving_bundle.json` y verificar el destino. |
| Evidencia de la interfaz | `artifacts/releases/brent-dashboard-v1.zip` | Mismo procedimiento con `configs/dashboard_bundle.json`. Los comandos completos están en el README. |
| Investigación original | `data/petroleo_dataset.csv`, `data/raw/2026-11-11-official/`, `data/processed/2026-11-11-official/`, `data/supervised/2026-11-11-official/` | Trasladar las carpetas originales autorizadas conservando estructura y manifiestos; el CSV se coteja con `original_sha256` de `configs/data.json`. No sustituirlos por una descarga revisada. |
| Experimentos y entorno histórico | Cuatro subcarpetas de `experiments/` y `artifacts/local/research-environment/` | Copiar desde un respaldo autorizado; conservar pesos, métricas y manifiestos completos. La batería científica verifica las referencias históricas. |
| Datos recientes y evidencias internas | `data/live/`, `docs/auditoria/`, `artifacts/local/app-screenshots/` | Conservar como respaldo local. No se requieren para inferencia con los paquetes. |

### Inventario del commit publicado 389530b

**75 rutas con cambios: 4 modificadas y 71 nuevas.** Las otras cinco rutas ya versionadas no cambian: `.gitattributes`, `scripts/__init__.py` y los tres materiales originales del profesor. El conjunto público completo conserva 80 archivos. `A` indica nuevo y `M`, modificado respecto al commit anterior a la publicación.

```text
M .gitignore
A .streamlit/config.toml
M README.md
A app.py
A artifacts/derived/brent-academic-v1/artifact_manifest.json
A artifacts/derived/brent-academic-v1/metrics.csv
A configs/app.json
A configs/dashboard_bundle.json
A configs/data.json
A configs/economic_events.json
A configs/final_evaluation.json
A configs/optimization.json
A configs/refinement.json
A configs/serving_bundle.json
A configs/smoke.json
A docs/PUBLICACION.md
M poetry.lock
M pyproject.toml
A scripts/manage_artifacts.py
A scripts/prepare_academic_metrics.py
A scripts/prepare_dashboard_artifacts.py
A scripts/prepare_data.py
A scripts/run_baseline.py
A scripts/run_final_evaluation.py
A scripts/run_optimization.py
A scripts/run_refinement.py
A scripts/run_smoke.py
A src/oil_forecast/__init__.py
A src/oil_forecast/academic.py
A src/oil_forecast/acquisition.py
A src/oil_forecast/app.css
A src/oil_forecast/app_charts.py
A src/oil_forecast/app_data.py
A src/oil_forecast/app_pages.py
A src/oil_forecast/artifacts.py
A src/oil_forecast/evaluation.py
A src/oil_forecast/experiment.py
A src/oil_forecast/inference.py
A src/oil_forecast/insights.py
A src/oil_forecast/models.py
A src/oil_forecast/plots.py
A src/oil_forecast/preprocessing.py
A src/oil_forecast/refresh.py
A src/oil_forecast/supervised.py
A src/oil_forecast/temporal.py
A src/oil_forecast/training.py
A src/oil_forecast/uncertainty.py
A src/oil_forecast/validation.py
A tests/test_academic.py
A tests/test_app.py
A tests/test_app_data.py
A tests/test_artifacts.py
A tests/test_dashboard_artifacts.py
A tests/test_diagnostics.py
A tests/test_early_stopping.py
A tests/test_evaluation.py
A tests/test_experiment.py
A tests/test_final_evaluation.py
A tests/test_fixed_training.py
A tests/test_inference.py
A tests/test_inner_validation.py
A tests/test_insights.py
A tests/test_losses.py
A tests/test_models.py
A tests/test_optimization.py
A tests/test_preparation.py
A tests/test_preprocessing.py
A tests/test_refinement.py
A tests/test_refresh.py
A tests/test_residual.py
A tests/test_supervised.py
A tests/test_temporal.py
A tests/test_training.py
A tests/test_uncertainty.py
A tests/test_validation.py
```

La preparación del índice no ejecuta commit ni push. Este inventario excluye datos, pesos, ZIP y recursos privados.

## B. Hugging Face Models

Se preparan los **dos ZIP originales exactos**. No reconstruirlos ni sustituir sus miembros para publicar.

| Archivo local | Bytes | SHA256 |
|---|---:|---|
| `artifacts/releases/brent-frozen-v1.zip` | 176.139 | `2ccf6f181772420dfbd1650c726b72ac6c936f1e2eb1fed82b3ef99c049f7f2a` |
| `artifacts/releases/brent-dashboard-v1.zip` | 2.594.014 | `2b26b26d90383c3ab0c27a563714cae0ca668f94538d7e1bfcdbcd2542a45620` |

El primero contiene tres checkpoints canónicos de semilla 42, transformadores, cuantiles, protocolo y datos de apoyo. El segundo conserva métricas, curvas y predicciones de ejemplo de los cuatro experimentos. Los experimentos completos y las otras semillas se mantienen localmente.

**Derechos pendientes:** los ZIP incluyen histórico Brent, ventanas con cotizaciones de proveedores como Yahoo y valores reales de evaluación. Sus hashes correctos no acreditan permiso de redistribución. No etiquetar el conjunto como datos libres ni asignarle MIT por defecto. OPEP continúa ausente. Revisar la autorización de cada fuente y la visibilidad del repositorio antes de cualquier carga.

La revisión documental del **17/11/2026** detecta una cuestión adicional: los [términos actuales de FRED](https://fred.stlouisfed.org/legal/) restringen el uso para desarrollo/entrenamiento de ML. No se ha determinado qué versión era aplicable al estudio histórico ni se concluye retrospectivamente una infracción. Antes de nuevas adquisiciones, reproducciones, publicación de datos/pesos o activación pública del servicio, aclarar el permiso con la institución/proveedor. Los experimentos históricos se conservan intactos.

[EIA permite reutilizar sus productos propios](https://www.eia.gov/about/copyrights_reuse.php) con atribución y fecha, pero distingue materiales de terceros; ello no resuelve las condiciones de otros proveedores o del canal FRED. El [README de yfinance](https://github.com/ranaroussi/yfinance#readme) distingue la licencia del software del uso personal de los datos y remite a Yahoo. La consulta de sus términos devolvió HTTP 999: no se ha certificado autorización de redistribución. Las 1.017 filas académicas destinadas a GitHub son métricas agregadas propias y un manifiesto de hashes, sin cotizaciones individuales, predicciones ni ventanas.

La ficha del repositorio Models deberá declarar: RNN-D/LSTM-B/GRU-C; LSTM-B/42 elegida antes de test; persistencia como mejor referencia media; 60 pasos y 20 entradas; horizontes directos 5/63/252 L–V; cuantiles nominales del 80 %; reajuste lógico de julio de 2024 y fecha real de entrenamiento no registrada; MAE operativo a 252 días de 21,525 frente a 11,435 de persistencia y R² −16,927; política temporal, licencias y límites de uso. Incluir nombres y hashes de ambos ZIP y enlace al código publicado.

La [ficha de modelos preparada](../deployment/huggingface/README.md) y el [inventario de 92 archivos](deployment/artifact_inventory.csv) amplían esta revisión. Se preparó únicamente una carpeta local ignorada, `artifacts/local/hf-models-staging-v1/`, con `README.md` y copias exactas de los dos ZIP. No se reconstruyen paquetes ni se publican observaciones por separado. La autorización de carga y el alcance permitido de los datos siguen pendientes incluso si el destino es privado. La sección E contiene los pasos futuros, todavía no ejecutados.

## C. Hugging Face Spaces: descartado para el alcance gratuito

El SDK Streamlit integrado está deprecado; Docker sería la vía compatible. Sin embargo, crear un Space Docker exige actualmente un plan de pago. No se elige ese destino, no se crea un Dockerfile innecesario ni se contrata hardware. CPU Basic sin tarifa horaria no equivale a acceso totalmente gratuito. Fuentes: [condiciones de Spaces](https://huggingface.co/docs/hub/spaces-overview) y [Streamlit en Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit).

## D. Condiciones actuales y alternativa gratuita

Consulta documental del **17/11/2026**, sin crear cuentas, contratar planes ni activar hardware:

| Destino | Condiciones comprobadas | Decisión para este proyecto |
|---|---|---|
| HF Models | Almacenamiento público gratuito sujeto a disponibilidad; 100 GB privados en el plan gratuito. | Preparación de un destino privado para los dos ZIP, sujeto a permisos y autorización. |
| HF Spaces Docker | CPU Basic: 2 vCPU, 16 GB RAM, 50 GB efímeros, sin tarifa horaria; crear Docker requiere plan de pago. Suspensión tras 48 h sin uso en CPU Basic. | No elegirlo como solución enteramente gratuita ni contratar nada. |
| Streamlit Community Cloud | Servicio gratuito conectado a GitHub; suspensión tras 12 h sin tráfico. La documentación publica rangos aproximados fechados en febrero de 2024: CPU 0,078–2 núcleos, RAM 690 MB–2,7 GB y hasta 50 GB de disco, sujetos a cambios. | Elegido para preparación; Linux, capacidad y ejecución remota aún no verificadas. |
| Render Free | 0,1 CPU, 512 MB, suspensión a los 15 min, disco efímero y 750 h/mes por workspace. | Menor margen frente al pico Windows medido; no se elige ni se promete equivalencia con Linux. |

Fuentes: [almacenamiento HF](https://huggingface.co/docs/hub/storage-limits), [condiciones de Spaces](https://huggingface.co/docs/hub/spaces-overview), [suspensión de Spaces](https://huggingface.co/docs/hub/spaces-gpus), [Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud) y [recursos y suspensión](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app); [planes Render](https://render.com/docs/compute-plans) y [límites Free](https://render.com/docs/free).

Community Cloud admite `pyproject.toml` mediante Poetry, según su [documentación de dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). La fase futura debe comprobar su versión efectiva de Python/Poetry, uso del lock y wheel CPU, la base de zonas horarias y memoria en Linux. No añadir ahora un `requirements.txt` paralelo ni prometer una instalación remota ya probada. El pico de 507,3 MiB observado previamente en Windows no acredita consumo Linux ni margen suficiente para concurrencia.

El despliegue futuro gratuito deberá escoger GitHub/`app.py`, aprovisionar ambos ZIP autorizados por HTTPS y SHA256, conservar el suplemento académico y comprobar actualización/fallback sin reentrenar. En cualquier servicio suspendido no se ejecutan actualizaciones de fondo. La última copia válida es local al almacenamiento disponible y el paquete congelado permite recuperar el servicio si ese almacenamiento se pierde.

## E. Preparación operativa del Prompt 12

### Artefactos y estructura Models

El [inventario de 92 archivos](deployment/artifact_inventory.csv) registra rutas relativas, tamaños, SHA256, función, necesidad para la aplicación y estado de autorización. Incluye los paquetes, sus miembros, checkpoints históricos, configuraciones, transformadores embebidos, cuantiles y suplemento académico. Los tres estados portables de semilla 42 coinciden exactamente con sus originales: RNN-D, LSTM-B seleccionada en desarrollo y GRU-C. No falta un escalador externo; cada entrada de `checkpoints.pt` contiene sus transformadores de entrada y objetivo.

La copia local de preparación está en `artifacts/local/hf-models-staging-v1/`, ignorada por Git:

```text
README.md                    # copia de deployment/huggingface/README.md
brent-frozen-v1.zip           # original exacto, 176.139 bytes
brent-dashboard-v1.zip        # original exacto, 2.594.014 bytes
```

La [ficha propuesta](../deployment/huggingface/README.md) explica arquitecturas, semillas, métricas negativas, intervalo nominal, selección previa de LSTM y limitaciones. No inventa licencia, identidad, fecha real de entrenamiento ni URL remota. Los dos ZIP suman 2.770.153 bytes. La preparación local no autoriza su carga, ni siquiera a un repositorio privado; ese destino sigue pendiente de aprobación y permisos.

### Archivos de alojamiento y descarga

| Archivo | Función |
|---|---|
| `pyproject.toml`, `poetry.lock` | Dependencias existentes sin cambios; no se añade un archivo paralelo de requirements. |
| `packages.txt` | Declara `tzdata` para resolver `Europe/Madrid` en Linux. No instala nada en Windows durante esta fase. |
| `.streamlit/config.toml` | Conserva el tema y permite la dirección de escucha del alojamiento; localmente usar `--server.address 127.0.0.1`. |
| `deployment/community-cloud/secrets.example.toml` | Plantilla pública sin token: repositorio, revisión inmutable y arranque automático. |
| `src/oil_forecast/deployment.py` | Verifica recursos existentes e instala solo los ausentes, coordinando sesiones concurrentes del proceso. |
| `src/oil_forecast/artifacts.py` | HTTPS, revisión Models de 40 hexadecimales, SHA256 externo, límites, rutas seguras e instalación sin sobrescritura. |
| `app.py` | Aprovisionamiento optativo al abrir la aplicación; botón existente para reintentar, sin cambios decorativos. |

El token de lectura se acepta exclusivamente para `https://huggingface.co/namespace/repository/resolve/COMMIT/archivo`. No se añade a la URL, no forma parte de claves de caché y no se guarda en archivos. Se conserva la autorización únicamente dentro del mismo origen HTTPS del Hub; se elimina al cambiar de origen, incluso hacia un CDN, y se rechaza HTTP antes de seguirlo. Los errores se muestran sin secretos ni URL firmadas. También se compara el manifiesto instalado con el hash del registro independiente, antes de cargar pesos, tanto al preparar recursos como al cargar paquetes ya instalados.

`BRENT_AUTO_PREPARE=true` activa la instalación al arrancar. Se intenta una vez por sesión si falta un paquete; navegar no repite descargas fallidas. El botón permite reintentar. Si se pierde el disco efímero, una sesión nueva puede recuperar los ZIP desde Models, sin el ordenador del estudiante. El funcionamiento requiere red y permisos efectivos de lectura; no se afirma haber probado un repositorio privado real.

`BRENT_HF_REPO_ID` y `BRENT_HF_REVISION` generan ambas URL. Siguen admitiéndose las dos URL explícitas `BRENT_INFERENCE_URL`/`BRENT_EVIDENCE_URL`. Un token configurado no se envía a una URL externa. El CLI usa `--use-hf-token` para leer `HF_TOKEN` del entorno; no admite el secreto como argumento.

### Publicación futura: pasos pendientes, no ejecutados

1. Aclarar por escrito el alcance permitido de FRED/Yahoo y de cada proveedor: transferencia al alojamiento, almacenamiento, inferencia, publicación de salidas e incorporación de nuevas observaciones. Un permiso académico local o hacer Models privado no resuelve automáticamente esos usos. Mantener los originales congelados.
2. Confirmar cuenta/namespace de Hugging Face, nombre del repositorio y privacidad; completar identificación/declaración académica y revisar la ficha. No se inventa un identificador. Autorizar después los archivos de código del Prompt 12 para su commit/push: el GitHub publicado todavía no contiene estas adaptaciones.
3. Tras autorización de carga, usar el CLI oficial en un entorno de publicación separado, sin añadirlo a las dependencias de inferencia. Autenticarse interactivamente con permisos de escritura, crear explícitamente el repositorio Models **privado** y verificar su privacidad antes de cargar. No usar una carga que cree implícitamente un repositorio público.
4. Revisar los tres archivos del staging, actualizar solo la ficha cuando cambie su estado y comprobar los SHA256 de ambos ZIP. Cargar los tres en un mismo commit. Los comandos futuros son:

```powershell
$hfModels = 'USUARIO_HF/REPOSITORIO_MODELS'
hf auth login
hf repos create $hfModels --repo-type model --private
# Verificar en la interfaz que el repositorio correcto existe y es privado.
hf upload $hfModels artifacts/local/hf-models-staging-v1 . --repo-type model
```

No ejecutar esos comandos hasta resolver permisos y autorización. Si el repositorio ya existe, revisar ese destino en lugar de crear otro. El CLI no se ha instalado en esta fase. Referencia: [CLI oficial](https://huggingface.co/docs/huggingface_hub/guides/cli).

5. Registrar el commit completo de Models que contiene ambos ZIP y obtener un token **solo lectura**, limitado a ese repositorio. El token de escritura del publicador no se usa en la aplicación. No pegar tokens en el chat, repositorio, URL ni comandos con argumentos secretos. Fuentes: [tokens](https://huggingface.co/docs/hub/security-tokens) y [revisiones de descarga](https://huggingface.co/docs/huggingface_hub/guides/download).
6. En Community Cloud, después de autorizar el despliegue, elegir el repositorio GitHub existente, rama `main`, entrada `app.py` y **Python 3.13** en Advanced settings. Verificar que se usa el lock y la wheel CPU; no GPU ni hardware de pago. El registro oficial no describe un paso de facturación; si aparece una solicitud de tarjeta, prueba facturada o pago, detenerse. Fuentes: [registro](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/create-your-account) y [despliegue/Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
7. Copiar la plantilla de configuración al gestor de Secrets de Cloud, sustituir namespace/commit y añadir allí `HF_TOKEN` si Models es privado. La aplicación lee directamente secretos de texto, número y booleano, además del entorno; no presupone que Streamlit exporte los booleanos. No crear un secreto público en Git. `BRENT_AUTO_PREPARE=true` prepara los recursos. La plantilla deja `BRENT_AUTO_REFRESH=false` hasta aclarar el permiso del proveedor. Activarlo posteriormente conserva la política de 24 h, la última copia válida y los pesos congelados.
8. Verificar realmente el build Linux, consumo de memoria y cinco páginas; contrastar doce combinaciones modelo/horizonte, MAE/MSE/RMSE/R², curvas, errores e intervalos. Probar fallo de proveedor, reinicio y nueva descarga tras perder disco. Abrir la URL desde otro dispositivo con el ordenador del estudiante apagado. Registrar solo entonces la URL definitiva y los resultados remotos reales.

### Alcance de las pruebas

Las pruebas locales de descarga privada y arranque en frío simulan el transporte HTTP, pero instalan los dos ZIP originales y ejecutan las cinco páginas sobre pesos reales en CPU. No acreditan disponibilidad de Hugging Face ni un permiso efectivo de un token real. Las pruebas de proveedores fallidos conservan la última copia válida y no dependen de una caída real.

Se reutiliza la venv temporal creada desde cero en el Prompt 11 para comprobar estos archivos nuevos: el pyproject y lock mantienen sus hashes. No se repite una instalación de las mismas dependencias sin cambios. Docker no está disponible y WSL no está instalado, por lo que **Linux, el build de Cloud y la descarga remota real permanecen pendientes**. Las pruebas sintéticas de entrenamiento se ejecutan en temporales y no reentrenan modelos del proyecto.

### Resultado verificado del cierre

| Comprobación ejecutada | Resultado real |
|---|---|
| Batería completa del proyecto | **201/201**, sin fallos ni omisiones, **87,159 s** de unittest; salida 0. |
| Interfaz en la venv aislada, sin `data/` ni `experiments/` | **17/17**, sin fallos ni omisiones, **38,048 s**; salida 0. Se usa el entorno instalado desde cero en el Prompt 11 con el mismo lock. |
| Resolución y dependencias | `poetry check --lock` y `pip check`: correctos, también en la copia aislada. No se instala ni actualiza ningún paquete del proyecto en esta fase. |
| Inventario y privacidad | **92/92** tamaños y SHA256 correctos; **87 candidatos públicos** sin credenciales reconocibles ni rutas personales reales detectadas. La única excepción de secretos en `.gitignore` es la plantilla pública sin token real. |
| Conservación antes/después | **109/109 originales protegidos idénticos**, **24/24 salidas experimentales**, **52/52 entradas históricas**, **14/14 payloads** portables, ambos ZIP y sus manifiestos. |
| Estado Git | HEAD sigue en `389530b`, cero archivos staged; las adaptaciones permanecen locales. `git diff --check` pasa. |
| Linux/Cloud/HF privado real | **Pendiente**, no se presenta como superado. No se han creado recursos, instalado CLI, cargado artefactos ni desplegado la aplicación. |

Las 23 regresiones nuevas cubren el aprovisionamiento privado, URL/revisión/credenciales, redirecciones seguras, CLI sin tokens en argumentos, límites y fallos, instalación idempotente/concurrente y arranque en frío. La revisión adicional detectó dos carencias que se corrigieron: lectura del secreto booleano `false` y anclaje al hash externo de manifiestos ya instalados. Las regresiones confirman cero consultas a proveedores cuando ese secreto desactiva la actualización y rechazo de metadatos alterados antes de `torch.load`. El transporte remoto se simula; los archivos, las redes CPU y las métricas utilizados son reales.

Los bloqueos para la publicación efectiva son permisos de las fuentes y autorización de carga/despliegue; faltan también namespace/repo de HF, revisión remota y token de lectura en el gestor del servicio. No se pide enviar tokens por chat. La capacidad Linux, el ciclo de suspensión/reanudación y el acceso real del profesor se comprobarán después de esas decisiones. Privado y gratuito no equivalen a permiso de uso.

## F. Publicación controlada en Models · Prompt 13

### Destino y lista de carga

El destino indicado es **`Juaandeleoon/brent-forecast-rnn-lstm-gru`**. Se propone un repositorio **Model privado**, no un Space ni un endpoint contratado. No se ha creado ni configurado remotamente; no existe una revisión fijada por esta ejecución. No se presume que un repositorio privado autorice transferir los datos o derivados.

| Archivo o grupo | Decisión actual |
|---|---|
| `deployment/huggingface/README.md` como `README.md` remoto | Único candidato documental preparado. Contiene resultados agregados, atribuciones y limitaciones; cualquier carga necesita autorización expresa. No se ha subido. |
| `brent-frozen-v1.zip` | **Excluido de carga**: contiene histórico, ventana de diez variables y pares de evaluación, además de pesos, transformadores y cuantiles cuya autorización no está acreditada. |
| `brent-dashboard-v1.zip` | **Excluido de carga**: incluye `predictions.csv` con precios reales y referencias individuales; las métricas agregadas restantes no hacen redistribuible todo el ZIP. |
| Checkpoints, transformadores y cuantiles aislados | Excluidos mientras no se acredite el alcance permitido de transferencia/uso de los derivados. No se equiparan automáticamente a un dataset ni se declara una prohibición universal de publicar pesos. |
| Datos originales, snapshots, ventanas y predicciones individuales | Excluidos; conservar intactos localmente. |
| Métricas propias agregadas | Se distinguen de las observaciones originales. El suplemento ya está publicado con el código; no se propone una carga adicional de sus archivos en Models en esta fase. |

Los ZIP originales no se abren para extraer o sustituir miembros destinados a una publicación alternativa. Se conservan intactos y también permanece intacto `artifacts/local/hf-models-staging-v1/`, incluida su ficha anterior. **No cargar esa carpeta:** contiene ambos ZIP excluidos. La ficha vigente es la del código fuente de documentación, no la copia histórica de staging-v1. El inventario registra correctamente los hashes diferentes de ambas fichas.

### Permisos: hechos observados y autorización que falta

| Fuente o contenido | Hecho y alcance pendiente |
|---|---|
| FRED | Las condiciones actuales restringen desarrollo/entrenamiento de ML incluso en los usos educativos descritos. No se ha determinado la versión aplicable al experimento histórico. Se necesita aclarar la base de uso y la transferencia de los datos y derivados al alojamiento y su uso en una app pública. |
| EIA | Permite reutilizar productos propios y solicita atribución/fecha, distinguiendo materiales de terceros. Esa condición no resuelve el canal FRED, Yahoo o Cboe. |
| Yahoo/yfinance | La licencia del software yfinance no es una licencia sobre las cotizaciones. Su documentación indica uso personal y remite a Yahoo. No se ha acreditado autorización para redistribuir las observaciones incluidas o servirlas desde el alojamiento; las consultas de términos Yahoo devolvieron HTTP 999. |
| VIX/Cboe | La ficha VIXCLS identifica copyright de Cboe, cita requerida y reproducción con permiso. No se presupone que el permiso de FRED se transfiera automáticamente a este repositorio y a la aplicación. |
| Estadísticas y pesos | MAE/MSE/RMSE/R², pérdidas y cuantiles resumidos son distintos de precios individuales. Los pesos no equivalen por sí solos al dataset; su formato tampoco demuestra libertad de distribución. El alcance permitido de los derivados sigue sin acreditarse. |

Fuentes primarias revisadas: [FRED](https://fred.stlouisfed.org/legal/), [EIA](https://www.eia.gov/about/copyrights_reuse.php), [aviso del proyecto yfinance](https://github.com/ranaroussi/yfinance#readme) y [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS). No se concluye retrospectivamente una infracción ni se inventa una licencia. El permiso docente para compartir código no sustituye las condiciones de los proveedores.

Para desbloquear los paquetes debe acreditarse la autorización o condición aplicable para: **(a)** transferir y almacenar observaciones y derivados en HF privado; **(b)** distribuir o servir pesos, normalizadores y cuantiles; **(c)** mostrar precios reales, gráficos y predicciones en una aplicación accesible por el profesor o públicamente; **(d)** incorporar nuevas observaciones. Debe verificarse qué titular y qué condiciones cubren cada uso. Una autorización de subida del estudiante no acredita por sí sola esos permisos.

### Autenticación segura

Se comprobó el entorno actual sin leer contenidos de credenciales: no hay ejecutable `hf`/`huggingface-cli`, SDK `huggingface_hub`, token en las variables oficiales comprobadas ni archivo de token en la ubicación oficial configurada. **No hay una sesión de terminal utilizable y verificada.** La sesión del navegador no se ha comprobado; crear una cuenta o iniciar sesión web no acredita una sesión del CLI.

No se instalaron dependencias ni se solicitó un token mientras la carga está bloqueada. Cuando proceda, preparar el CLI oficial en un entorno de publicación separado, siguiendo su [instalación oficial](https://huggingface.co/docs/huggingface_hub/guides/cli), y ejecutar:

```powershell
hf auth login
hf auth whoami
```

Elegir autenticación oficial por navegador cuando esté disponible y comprobar que la identidad corresponde a `Juaandeleoon`. No usar `hf env` para recopilar configuración personal ni pegar tokens en comandos, README, Git o chat. La gestión interna de credenciales corresponde al mecanismo oficial; no se copia un token manualmente a archivos del proyecto. Para la aplicación futura se utilizará un secreto de lectura limitado al repositorio, separado de la autenticación de escritura del publicador. Referencias: [CLI/autenticación](https://huggingface.co/docs/huggingface_hub/guides/cli) y [ubicaciones oficiales de credenciales](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

### Verificación local y continuación del Prompt 14

Se volvieron a comprobar tamaños y SHA256 de ambos ZIP contra sus registros, sus 16 miembros totales y hashes de payloads. Se ejecutó inferencia real local con las tres redes en CPU y persistencia: **12 combinaciones modelo/horizonte**, con 5/63/252 días e intervalos originales. **92/92** filas del inventario mantienen tamaños y hashes correctos tras actualizar únicamente la ficha vigente. La verificación no entrena, no recalibra ni accede a proveedores.

Las pruebas remotas —creación privada, descarga autenticada, SHA256 descargado y posterior inferencia— están **pendientes**, porque no se publica contenido bloqueado. No se repite la batería del Prompt 12 de 201 pruebas: los cambios del Prompt 13 son documentales y de plantilla, sin cambios de inferencia.

Para continuar con el Prompt 14:

1. Resolver y documentar los permisos indicados; mientras falten, mantener ambos ZIP y los derivados aislados excluidos. No renombrar ni reconstruir artefactos para eludir el bloqueo.
2. Preparar autenticación oficial y comprobar el namespace. Revisar identidad/declaración académica y la ficha actualizada. Si cambia el alcance autorizado, actualizar la lista exacta de archivos elegibles.
3. Presentar destino completo, privacidad, archivos y hashes, y solicitar autorización expresa para la creación/carga. **No ejecutar los comandos de carga del Prompt 12 con el bloqueo vigente.**
4. Solo después: crear o comprobar el destino privado, publicar exclusivamente los archivos aprobados y fijar el commit completo real. Descargar los archivos aprobados en un directorio nuevo, verificar SHA256 y probar las tres redes CPU si los ZIP están autorizados.
5. Configurar `BRENT_HF_REPO_ID` y `BRENT_HF_REVISION` reales y un secreto de lectura en la plataforma. Los cambios locales del Prompt 12/13 todavía necesitarán autorización para publicarse en GitHub. El despliegue de Streamlit requiere una fase autorizada distinta; no se ejecuta aquí.

## G. Cierre del Prompt 14: demostración pública de resultados

### Estado y alcance

La interrupción anterior dejó evaluada la alternativa, pero no había archivos de implementación de la demostración. Se completaron únicamente esa entrada, su cargador ligero, la exportación explícita, las dependencias mínimas y sus pruebas. No se añaden nuevas funciones de predicción ni se modifica `app.py` o ningún recurso científico original durante esta fase.

Las cinco páginas son Visión general, Predicción, Comparación, Evaluación y Metodología. **Todas muestran «Demostración de resultados · Sin inferencia neuronal».** Predicción explica el bloqueo y muestra únicamente errores históricos reales de LSTM-B/42 y persistencia; no hay campos de precio, simulaciones, cargas de modelos ni pronósticos nuevos. Comparación y Evaluación separan experimento, variante, conjunto, semestre, horizonte y semilla; conservan R² negativos y mejoras desfavorables. Las medias se calculan entre métricas por semilla disponible, con recuentos visibles, sin multiplicar el número de orígenes.

La fuente es el suplemento de métricas agregadas ya publicado: 1.017 filas y 14 columnas, con al menos 111 observaciones por grupo. No contiene precios, `y_true`, `y_pred`, ventanas ni fechas de predicciones individuales. Su CSV y manifiesto permanecen idénticos. El cargador ligero verifica un SHA256 de manifiesto independiente y después el tamaño y hash del CSV; no abre los ficheros históricos citados en la procedencia.

| Función | Aplicación completa preservada | Demostración limitada |
|---|---|---|
| Tres redes y persistencia | Inferencia real CPU con pesos congelados | Consulta de resultados históricos agregados |
| Horizontes 5/63/252 | Precios, cambios e intervalos directos | Errores reales por horizonte; sin precios futuros |
| Histórico y diagnóstico | Observaciones, predicciones y curvas originales | Métricas y errores agregados por semestre |
| Actualización de datos | Mecanismo existente con última copia válida y sin reentrenar | No consulta proveedores; estudio estático congelado |
| Recursos externos | Dos paquetes originales pendientes de permiso | Ningún paquete Models ni credencial |

Esta alternativa permite una URL de consulta académica independiente del ordenador del estudiante cuando se autorice desplegarla. **No sustituye la aplicación completa ni satisface por sí sola el requisito docente de probar inferencia neuronal.** No se ha encontrado una vía demostrada para servir esa inferencia remota sin alojar o utilizar pesos cuyo permiso esté acreditado. Tampoco se presenta el portal como una solución retrospectiva a las condiciones del estudio histórico.

### Lista exacta de exportación de la demostración

El constructor `scripts/prepare_public_demo.py` copia únicamente estos **nueve archivos**, conservando rutas, y rechaza destinos existentes:

```text
deployment/public_app/app.py
deployment/public_app/requirements.txt
deployment/public_app/constraints.txt
.streamlit/config.toml
src/oil_forecast/__init__.py
src/oil_forecast/public_study.py
src/oil_forecast/app.css
artifacts/derived/brent-academic-v1/metrics.csv
artifacts/derived/brent-academic-v1/artifact_manifest.json
```

Ejemplo local, sin publicación:

```powershell
poetry run python -B scripts/prepare_public_demo.py --output artifacts/local/public-demo-review
poetry run streamlit run deployment/public_app/app.py --server.address 127.0.0.1 --server.port 8502
```

El destino del ejemplo debe ser nuevo. La copia de comprobación de esta fase se hizo en un directorio temporal; no se reemplaza un paquete anterior. El constructor verifica los hashes de los agregados antes de copiar y devuelve un inventario de bytes/SHA256. La exportación no incluye datos originales, ZIP, checkpoints, transformadores, cuantiles, predicciones individuales, secretos, `.venv`, `data/`, `experiments/` ni el código de carga de la aplicación completa.

### Dependencias y condiciones gratuitas

Community Cloud admite un archivo de dependencias situado junto a la entrada y lo busca antes que en la raíz. Por eso esta aplicación alternativa independiente tiene su propio `deployment/public_app/requirements.txt`: NumPy 2.5.4, pandas 3.1.0, Streamlit 1.67.0 y Plotly 7.1.0. `constraints.txt` limita sus dependencias a las 35 versiones de su cierre presentes en el lock existente. Las restricciones no instalan paquetes por sí mismas. La aplicación completa conserva su pyproject, lock y entorno originales.

Esta es una excepción deliberada a la recomendación anterior de evitar archivos paralelos para una misma aplicación: las dos entradas tienen alcances y dependencias distintos. No hay Torch, yfinance, GPU, token ni servicio de pago en la demostración. El exportado conserva las rutas para usar la misma entrada en local y en Cloud.

Seleccionar **Python 3.13** en Advanced settings. Community Cloud publica servicio gratuito, suspensión por inactividad y límites aproximados de CPU/memoria/disco; no se garantiza disponibilidad continua. Sus rangos de recursos documentados están fechados en febrero de 2024, no son una medición de este proyecto. Si se solicita tarjeta, facturación o un recurso de pago, detenerse. No se ha instalado Docker o WSL ni se ha ejecutado un build Linux.

Referencias oficiales ya revisadas: [dependencias y prioridad](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [Python y despliegue](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [servicio](https://docs.streamlit.io/deploy/streamlit-community-cloud) y [límites](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

### Verificación final de esta alternativa

El comando `poetry run python -B -W error -m unittest discover -s tests -p test_public_demo.py -v` supera **9/9 pruebas**, sin fallos ni omisiones, en **9,249 segundos**, salida 0. Incluye hashes y rechazo de alteraciones, coincidencia de las doce medias finales y semilla 42 con el CSV original, separación de conjuntos/semestres/variantes, cinco páginas y selectores, R² negativos y la exportación de nueve archivos. Una prueba ejecuta la copia en un proceso aislado que impide importar PyTorch, yfinance y los módulos privados, y bloquea conexiones externas. Solo se permiten los sockets de loopback que necesita internamente asyncio/Streamlit en Windows. El arnés se corrigió para distinguirlos y para leer UTF-8; no fue necesario cambiar la aplicación por esos ajustes de pruebas.

Los **231 archivos originales protegidos** permanecen idénticos al inicio de esta fase, incluidos `app.py`, módulos de la aplicación completa, checkpoints, transformadores, cuantiles, experimentos, datos, configuraciones y evidencias previas. También coinciden **92/92** tamaños y hashes del inventario anterior. El nuevo test de la demostración se excluye de ese recuento de originales. Se preservan los resultados de las 201 pruebas anteriores; no se repiten entrenamientos ni la batería completa sin cambios de sus componentes.

Se comprobó con el parser de pip que los cuatro requisitos directos y las 35 restricciones se leen correctamente y coinciden con las versiones ya instaladas. No se instalaron dependencias en el entorno original. La exportación temporal contiene **185.529 bytes en nueve archivos**, sin datos individuales, estados de redes, tokens ni rutas personales reales detectadas. No quedan archivos staged y HEAD sigue en `389530b`; `git diff --check` pasa.

**Pendiente de ejecución real:** instalación de esas dependencias en un entorno mínimo nuevo, Linux, build de Community Cloud, prueba de capacidad y revisión desde una URL pública. Se creó una venv temporal de comprobación, pero no se instalaron paquetes ni se presenta como una instalación limpia superada. No se han revisado visualmente nuevos navegadores ni se afirma un despliegue remoto verificado. Estas comprobaciones se completarán, cuando corresponda, después de autorizar la publicación; no se abre un servicio como efecto de las pruebas locales.

### Qué puede publicarse y qué permanece bloqueado

| Contenido | Estado |
|---|---|
| Código/documentación propios y métricas agregadas de la exportación | Candidatos revisados para una demostración de resultados, sujetos a autorización del estudiante. No se inventa una licencia ni se declara legalidad retrospectiva del estudio. |
| Datos históricos originales y valores reales dentro de predicciones | Excluidos; el alcance de transferencia y visualización permanece sin acreditar. |
| Checkpoints RNN/LSTM/GRU | No se equiparan automáticamente al dataset, pero su transferencia/uso remoto sigue sin permiso acreditado. |
| Transformadores, normalizadores y cuantiles | Derivados separados; autorización de transferencia y uso remoto aún pendiente. |
| Dos ZIP originales y staging-v1 | Conservados intactos y excluidos de carga. Hacerlos privados no resuelve los permisos. |

EIA permite sus productos propios y distingue terceros; las condiciones actuales de FRED y el alcance de Yahoo/Cboe siguen pendientes para el servicio completo. No apareció una autorización nueva. La alternativa evita distribuir esos archivos; no los oculta dentro del código ni cambia su formato para eludir el bloqueo. Véanse las fuentes y el desglose de la sección F.

### Publicación e intervención imprescindibles

No se han creado recursos ni modificado GitHub. Antes de obtener una URL hay que aprobar expresamente que la entrega pública inicial sea una **demostración parcial**, no la aplicación de inferencia completa.

La propuesta mínima para actualizar el repositorio existente contiene estos **nueve archivos con cambios**; cuatro archivos de la exportación ya están publicados e intactos (`__init__.py`, CSS y los dos agregados):

```text
.streamlit/config.toml
README.md
docs/PUBLICACION.md
deployment/public_app/app.py
deployment/public_app/requirements.txt
deployment/public_app/constraints.txt
src/oil_forecast/public_study.py
scripts/prepare_public_demo.py
tests/test_public_demo.py
```

Los demás cambios locales anteriores no se incluirían automáticamente en ese commit. Tras autorización de esa lista y del despliegue: publicar el código revisado en el GitHub existente, crear la app gratuita desde `Juandeleongarcia/caso-2-prediccion-petroleo`, rama `main`, entrada `deployment/public_app/app.py`, Python 3.13 y **Secrets vacío**, con visibilidad pública revisada. Después comprobar build, cinco páginas y reanudación desde una sesión nueva, registrar consumo y URL real y abrirla desde otro dispositivo sin depender del ordenador del estudiante. No se ejecutan esos pasos aquí.

Para desplegar **la aplicación completa**, antes debe acreditarse permiso para alojar y servir datos y derivados, además de aprobar los archivos exactos y destino. Después hará falta autenticar Models, publicar solo artefactos autorizados, fijar su commit real, verificar descarga/SHA256 e inferencia CPU y configurar un secreto de lectura. Los pesos, métricas y resultados científicos originales no se alteran para resolver ese proceso. No hay URL completa disponible mientras esos pasos sigan pendientes.

## H. Entrega académica y actualización final de main

### Alcance y archivos propuestos

Se propone actualizar **23 rutas: diez modificadas y trece nuevas**, sobre `main` en `389530b78d42a1f76273c8f2608c08cc5c6805ce`. El árbol candidato completo contiene **93 archivos** (80 anteriores y trece nuevos), incluyendo los materiales docentes y los dos agregados ya publicados. Ningún archivo científico congelado se añade ni se sustituye. Esta lista sustituye la propuesta reducida de la sección G para el siguiente commit; sigue pendiente de autorización expresa.

**Modificados:**

```text
.gitignore
.streamlit/config.toml
README.md
app.py
docs/PUBLICACION.md
scripts/manage_artifacts.py
src/oil_forecast/app_pages.py
src/oil_forecast/artifacts.py
tests/test_app.py
tests/test_artifacts.py
```

**Nuevos:**

```text
deployment/community-cloud/secrets.example.toml
deployment/huggingface/README.md
deployment/public_app/app.py
deployment/public_app/constraints.txt
deployment/public_app/requirements.txt
docs/deployment/artifact_inventory.csv
packages.txt
scripts/prepare_public_demo.py
src/oil_forecast/deployment.py
src/oil_forecast/public_study.py
tests/test_deployment.py
tests/test_public_demo.py
tests/test_remote_artifacts.py
```

No hay staging automático ni autorización de commit, push o despliegue en esta fase. El destino del código, cuando se autorice, será exclusivamente `Juandeleongarcia/caso-2-prediccion-petroleo`, rama `main`.

### Verificación del cierre

- **210/210 pruebas**, sin fallos ni omisiones, en **754,412 s**, salida 0; se usa UTF-8 y las advertencias Python se tratan como errores. Una primera ejecución detectó únicamente el desacuerdo de codificación de dos subprocesos Windows del arnés; corregido sin cambiar producción ni resultados. Su batería focal pasa 4/4 (13,859 s).
- Lock y dependencias coherentes (`poetry check --lock`, `python -m pip check`) y diff sin errores de espacios (`git diff --check`). Sin instalaciones nuevas.
- Aplicación original comprobada con los artefactos locales: tres redes reales y persistencia, tres horizontes y cinco páginas. Revisión Chrome de escritorio y viewport móvil 390 × 844, sin acreditar un teléfono físico ni un despliegue.
- Cotejo SHA256: 234/239 archivos de la auditoría previa idénticos; los otros cinco son cambios intencionados de código de presentación, prueba y documentación. Los 163 recursos protegidos y las 92 entradas del inventario permanecen idénticos. Sin nuevos entrenamientos del estudio.
- No se encontraron cachés o temporales del código que justificaran una nueva eliminación. Los logs de verificación se conservan exclusivamente bajo `artifacts/local/`, ignorados.

### Conservación y permisos

| Permanecen excluidos y conservados | Motivo |
|---|---|
| `data/`, incluido `petroleo_dataset.csv`, snapshots y copias de actualización | Datos originales/derivados y permisos de las fuentes; no son temporales. |
| `experiments/` completo | Evidencias, predicciones, historiales, checkpoints y resultados originales congelados. |
| `artifacts/releases/*.zip` y `artifacts/local/` | Pesos, transformadores, cuantiles, históricos, paquetes instalados, copias y verificaciones locales. |
| `docs/auditoria/`, capturas locales | Evidencias de trabajo conservadas, sin añadir informes internos al repositorio público. |
| `.venv/`, cachés y configuración local de Poetry | Recursos regenerables/locales; no se necesitan en Git. |
| `.env*`, secretos reales, claves y credenciales | Información privada; solo se permite la plantilla sin valores secretos. |

El escaneo de los candidatos no detectó tokens reconocibles, claves privadas ni rutas personales reales. Las credenciales ficticias de pruebas de rechazo no son credenciales del estudiante. Esto no convierte en publicables los ZIP ni autoriza nuevas adquisiciones. La ficha de Models conserva las restricciones; no se asigna una licencia de datos o pesos inexistente. La ausencia de una licencia abierta propia no impide presentar el código autorizado, pero no permite anunciar permisos de reutilización que no se han concedido.

### Aplicación completa local y disponibilidad remota

La instalación y arranque reproducibles están en el README: obtener ambos ZIP de una copia autorizada, verificar su SHA256 mediante los registros de `configs/`, instalar en destinos nuevos y arrancar `app.py`. Si ya están instalados, solo verificarlos. Con ambos paquetes se muestran inferencia, comparación, predicciones originales y curvas sin necesitar `data/` ni `experiments/`. Estas dos carpetas sí son necesarias para reproducir íntegramente la investigación y su batería científica.

La actualización de datos conserva una última copia válida y no reentrena. Sus pruebas utilizan proveedores controlados; no certifican disponibilidad continua ni autorización de los proveedores externos. El arranque recomendado para la defensa pausa adquisiciones automáticas, manteniendo toda la inferencia neuronal local.

La aplicación está preparada para CPU y aprovisionamiento con revisión fija: **es una candidata técnica a un servicio gratuito, no un despliegue certificado**. Faltan ejecución real Linux, consumo/límites y pruebas del servicio remoto. El bloqueo de distribución/uso de datos y derivados es independiente de esa viabilidad técnica; hacer un repositorio privado no lo resuelve. No se necesita una URL pública para demostrar aquí la aplicación completa local, salvo una exigencia docente adicional.

Para una URL completa: resolver el permiso del contenido exacto, autorizar la carga, verificar los dos paquetes desde una revisión remota real e inmutable y configurar un secreto de lectura cuando corresponda; después autorizar y probar el despliegue. La alternativa de resultados agregados puede plantearse separadamente, identificada como demostración sin inferencia. Ninguna se publica en este cierre.

### Pendientes para la entrega

- Autorizar el commit/push de la lista anterior; no incluir artefactos bloqueados.
- Confirmar identidad y declaración de autoría individual/asistencia de IA conforme a las normas del centro.
- Preparar la presentación con **máximo cinco diapositivas de contenido**, sin contar portadas.
- Para ejecución por el profesor en otro equipo, acordar acceso autorizado a los dos paquetes. No hay una URL de descarga aprovisionada.
- Declarar la ausencia de ORB, los retardos asumidos y las revisiones retrospectivas; no afirmar disponibilidad histórica certificada ni ausencia absoluta de fuga.
- Mantener la conclusión negativa: persistencia vence a las redes en media de test en los tres horizontes. Las 4/3/5 épocas finales son la regla de mediana de validación; el total acreditado es de 204 ajustes principales más tres repeticiones, 1.481 épocas.

## Reproducción y límites de la preparación

NumPy 2.5.4 se declara directamente porque el código lo importa; ya estaba instalado y fijado en el lock. El lock conserva 53 entradas para 52 paquetes distintos; no se han cambiado dependencias durante el cierre final. Las dependencias de figuras científicas siguen siendo opcionales y separadas. Los dos archivos históricos del entorno de investigación se conservan intactos.

Las claves de rutas Windows de los manifiestos históricos se interpretan de forma portable, sin reescribir los archivos originales ni cambiar folds o resultados. Esto se prueba con rutas POSIX y con los 52 hashes de entrada; no equivale a haber ejecutado el estudio en Linux.

Un clon de GitHub contiene estructura, código y métricas agregadas. La aplicación completa requiere los dos paquetes externos autorizados. Reproducir exactamente la investigación requiere además los snapshots originales y el entorno archivado; una descarga actual puede contener revisiones. Algunas pruebas se omiten si faltan esos recursos, por lo que un resultado parcial sin ellos no certifica toda la investigación.

El commit/push original fue completado; el Prompt 17 autoriza expresamente la publicación de las adaptaciones de código revisadas, con los artefactos restringidos excluidos. La identificación académica y la declaración final del estudiante siguen pendientes; puede conservarse el código sin una licencia de reutilización hasta que su titular decida. Los permisos de las fuentes requieren aclaración antes de publicar datos/pesos o activar públicamente el servicio. Antes de desplegar quedan también la autenticación, los secretos y la validación Linux/CPU; el contenedor se necesita si se elige Docker. La presentación académica de máximo cinco diapositivas de contenido se prepara en una fase distinta.

## I. Publicación definitiva del código — Prompt 17

El alcance autorizado es exactamente la lista de 23 rutas de la sección H: diez modificadas y trece nuevas, con 93 archivos en el árbol público final. Solo se actualizan las referencias de estado del README y de esta guía respecto al cierre anterior. No se añaden funciones ni se reorganizan rutas.

La revisión no encontró cachés o temporales regenerables que justificaran nuevas eliminaciones. Se conservan todos los experimentos, snapshots, entrenamientos, predicciones, gráficos, configuraciones y evidencias originales. El cotejo confirma 163 recursos protegidos y las 92 entradas del inventario con tamaños y SHA256 idénticos. El escaneo de los candidatos no detectó tokens reconocibles, claves privadas ni rutas personales reales.

No se repite la batería científica de 210/210 pruebas ya superada en el Prompt 16: el código probado no cambia. La batería de la aplicación original vuelve a superar **17/17 pruebas, sin omisiones, en 36,881 s**. También pasan las comprobaciones del lock y dependencias; Streamlit arranca en loopback y responde HTTP 200 en su endpoint de salud, con consultas automáticas desactivadas. Se revisa además el índice exacto antes del commit. Las pruebas y sus logs permanecen en `artifacts/local/verification-prompt17/`, fuera de Git.

No se detecta un nuevo bloqueo para publicar el código, documentación propia y agregados revisados. Los materiales docentes conservan su atribución y la instrucción original de incorporarlos al repositorio académico; no se les atribuye una licencia general. No se asigna una licencia abierta ficticia al código propio. Las restricciones de FRED/Yahoo y derivados continúan bloqueando los dos ZIP, datos y pesos, aunque se propusiera un destino privado.

La publicación no incluye Hugging Face Models ni Streamlit Community Cloud. El profesor puede consultar el código y las métricas agregadas en GitHub y ver la aplicación completa localmente; para ejecutarla desde un clon necesita acceso autorizado a ambos paquetes. Permanecen pendientes la presentación de máximo cinco diapositivas de contenido, la identificación/declaración académica final y la resolución de permisos si se pretende distribuir o alojar artefactos. ORB sigue ausente y la disponibilidad histórica point-in-time no está certificada.

El commit se identifica en el historial de `main`; el cierre exige comprobar que `origin/main` y HEAD coinciden, que el árbol remoto contiene las 93 rutas esperadas y que el estado local queda limpio. Los logs de verificación del push se conservan localmente; no se guardan tokens en informes ni mensajes.
