# Inventario y preparación de publicación

Esta guía prepara los destinos del proyecto. No se han ejecutado commit, push, cargas a Hugging Face ni despliegues. El código se prepara para GitHub; los paquetes siguen siendo locales y la difusión de sus datos requiere revisar derechos de redistribución.

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

La preparación local del Prompt 11 no autoriza publicar. Antes del primer commit o push se debe presentar el inventario y obtener autorización expresa del estudiante. Los siguientes comandos de revisión y preparación son reproducibles; los de commit/push permanecen sin ejecutar:

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

Se prepararon **75 rutas en el índice —71 nuevas y 4 modificadas—**, sin commit ni push. Se exportaron los 80 archivos del índice a la instalación temporal: el lock, los dos paquetes y el manifiesto académico también verifican con los bytes que publicará Git. La normalización de finales de línea afecta a cuatro archivos de texto y no cambia los hashes del suplemento académico. No quedan cambios públicos fuera del índice después de actualizar esta guía. El remoto y `HEAD` continúan en el commit inicial revisado.

### Recuperación de recursos conservados fuera de Git

| Recurso | Ubicación local preservada | Recuperación en otro equipo |
|---|---|---|
| Paquete de inferencia | `artifacts/releases/brent-frozen-v1.zip` | Copiar desde un respaldo autorizado; instalar con `manage_artifacts.py install --registry configs/serving_bundle.json` y verificar el destino. |
| Evidencia de la interfaz | `artifacts/releases/brent-dashboard-v1.zip` | Mismo procedimiento con `configs/dashboard_bundle.json`. Los comandos completos están en el README. |
| Investigación original | `data/petroleo_dataset.csv`, `data/raw/2026-11-11-official/`, `data/processed/2026-11-11-official/`, `data/supervised/2026-11-11-official/` | Trasladar las carpetas originales autorizadas conservando estructura y manifiestos; el CSV se coteja con `original_sha256` de `configs/data.json`. No sustituirlos por una descarga revisada. |
| Experimentos y entorno histórico | Cuatro subcarpetas de `experiments/` y `artifacts/local/research-environment/` | Copiar desde un respaldo autorizado; conservar pesos, métricas y manifiestos completos. La batería científica verifica las referencias históricas. |
| Datos recientes y evidencias internas | `data/live/`, `docs/auditoria/`, `artifacts/local/app-screenshots/` | Conservar como respaldo local. No se requieren para inferencia con los paquetes. |

### Inventario exacto del commit propuesto

**75 rutas con cambios: 4 modificadas y 71 nuevas.** Las otras cinco rutas ya versionadas no cambian: `.gitattributes`, `scripts/__init__.py` y los tres materiales originales del profesor. El conjunto público completo conserva 80 archivos. `A` indica nuevo; `M`, modificado respecto a `HEAD`.

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

Después de autorizar contenidos, crear y comprobar el repositorio Models y su visibilidad. Con el CLI oficial de Hugging Face disponible en el entorno de publicación, los comandos de carga serán:

```powershell
$hfModels = 'USUARIO_HF/REPOSITORIO_MODELS'
hf auth login
hf upload $hfModels artifacts/releases/brent-frozen-v1.zip brent-frozen-v1.zip --repo-type model
hf upload $hfModels artifacts/releases/brent-dashboard-v1.zip brent-dashboard-v1.zip --repo-type model
```

Sustituir el identificador por uno real ya revisado: `hf upload` puede crear un repositorio inexistente. La ficha y la revisión exacta de Models se fijarán en esa fase. No se ha instalado el CLI ni publicado archivos durante esta preparación. Referencia: [CLI oficial de Hugging Face](https://huggingface.co/docs/huggingface_hub/guides/cli).

## C. Hugging Face Spaces

El Space necesita `app.py`, `src/`, `configs/`, `.streamlit/config.toml`, los archivos de dependencias, `scripts/manage_artifacts.py` y el suplemento académico. No necesita `data/`, `experiments/`, el entorno de Windows ni las auditorías internas. Las pruebas incluyen una copia aislada sin las carpetas de investigación.

El SDK Streamlit integrado de Spaces está deprecado; la vía compatible es **Docker SDK con Streamlit**. Sin embargo, la documentación consultada el **17/11/2026** exige un plan PRO personal o Team/Enterprise para crear Spaces Docker. CPU Basic no tiene tarifa horaria, pero esto **no convierte la creación de un Space Docker en una opción totalmente gratuita**. No se contrata ningún plan ni se escoge este destino como solución gratuita definitiva. Si una fase posterior autoriza y permite este destino, preparar y probar un `Dockerfile` y el README del Space con:

```yaml
---
title: Brent Forecast
sdk: docker
app_port: 7860
---
```

El contenedor deberá usar Python 3.13 y el lock de Poetry, instalar la base de zonas horarias para `Europe/Madrid`, ejecutarse con permisos de escritura de un usuario sin privilegios y escuchar en `0.0.0.0:7860`. La configuración local conserva `127.0.0.1`; esa dirección se sustituirá mediante los argumentos de Streamlit dentro del contenedor. El build y la ejecución Linux quedan pendientes y no se presentan como comprobados. Fuentes: [Streamlit en Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit) y [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).

### Carga de artefactos en el Space

Después de publicar los ZIP autorizados, fijar la revisión del repositorio Models que contiene ambos archivos. Configurar en los ajustes del Space:

```text
BRENT_INFERENCE_URL=https://huggingface.co/USUARIO_HF/REPOSITORIO_MODELS/resolve/REVISION_MODELS/brent-frozen-v1.zip
BRENT_EVIDENCE_URL=https://huggingface.co/USUARIO_HF/REPOSITORIO_MODELS/resolve/REVISION_MODELS/brent-dashboard-v1.zip
```

Los valores anteriores son patrones para sustituir, no URL provisionadas. La aplicación permite instalar con «Preparar o verificar recursos» y comprueba los SHA256 de los registros versionados. Las URL públicas HTTPS ancladas a revisión son compatibles con el cargador actual.

Un repositorio Models privado **no queda autenticado por definir `HF_TOKEN`**: el descargador actual no añade cabeceras de autorización. El acceso privado requiere aprovisionamiento autenticado fuera de la aplicación o una ampliación específica posterior. No incrustar tokens en URLs ni en Git. La elección de privacidad y el método autorizado de acceso quedan pendientes.

En otro equipo, los dos ZIP autorizados también pueden instalarse localmente, siempre en destinos todavía inexistentes:

```powershell
poetry run python scripts/manage_artifacts.py install --archive artifacts/releases/brent-frozen-v1.zip --registry configs/serving_bundle.json --destination artifacts/local/brent-frozen-v1
poetry run python scripts/manage_artifacts.py install --archive artifacts/releases/brent-dashboard-v1.zip --registry configs/dashboard_bundle.json --destination artifacts/local/brent-dashboard-v1
poetry run python scripts/manage_artifacts.py verify --directory artifacts/local/brent-frozen-v1
poetry run python scripts/manage_artifacts.py verify --directory artifacts/local/brent-dashboard-v1
poetry run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Si ya están instalados, verificar en vez de reinstalar. No ejecutar `build` para reemplazar los ZIP congelados. El estado de datos recientes es local y puede perderse al reiniciarse un alojamiento efímero; la aplicación conserva el respaldo portable y comprueba la antigüedad al arrancar. Las actualizaciones actúan mientras se ejecuta o con una sesión activa, sin reentrenar pesos.

## D. Condiciones actuales y alternativa gratuita

Consulta documental del **17/11/2026**, sin crear cuentas, contratar planes ni activar hardware:

| Destino | Condiciones comprobadas | Decisión para este proyecto |
|---|---|---|
| HF Models | Almacenamiento público gratuito sujeto a disponibilidad; 100 GB privados en el plan gratuito. | Destino posible de los dos ZIP, sujeto a permisos y autenticación. |
| HF Spaces Docker | CPU Basic: 2 vCPU, 16 GB RAM, 50 GB efímeros, sin tarifa horaria; crear Docker requiere plan de pago. Suspensión tras 48 h sin uso en CPU Basic. | No elegirlo como solución enteramente gratuita ni contratar nada. |
| Streamlit Community Cloud | Servicio gratuito conectado a GitHub; suspensión tras 12 h sin tráfico. La documentación publica rangos aproximados fechados en febrero de 2024: CPU 0,078–2 núcleos, RAM 690 MB–2,7 GB y hasta 50 GB de disco, sujetos a cambios. | Alternativa gratuita por verificar en Linux/CPU; no se garantiza capacidad ni disponibilidad continua. |

Fuentes: [almacenamiento HF](https://huggingface.co/docs/hub/storage-limits), [condiciones de Spaces](https://huggingface.co/docs/hub/spaces-overview), [suspensión de Spaces](https://huggingface.co/docs/hub/spaces-gpus), [Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud) y [recursos y suspensión](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

Community Cloud admite `pyproject.toml` mediante Poetry, según su [documentación de dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). La fase futura debe comprobar su versión efectiva de Python/Poetry, uso del lock y wheel CPU, la base de zonas horarias y memoria en Linux. No añadir ahora un `requirements.txt` paralelo ni prometer una instalación remota ya probada. El pico de 507,3 MiB observado previamente en Windows no acredita consumo Linux ni margen suficiente para concurrencia.

El despliegue futuro gratuito deberá escoger GitHub/`app.py`, aprovisionar ambos ZIP autorizados por HTTPS y SHA256, conservar el suplemento académico y comprobar actualización/fallback sin reentrenar. En cualquier servicio suspendido no se ejecutan actualizaciones de fondo. La última copia válida es local al almacenamiento disponible y el paquete congelado permite recuperar el servicio si ese almacenamiento se pierde.

## Reproducción y límites de la preparación

NumPy 2.5.4 se declara directamente porque el código lo importa; ya estaba instalado y fijado en el lock. La actualización del lock no cambia ninguna de sus 52 versiones. Las dependencias de figuras científicas siguen siendo opcionales y separadas. Los dos archivos históricos del entorno de investigación se conservan intactos.

Las claves de rutas Windows de los manifiestos históricos se interpretan de forma portable, sin reescribir los archivos originales ni cambiar folds o resultados. Esto se prueba con rutas POSIX y con los 52 hashes de entrada; no equivale a haber ejecutado el estudio en Linux.

Un clon de GitHub contiene estructura, código y métricas agregadas. La aplicación completa requiere los dos paquetes externos autorizados. Reproducir exactamente la investigación requiere además los snapshots originales y el entorno archivado; una descarga actual puede contener revisiones. Algunas pruebas se omiten si faltan esos recursos, por lo que un resultado parcial sin ellos no certifica toda la investigación.

Antes del commit/push del código se solicita autorización expresa sobre el inventario revisado. La identificación académica y la declaración final del estudiante siguen pendientes; puede conservarse el código sin una licencia de reutilización hasta que su titular decida. Los permisos de las fuentes requieren aclaración antes de publicar datos/pesos o activar públicamente el servicio. Antes de desplegar quedan también la autenticación, los secretos y la validación Linux/CPU; el contenedor se necesita si se elige Docker. La presentación académica de máximo cinco diapositivas de contenido se prepara en una fase distinta.
