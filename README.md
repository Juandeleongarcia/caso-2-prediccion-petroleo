# Brent Forecast · Predicción del precio del petróleo Brent

Trabajo individual de **Juan de León García**, Caso 2 de Aprendizaje Automático, **Universidad Alfonso X el Sabio (UAX)**. Compara RNN, LSTM y GRU con persistencia para predecir el Brent a **5, 63 y 252 días hábiles** e integra las redes originales en una aplicación con inferencia real en CPU.

**Conclusión:** persistencia obtiene los menores errores medios de test en los tres horizontes. Las redes no demostraron superioridad. Se mantienen los resultados negativos, las decisiones previas de selección y las limitaciones de generalización, especialmente a 252 días. Es un estudio académico, no una herramienta de predicción financiera fiable.

| Recurso | Acceso y finalidad |
|---|---|
| [Aplicación Brent Forecast](https://brent-forecast.vercel.app/) | Web pública HTML/JavaScript con backend Python/PyTorch e inferencia real; cinco secciones. |
| [Repositorio GitHub](https://github.com/Juandeleongarcia/caso-2-prediccion-petroleo) | Código, configuraciones, pruebas, documentación y métricas agregadas publicables. |
| [Hugging Face Models](https://huggingface.co/Juaandeleoon/caso-2-prediccion-petroleo) | Repositorio **privado** de artefactos; no es una descarga pública. El acceso de la profesora sigue pendiente. |
| [Índice de documentación](docs/README.md) | Recorrido de evaluación, documentación operativa y separación del historial técnico. |
| [PDF de enlaces para la entrega](docs/entrega/Enlaces_Brent_Forecast.pdf) | Segundo documento de entrega; declara expresamente la restricción de acceso a Hugging Face. |

**Estado comprobado el 19/11/2026:** la web devuelve `/api/ready=200`, con fuente `hf` y horizontes 5/63/252. `/api/models` identifica RNN, LSTM, GRU y persistencia, y un snapshot live verificado adquirido el 19/11/2026. La actualización de observaciones está activa; los pesos siguen congelados. Streamlit (`app.py`) permanece como respaldo completo local.

El [enunciado oficial](docs/assignment/ENUNCIADO.pdf), las [instrucciones del profesor](docs/README_profesor.md) y su script de descarga se conservan sin cambios. Un clon público **no incluye datos originales, checkpoints, ZIP privados ni todos los experimentos**: véanse las instrucciones y permisos de reproducción más abajo.

## Datos y fuentes

El snapshot `2026-11-11-official` contiene 7.008 decisiones lunes–viernes desde el 03/01/2000 y 6.501 etiquetas Brent originales observadas. Las 507 fechas restantes no se convierten en precios sintéticos. La última observación Brent del snapshot es el 03/11/2026. El CSV original del profesor permanece intacto y protegido por SHA256.

Las diez variables económicas son Brent, WTI, gas natural, tipo del BCE, USD/EUR, CNY/USD, VIX, índice del dólar, oro y cobre. Se añaden diez indicadores de ausencia durante la transformación: veinte entradas por paso temporal. Se conservan valores económicos negativos reales, incluido el WTI; no se eliminan automáticamente por ser negativos.

FRED aporta las series spot y macroeconómicas; Yahoo Finance complementa cierres de metales e índice del dólar. La cesta ORB de la OPEP figura en el enunciado, pero su descarga oficial devolvió HTTP 403 en la adquisición documentada. Queda ausente y registrada como limitación, sin inventar datos ni sustituirla por otra serie. No se utilizan OHLC ni volumen en la matriz final.

Fuentes: [FRED](https://fred.stlouisfed.org/), [EIA](https://www.eia.gov/opendata/), [Yahoo Finance](https://finance.yahoo.com/) y [OPEP ORB](https://www.opec.org/opec-basket-price.html).

## Disponibilidad temporal y límites de los datos

- **Decisión:** 08:00 en Europe/Madrid, almacenada también en UTC y respetando el horario de verano.
- **Calendario:** lunes–viernes, sin comprimir festivos distintos entre mercados. Los horizontes cuentan 5, 63 y 252 posiciones desde el origen; un objetivo sin Brent original no se evalúa.
- **Entradas:** unión hacia atrás por disponibilidad; nunca relleno hacia atrás, interpolación con el futuro ni covariables futuras desconocidas. Se conservan procedencia, fecha de observación, disponibilidad y antigüedad; las entradas demasiado antiguas quedan ausentes.
- **Retardos asumidos:** diez días lunes–viernes para spot, gas y cambios; dos para BCE, VIX y cierres Yahoo. Son supuestos de trabajo explícitos, no horas oficiales de publicación ni garantías históricas.
- **Etiquetas:** Brent original sin relleno. Cada entrenamiento exige que sus tres etiquetas, incluida la de 252 días, ya estén disponibles al corte.
- **Transformaciones:** medianas, medias y desviaciones se ajustan solo sobre las filas temporales únicas del entrenamiento permitido. Las máscaras de ausencia se añaden después; las fechas de publicación no son entradas numéricas de las redes.

Las fuentes corresponden al vintage descargado, sin reconstrucción completa de publicaciones y revisiones históricas. Las comprobaciones prueban causalidad respecto a la política implementada, pero no que un valor revisado ya existiera en una fecha histórica. **El dataset no se presenta como point-in-time certificado.** Una garantía más fuerte exigiría vintages verificadas o snapshots prospectivos.

Los cambios H.10 contienen observaciones diarias difundidas por bloques; no se considera que la observación de una fecha estuviera publicada ese mismo día. Referencias: [Reserva Federal, H.10](https://www.federalreserve.gov/releases/h10/) y [ALFRED](https://alfred.stlouisfed.org/).

## Metodología congelada

Se construyen ventanas de 60 pasos y tres salidas directas. Los 5.494 ejemplos completos comparten los tres objetivos; si falta una etiqueta original se excluye el origen para todos los horizontes. Las entradas conservan ausencias para imputación posterior, sin imputar etiquetas.

La formulación es residual: `precio futuro = Brent disponible en el origen + cambio predicho`. Persistencia predice exactamente ese último Brent disponible, con cambio cero, sin parámetros entrenables ni tiempo de entrenamiento atribuido. Los errores se calculan después de invertir el escalado y reconstruir precios en USD/barril.

La comparación es justa respecto a esa información común, pero su referencia Brent lleva un retardo asumido de diez días lunes–viernes. El horizonte de cinco días se cuenta desde el origen, no desde la cotización de referencia: no equivale a comparar con un precio spot del mismo día disponible instantáneamente.

| Uso | Años de los orígenes | Finalidad |
|---|---|---|
| Desarrollo | 2015, 2017, 2019 | Selección de variantes, épocas y red |
| Calibración | 2021–2022 | Residuos prequential y cuantiles de incertidumbre |
| Test | 2024; objetivos largos durante 2025 | Evaluación final sin modificar decisiones anteriores |

El walk-forward es expansivo, con reajustes semestrales en enero y julio. El corte se sitúa en la decisión del día lunes–viernes anterior y purga ejemplos por maduración conjunta de las tres etiquetas. Los modelos y persistencia se comparan sobre los mismos orígenes y fechas objetivo.

Las redes tienen una capa de 32 unidades, batch 64, Adam, clipping 1 y tres salidas. Las variantes finales conservan learning rate 0,0003 y weight decay 0,0001. La semilla operativa 42 es una convención previa, no la mejor semilla elegida retrospectivamente; 123 y 2026 se usan para evaluar estabilidad.

| Experimento conservado | Ejecutor | Ajustes | Épocas ejecutadas | Optimización CPU (s) |
|---|---|---:|---:|---:|
| Prueba funcional inicial | `run_smoke.py` | 18 | 54 | 25,572 |
| Comparación A/B | `run_optimization.py` | 72 | 623 | 266,052 |
| Inicio persistente y pérdida robusta C/D | `run_refinement.py` | 60 | 571 | 238,551 |
| Evaluación final | `run_final_evaluation.py` | 54 | 216 | 149,471 |
| **Experimentos principales** | Cuatro directorios originales | **204** | **1.464** | **679,647** |
| Repeticiones acreditadas de reproducibilidad | A-RNN/42: 8; C-RNN/42: 6; LSTM/42: 3 épocas | **3** | **17** | **4,319** |
| **Total acreditado** | No incluye pruebas sintéticas ni ejecuciones sin registro | **207** | **1.481** | **683,966** |

Los cuatro experimentos están en `experiments/` y sus `experiment.json`, historiales y checkpoints originales. Las tres repeticiones se acreditan en sus registros de reproducibilidad. Cada ajuste aprende conjuntamente los tres horizontes: no se cuentan tres modelos por cada ejecución. Los 204 ajustes principales se distribuyen en 72 RNN, 60 LSTM y 72 GRU; con las repeticiones, 74/61/72. Los tiempos son de optimización, no duración total del proyecto; las fechas físicas de ejecución no quedaron registradas. Todas las ejecuciones acreditadas usan CPU, precisión float32 y un hilo de PyTorch.

Se probaron A (learning rate 0,001) y B (0,0003), C (inicio equivalente a persistencia, MSE) y D (ese inicio, SmoothL1). La estabilidad emplea semillas 42, 123 y 2026 en las variantes admitidas por el protocolo. No se realizó una búsqueda extensa de tamaños de red, ventanas, dropout o número de capas.

La validación interior reserva 252 posiciones de origen con etiquetas maduras y purga train por disponibilidad de sus tres resultados. Sus escaladores solo usan el entrenamiento interior. El checkpoint minimiza MSE interior; paciencia 5 y máximo 20 épocas. La época cero es diagnóstico y no era candidata a checkpoint: 18 de los 60 ajustes C/D conservan un MSE interior mayor que al inicio equivalente a persistencia. La selección entre épocas completadas no garantizaba superar el baseline. No se usa el bloque exterior para elegir época.

Los 204 historiales principales reducen el MSE de entrenamiento entre el inicio y el final: existe aprendizaje de optimización. Esto no demuestra generalización; las curvas y el test muestran que mejorar entrenamiento puede coexistir con peores errores futuros. No se ocultan esas ejecuciones ni se sustituyen por semillas favorables.

La formulación ya era residual antes de las variantes C/D. C cambia realmente el inicio: pesos de salida cero y sesgos `−media_train/escala_train`, de modo que el cambio reconstruido inicial es cero. D mantiene ese inicio y optimiza SmoothL1 normalizada con beta 1. La concentración de residuos de entrenamiento a 63 días justificó esa única alternativa; no se ajustaron pesos por horizonte mirando test.

RNN-D y GRU-C mejoraron frente a sus B en desarrollo; LSTM conservó B. El indicador de selección fue MAE/MAE_persistencia, con igual peso por bloque, horizonte y semilla. LSTM-B obtuvo 1,266271, GRU-C 1,296945 y RNN-D 1,388056. Se eligió LSTM-B antes de consultar calibración o test.

En evaluación final se reajusta el historial maduro completo, con escaladores propios, durante el techo de la mediana de las 18 mejores épocas de desarrollo: **RNN-D 4, LSTM-B 3, GRU-C 5**. Se conserva la última época prefijada, sin Early Stopping ni selección con resultados finales. Las medianas son 3,5/3/5 para RNN/LSTM/GRU, calculadas con seis bloques por tres semillas, y se redondean hacia arriba. No son el total de épocas de experimentación. La transferencia de esas épocas a un historial mayor es una heurística: no se contrastó mediante nuevos reajustes completos en desarrollo. No se puede atribuir el fracaso de generalización únicamente al número de épocas ni prometer que entrenar más lo resuelva.

El test es online con política congelada, no un holdout estático de cada precio observado. Ninguna muestra con origen en 2024 entrena los modelos. Un reajuste posterior sí puede utilizar precios ya publicados como etiquetas de orígenes antiguos. Los 472 orígenes de calibración y 238 de test conservan los pares del baseline.

## Resultados finales de test

Cada modelo y semilla evalúa los mismos **238 orígenes por horizonte**. Las cifras neuronales son medias de métricas de tres semillas, no un ensemble ni 714 observaciones independientes. Persistencia tiene un único resultado. MAE y RMSE se expresan en USD/barril; MSE, en USD²/barril²; R² es adimensional. La media de RMSE no tiene por qué coincidir con la raíz del MSE medio.

| Horizonte | Modelo | MAE | MSE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| 5 | Persistencia | 3,445840 | 19,483131 | 4,413970 | 0,364290 |
| 5 | RNN-D | 3,699547 | 21,604730 | 4,646799 | 0,295065 |
| 5 | LSTM-B | 3,660330 | 21,561148 | 4,642512 | 0,296487 |
| 5 | GRU-C | 3,846121 | 23,197997 | 4,814792 | 0,243079 |
| 63 | Persistencia | 5,124454 | 42,414020 | 6,512605 | −0,434691 |
| 63 | RNN-D | 7,012060 | 70,022672 | 8,323865 | −1,368577 |
| 63 | LSTM-B | 5,485524 | 49,380355 | 7,020934 | −0,670333 |
| 63 | GRU-C | 8,894561 | 103,734157 | 10,173887 | −2,508897 |
| 252 | Persistencia | 11,435420 | 162,885949 | 12,762678 | −4,806829 |
| 252 | RNN-D | 15,613731 | 324,332588 | 17,770127 | −10,562347 |
| 252 | LSTM-B | 13,683567 | 248,847090 | 14,732834 | −7,871315 |
| 252 | GRU-C | 19,022181 | 487,286644 | 22,055975 | −16,371604 |

La mejora es `100 × (1 − MAE_modelo/MAE_persistencia)`: un valor negativo significa peor resultado. Al promediar las tres semillas, ninguna arquitectura supera a persistencia en ninguno de los tres horizontes.

La LSTM a 252 días muestra inestabilidad: sus MAE con 42/123/2026 son 21,525/11,363/8,162. La semilla 2026 mejora el baseline un 28,63 %, pero elegirla después de test sería selección retrospectiva. La semilla operativa 42 lo empeora un 88,23 % y sobrestima todos los objetivos largos evaluados. Se conserva la decisión previa y se comunica esta limitación.

RNN/LSTM/GRU tienen 1.827/7.011/5.283 parámetros. Los dos reajustes de test consumieron, en promedio por semilla, 3,45/3,68/10,97 s de optimización, compartidos por sus tres horizontes. Los 54 ajustes completos sumaron 149,47 s de optimización y 187,16 s incluyendo diagnósticos; una repetición de verificación añadió 1,51 s de optimización. Persistencia tiene cero parámetros entrenables.

La aplicación sirve la semilla 42, que debe distinguirse de esas medias:

| Horizonte | LSTM-B/42 MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| 5 | 3,574699 | 21,454060 | 4,631853 | 0,299981 |
| 63 | 5,573487 | 47,873313 | 6,919054 | −0,619356 |
| 252 | 21,525380 | 502,868895 | 22,424738 | −16,927106 |

**Diagnóstico de errores reales.** LSTM-B/42 a 252 días sobreestima en los 238 casos de test. Su mayor error absoluto es 34,556946 USD/barril: origen 19/04/2024, objetivo 08/04/2025, real 64,86 y predicción 99,416946. Los mayores MAE trimestrales, agrupando por fecha objetivo, aparecen en 2025Q4 (24,929985; 54 casos) y 2025Q2 (24,709176; 58). La aplicación permite consultar también los casos de menor error y la persistencia sobre exactamente las mismas observaciones. No se identifican causas económicas de esos errores como si fueran hechos demostrados. Los acontecimientos de 2008 o 2020 son contexto histórico; no se inventan backtests de periodos no evaluados.

### Cálculo y conservación de las métricas

**Definición y trazabilidad de R².** Se calcula `1 − SSE/SST` directamente con `y_true` y `y_pred` originales, por experimento, conjunto, modelo, variante, semilla, semestre y horizonte. El total del conjunto se recalcula sobre sus pares; no promedia R² semestrales. R² es adimensional, puede ser negativo y no equivale a una probabilidad ni al cuadrado de una correlación. Su referencia es la media de los valores reales del conjunto evaluado, conocida retrospectivamente, y no sustituye a persistencia como baseline operativo. Se sigue la [definición de referencia de R²](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html), sin incorporar scikit-learn: con menos de dos observaciones o varianza objetivo nula se muestra como no disponible, sin forzarlo a 0 o 1. Su comparación entre conjuntos debe considerar sus distintas variabilidades.

El suplemento **`artifacts/derived/brent-academic-v1/`** contiene solo `metrics.csv` y `artifact_manifest.json`, con hashes de las fuentes y del resultado derivado. Se distribuye con el código y no reemplaza los paquetes congelados. La aplicación verifica su integridad y anexa únicamente R² a copias en memoria de las métricas originales, tras reconciliar n, MAE, MSE, RMSE y el sesgo cuando estaba registrado. Los paquetes antiguos contienen predicciones de ejemplo de semilla 42; por eso este suplemento utiliza las predicciones completas de los experimentos para calcular correctamente también las otras semillas. No es necesario distribuir esas predicciones completas para ejecutar la aplicación.

Para reconstruirlo en una ubicación nueva, sin entrenar ni sobrescribir una carpeta existente:

```bash
poetry run python -B scripts/prepare_academic_metrics.py --output artifacts/derived/brent-academic-rebuilt
```

## Intervalos nominales del 80 %

Se calculan por modelo, semilla y horizonte mediante residuos de calibración prequential en unidades originales. Cada predicción de calibración procede del reajuste permitido en su origen. El intervalo es `predicción ± q`, donde `q` ocupa la posición `ceil((n+1)×0,8)` de los errores absolutos ordenados, sin interpolación y limitada a `n`.

Los cuantiles usan 471 orígenes por grupo. Se excluye de los tres horizontes el origen 29/12/2022: su etiqueta larga se publica bajo la política exactamente al inicio de test, 01/01/2024 a las 08:00 de Madrid. La congelación exige disponibilidad estrictamente anterior. La evaluación puntual de calibración conserva sus 472 orígenes; no se reescriben resultados anteriores.

`intervals.json` queda congelado antes de los reajustes y predicciones de test. No se agrupan semillas como observaciones adicionales, no se recorta el límite inferior y no se ajusta la amplitud mirando test. Se conservan cobertura, amplitud y puntuación de Winkler por semestre y horizonte.

| Modelo | Cobertura test: 5 / 63 / 252 días | Amplitud media: 5 / 63 / 252 (USD/barril) |
|---|---|---|
| Persistencia | 97,48 / 100 / 100 % | 19,00 / 37,08 / 83,72 |
| RNN-D, media de semillas | 93,84 / 92,72 / 76,05 % | 17,68 / 29,28 / 47,52 |
| LSTM-B, media de semillas | 94,96 / 95,94 / 99,02 % | 18,35 / 30,32 / 63,00 |
| GRU-C, media de semillas | 92,72 / 83,47 / 63,59 % | 17,60 / 28,76 / 53,31 |
| LSTM-B/42, aplicación | 94,54 / 100 / 97,06 % | 18,62 / 34,18 / 63,40 |

Una cobertura alta con bandas amplias no significa calibración perfecta al 80 %. GRU a 252 días presenta infracobertura. La dependencia temporal, los errores solapados y los cambios de régimen impiden prometer cobertura exacta. La cobertura sobre el pool de calibración es descriptiva, no una evaluación independiente.

## Aplicaciones, fechas y vigencia

La web mantiene **Visión general, Predicción, Comparación, Evaluación y Metodología**: histórico, predicciones reales de las tres redes y persistencia, resultados archivados frente a valores reales, errores, cuatro métricas, curvas de entrenamiento/validación e intervalos. Los selectores utilizan los datos auténticos; cuando faltan recursos se indica el bloqueo, sin predicciones simuladas. La API sirve las redes de semilla 42 y conserva la selección previa de LSTM-B/42; esas salidas no deben confundirse con las medias de tres semillas de las tablas.

El checkpoint servido corresponde al **reajuste lógico del backtest del 01/07/2024**, con corte 28/06/2024. Su último origen de entrenamiento es 28/06/2023 y su última etiqueta admitida estaba disponible el 28/06/2024. La fecha física de ejecución del entrenamiento no quedó registrada. No se deduce de la fecha lógica ni de la modificación del archivo.

La aplicación distingue adquisición de datos, fecha de observación, disponibilidad, origen de predicción, referencia Brent y corte/reajuste lógico del modelo. El precio de referencia admisible puede ser anterior a la última cotización observada. Actualizar la ventana no actualiza los pesos: ha vencido su cadencia semestral y no se ha realizado un reajuste de producción posterior a test.

Los acontecimientos históricos se documentan en [configs/economic_events.json](configs/economic_events.json). Son contexto retrospectivo, no nuevas variables ni backtests inventados. Las explicaciones de predicción son reglas deterministas; los intervalos no son probabilidades de acierto. La calculadora de costes muestra escenarios spot en USD sin primas, transporte ni impuestos.

### Actualización diaria de observaciones

El [workflow Brent daily observations](.github/workflows/update-live-data.yml) dispone de cron diario **06:37 UTC** y ejecución manual. Adquiere las diez series, valida esquema, fechas, ausencias y duplicados, y comprueba inferencia con las tres redes congeladas antes de publicar atómicamente un snapshot en `live/` del Models privado. Conserva la última copia válida si falla una fuente o validación. No sobrescribe el CSV del profesor, los ZIP ni los resultados científicos.

El 19/11/2026, `BRENT_DAILY_ENABLED=1` está habilitado y la [ejecución manual #7](https://github.com/Juandeleongarcia/caso-2-prediccion-petroleo/actions/runs/44558822525) terminó correctamente. La web consume su snapshot verificado. Las ejecuciones observadas hasta esta revisión son manuales: todavía no se acredita una ejecución disparada por el cron. La [validación previa sin publicación](https://github.com/Juandeleongarcia/caso-2-prediccion-petroleo/actions/runs/44556450853) conserva la prueba `status=validated` y `weights_updated=false`.

Vercel mantiene las revisiones científicas fijas y consulta el canal live cada cinco minutos **por instancia y bajo demanda**. Resuelve un commit completo de datos, comprueba SHA256, política temporal y vinculación al manifiesto de modelos; así cambia observaciones sin redeploy diario. Una caída conserva la copia validada disponible; un reinicio sin caché puede volver al bundle congelado. La web informa de la fecha realmente utilizada.

Las observaciones nuevas respetan el máximo entre el retardo asumido y su primera adquisición efectiva; no se antedatan datos. El histórico retrospectivo original no se convierte por ello en point-in-time. **Actualización de datos y reentrenamiento son procesos distintos.** El workflow operativo no reentrena. Existe un [procedimiento propuesto de reentrenamiento manual](docs/ACTUALIZACION_DIARIA.md), pero su ejecución, validación y publicación controlada de nuevos modelos siguen pendientes.

La guía [Actualización diaria](docs/ACTUALIZACION_DIARIA.md) explica secretos de servidor, autorizaciones, validaciones, diagnóstico y comprobación del snapshot en la API. No se guardan tokens en código, navegador, documentación o logs.

## Estructura del repositorio

| Ruta | Contenido y conservación |
|---|---|
| `src/oil_forecast/` | Adquisición, protocolo temporal, arquitecturas, entrenamiento, evaluación, incertidumbre e inferencia. |
| `scripts/` | Ejecutores del estudio, actualización live y gestión verificada de artefactos. |
| `configs/` | Fuentes, experimentos, política temporal y registros de hashes; no contienen credenciales. |
| `tests/`, `web/tests/` | Pruebas Python e integración/frontend JavaScript. |
| `.github/workflows/` | Automatización operativa de observaciones, sin entrenamiento automático. |
| `web/` | Frontend y backend de la aplicación completa; [guía local](web/README.md). |
| `deployment/vercel/`, `deployment/vercel-app/` | Plantilla y snapshot autónomo de producción; la duplicación es deliberada y se comprueba con manifiestos. |
| `app.py`, `.streamlit/` | Aplicación Streamlit original de respaldo. |
| `artifacts/derived/brent-academic-v1/` | Métricas agregadas adicionales y manifiesto de R²; no sustituyen resultados originales. |
| `docs/` | Materiales docentes, índice, guías técnicas, inventario de artefactos y PDF de enlaces. |
| `deployment/huggingface/README.md` | Ficha académica preparada; no equivale a una licencia de redistribución. |
| `data/`, `experiments/` | Datos y evidencias científicas originales **locales, excluidos de Git**. |
| `artifacts/releases/`, `artifacts/local/` | ZIP verificados, instalación y evidencias locales **excluidos de Git**. |
| `presentacion/`, `docs/auditoria/` | Presentaciones definitivas/borradores e informes internos conservados localmente. |

La alternativa histórica `deployment/public_app/` está identificada como **demostración de resultados sin inferencia neuronal**. Sus scripts y pruebas se conservan por trazabilidad; no es la web principal ni sustituye a Streamlit. No se reorganizan rutas de ejecución ni se eliminan evidencias científicas.

## Reproducción e instalación local

Entorno contrastado: **Python 3.13.7, Poetry 2.5.1, Windows x86-64 y CPU**. `pyproject.toml` y `poetry.lock` conservan las dependencias científicas. La instalación Linux de Actions y el despliegue Vercel utilizan su configuración independiente de producción; no debe sustituirse por un lock generado arbitrariamente.

```bash
poetry install
poetry check --lock
poetry run python -m pip check
```

### Artefactos necesarios

Para ejecutar la aplicación completa se necesitan **`brent-frozen-v1.zip`** (pesos RNN-D/42, LSTM-B/42 y GRU-C/42, transformadores, cuantiles, protocolo y recursos de inferencia) y **`brent-dashboard-v1.zip`** (evidencia y gráficos). Sus identidades están en [serving_bundle.json](configs/serving_bundle.json) y [dashboard_bundle.json](configs/dashboard_bundle.json). El [inventario de 92 artefactos](docs/deployment/artifact_inventory.csv) conserva tamaños, SHA256 y función.

Se obtienen del propietario o de una copia privada con acceso y traslado autorizados; la web usa credenciales exclusivamente de servidor. **El acceso de la profesora al repositorio privado de Hugging Face aún no está concedido.** No convertir el repositorio en público ni compartir tokens como solución. Una nueva descarga de proveedores puede contener revisiones y no reproduce por sí sola el snapshot científico cerrado.

Con los ZIP originales en `artifacts/releases/`:

```bash
poetry run python scripts/manage_artifacts.py install --archive artifacts/releases/brent-frozen-v1.zip --registry configs/serving_bundle.json --destination artifacts/local/brent-frozen-v1
poetry run python scripts/manage_artifacts.py install --archive artifacts/releases/brent-dashboard-v1.zip --registry configs/dashboard_bundle.json --destination artifacts/local/brent-dashboard-v1
poetry run python scripts/manage_artifacts.py verify --directory artifacts/local/brent-frozen-v1
poetry run python scripts/manage_artifacts.py verify --directory artifacts/local/brent-dashboard-v1
```

La instalación verifica archivo, rutas y miembros, y rechaza sobrescribir destinos existentes. Si ya están instalados, ejecutar solo `verify`. Con ambos paquetes y el suplemento agregado versionado, la aplicación funciona sin el árbol completo `data/`/`experiments/` y sin descargar proveedores. Reproducir íntegramente el estudio o su batería científica requiere además esos datos y experimentos originales.

### Web completa en local

Desde la raíz, con ambos paquetes instalados (PowerShell):

```powershell
$env:BRENT_MODEL_SOURCE = "local"
$env:BRENT_ALLOW_PROVIDER_ACCESS = "0"
poetry run python -B web/server.py
```

Abrir `http://127.0.0.1:8765`. Es un servidor de desarrollo local. La configuración de producción se conserva en `deployment/vercel-app`: Root Directory de Vercel `deployment/vercel-app`, preset **Other**, build `node build.mjs`, salida `public`. Véanse [importación de Vercel](docs/VERCEL_IMPORTACION.md) y [plan de despliegue](docs/PLAN_DESPLIEGUE.md). No cambiar pins, hashes o interruptores de autorización para solventar fallos de acceso.

### Streamlit como respaldo completo

```powershell
$env:BRENT_AUTO_REFRESH = "0"
$env:BRENT_AUTO_PREPARE = "0"
poetry run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Abrir `http://127.0.0.1:8501`. Estas variables evitan adquisiciones o preparación automática durante la revisión local. La actualización controlada de Streamlit conserva sus funciones y requiere autorización para consultar proveedores; no entrena redes.

### Verificación sin reentrenar el estudio

Con los recursos originales disponibles:

```bash
poetry run python -X utf8 -B -W error -m unittest discover -s tests -v
node --test web/tests/frontend.test.mjs web/tests/build.test.mjs
```

Las pruebas sintéticas de entrenamiento trabajan en temporales; no alteran los modelos del proyecto. Un clon sin artefactos privados no acredita la batería completa. El cierre del 19/11/2026 ha superado **253/253 pruebas Python y 7/7 JavaScript**, sin omisiones. La comparación SHA256 protege datos, experimentos, pesos, cuantiles, manifiestos y presentación definitiva.

Los ejecutores históricos `prepare_data.py`, `run_baseline.py`, `run_smoke.py`, `run_optimization.py`, `run_refinement.py` y `run_final_evaluation.py` documentan el procedimiento, pero **no forman parte del arranque**. No ejecutarlos sobre resultados existentes: cualquier reproducción requiere permisos, destinos nuevos y el protocolo original. El script del profesor puede sobrescribir su CSV si se ejecuta; la aplicación no lo invoca.

## Cumplimiento académico y entrega

| Aspecto | Estado y evidencia |
|---|---|
| Trabajo individual, código y README en GitHub | Autor identificado; código, protocolos y pruebas versionados. Se declara asistencia de IA. |
| RNN/LSTM/GRU y comparación con persistencia | Experimentos y resultados reales; selección solo en desarrollo; mismas observaciones de evaluación. |
| Ventanas, escalado, walk-forward y cuatro métricas | Implementados y probados; particiones, supuestos y R² explicados arriba. |
| Horizontes 5/63/252 e incertidumbre | Inferencia real con pesos originales y cuantiles congelados; límites de cobertura declarados. |
| Aplicación funcional, explicación e historia económica | Web pública con backend Python; cinco secciones y Streamlit local de respaldo. |
| Variables externas exigidas | **Parcial:** diez series utilizadas; ORB de OPEP no integrada. OHLC/volumen complementarios no son entradas de estas redes. |
| Actualización dinámica | Observaciones automatizadas y canal live operativo; siguiente ejecución programada aún por acreditar. |
| Reentrenamiento periódico mencionado en el enunciado | **Pendiente:** procedimiento futuro documentado; pesos congelados y sin workflow operativo de reentrenamiento de producción. |
| Máximo cinco diapositivas de contenido más portada | Presentación definitiva UAX disponible: seis diapositivas, portada más cinco de contenido. |
| Reproducibilidad desde un clon público | **Parcial:** código y configuración disponibles; artefactos privados y evidencias originales requieren acceso autorizado. |

La entrega solicitada contiene **dos PDF**:

1. `presentacion/entrega_final_2026-11-19/Brent_Forecast_Defensa_Final_UAX.pdf` (local, seis páginas). Su PowerPoint y notas se conservan intactos junto al PDF. Esta carpeta permanece excluida de Git; no se publica contenido científico individual embebido ni material de plantilla sin revisar sus permisos.
2. [Documento de enlaces](docs/entrega/Enlaces_Brent_Forecast.pdf), con GitHub, aplicación y Models privado; indica el acceso pendiente de la profesora.

No hay ZIP de entrega ni subida a Moodle. El estudiante debe entregar esos dos PDF por el canal académico y resolver el acceso autorizado a los artefactos privados si se requiere reproducción independiente. ORB y reentrenamiento periódico siguen siendo limitaciones explícitas; no se sustituyen por una declaración de cumplimiento total.

## Permisos, privacidad y autoría

GitHub publica código, configuración, documentación y estadísticas agregadas. `.gitignore` excluye datos, experimentos, checkpoints, ZIP, cachés, credenciales y materiales locales. Los modelos y paquetes ya alojados en Models siguen privados; un repositorio privado no concede por sí mismo derechos de uso o redistribución.

El propietario confirmó autorización para adquirir periódicamente las diez series, almacenarlas mediante Actions/Hugging Face y usarlas en la web académica. Esa confirmación de alcance no equivale a una licencia general de los proveedores, ni autoriza publicar datos o derivados adicionales. Se mantienen controles de autorización, secretos de servidor, revisiones fijas y SHA256. Las condiciones y limitaciones históricas se conservan en la [guía de publicación](docs/PUBLICACION.md).

Se ha utilizado asistencia de IA para programación, revisión y documentación. Juan de León García asume la revisión, defensa y responsabilidad académica conforme a las normas aplicables. No se atribuye a la herramienta la autoría del trabajo ni se certifica una originalidad que no pueda comprobarse.
