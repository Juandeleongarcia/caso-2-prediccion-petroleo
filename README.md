# Predicción del precio del petróleo Brent

Proyecto individual del Caso 2 de Aprendizaje Automático de la UAX: comparación de RNN, LSTM y GRU con persistencia a **5, 63 y 252 días hábiles**. Se conservan intactos el [enunciado oficial](docs/assignment/ENUNCIADO.pdf) y los [materiales del profesor](docs/README_profesor.md).

**Conclusión principal:** ninguna arquitectura neuronal supera a persistencia en el promedio de las tres semillas de test, en ninguno de los tres horizontes. LSTM-B/42 fue elegida exclusivamente en desarrollo; se mantiene como demostración de aprendizaje profundo y sus malos resultados se muestran expresamente. Los pesos, transformadores, predicciones y resultados originales permanecen congelados.

**Estado:** estudio cerrado y aplicación Streamlit funcional en Windows/CPU. La publicación de los cambios está pendiente de autorización; no hay modelos ni aplicación publicados en Hugging Face. El despliegue Linux y la presentación académica siguen pendientes.

Para revisar el trabajo: [datos y fuentes](#datos-y-fuentes), [metodología](#metodología-congelada), [resultados y cuatro métricas](#resultados-finales-de-test), [aplicación](#aplicación-brent-forecast--energy-intelligence) y [cumplimiento académico](#cumplimiento-del-enunciado-y-aclaración-docente). La [guía de publicación](docs/PUBLICACION.md) contiene inventario, exclusiones, permisos y pasos posteriores.

| Destino | Enlace y estado |
|---|---|
| Código | [Repositorio GitHub existente](https://github.com/Juandeleongarcia/caso-2-prediccion-petroleo), rama `main`; los cambios de esta entrega aún no se han subido. |
| Modelos y evidencia portable | Hugging Face Models: pendiente de permisos y publicación; no se inventa una URL. |
| Aplicación alojada | Pendiente. Community Cloud es una alternativa gratuita por validar; crear un Space Docker exige actualmente un plan de pago. |

## Instalación e inferencia sin entrenar

Entorno comprobado: Windows, Python 3.13.7, Poetry 2.5.1 y CPU. Desde la raíz del repositorio:

```bash
poetry install
poetry check --lock
poetry run python -m pip check
```

La inferencia necesita el paquete congelado `brent-frozen-v1.zip`, además del código. Su identidad y SHA256 se registran en [configs/serving_bundle.json](configs/serving_bundle.json). El ZIP se conserva en `artifacts/releases/`, fuera de Git. Debe trasladarse desde una copia autorizada; no se ha publicado automáticamente en ningún servicio externo.

```bash
poetry run python scripts/manage_artifacts.py install --archive artifacts/releases/brent-frozen-v1.zip --registry configs/serving_bundle.json --destination artifacts/local/brent-frozen-v1
poetry run python scripts/manage_artifacts.py verify --directory artifacts/local/brent-frozen-v1
```

La instalación comprueba el SHA256 esperado, las rutas y los hashes de los miembros del archivo antes de cargar los pesos. Rechaza contenido inesperado y destinos existentes. Si el paquete ya está instalado, basta con verificarlo. Un clon sin el ZIP todavía no dispone de los pesos ni de los datos locales necesarios para inferencia.

```python
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, "src")
from oil_forecast.inference import load_forecaster

bundle = Path("artifacts/local/brent-frozen-v1")
service = load_forecaster(bundle)
window = pd.read_csv(bundle / "latest_window.csv")
forecast = service.predict(window)
history = pd.read_csv(bundle / "brent_history.csv")
metrics = service.evaluation_metrics("test")
```

No se necesitan los snapshots ni los experimentos originales para este ejemplo. La API devuelve precio, cambio, fechas objetivo e intervalos de la red y de persistencia. Valida las 60 filas, el calendario, el orden de variables y la disponibilidad de cada entrada; no necesita etiquetas futuras ni reajusta escaladores o cuantiles.

## Aplicación: Brent Forecast | Energy Intelligence

Desde PowerShell, en la raíz y con las dependencias instaladas:

```powershell
poetry run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Abre `http://127.0.0.1:8501`. La aplicación tiene cinco vistas: mercado e indicadores disponibles; predicciones reales de persistencia/RNN-D/LSTM-B/GRU-C; comparación por horizonte, conjunto y semillas; curvas y errores de los experimentos guardados; metodología y límites. Muestra salidas directas a 5/63/252 días lunes–viernes y bandas nominales del 80 %, sin interpolar una trayectoria futura. Los orígenes antiguos se consultan mediante predicciones archivadas, nunca ejecutando sobre ellos un checkpoint posterior.

El segundo paquete, `brent-dashboard-v1.zip`, contiene una selección compacta de las evidencias experimentales, sin pesos adicionales: todas las métricas y curvas guardadas, predicciones de la semilla canónica 42 y persistencia de los cuatro experimentos. El criterio de ejemplo es fijo (2019H1 en desarrollo y primer semestre del conjunto final), sin selección por resultados favorables. La ausencia de validación en los reajustes finales de épocas prefijadas permanece explícita. Su identidad se registra en [configs/dashboard_bundle.json](configs/dashboard_bundle.json).

Para instalar esa evidencia en otro equipo:

```powershell
poetry run python scripts/manage_artifacts.py install --archive artifacts/releases/brent-dashboard-v1.zip --registry configs/dashboard_bundle.json --destination artifacts/local/brent-dashboard-v1
```

Los paquetes ya instalados no se sobrescriben. Con ambos paquetes, la aplicación funciona sin `data/` ni `experiments/` y sin conexión. Sin evidencia, mantiene inferencia y comparación final e indica qué gráficos faltan. Sin pesos, explica cómo preparar los recursos y permite consultar la metodología. El botón «Preparar o verificar recursos» verifica los directorios existentes o instala los ZIP locales confiables.

La aplicación comprueba automáticamente los datos al iniciarse y cada cinco minutos con una sesión activa; descarga una copia nueva cuando han transcurrido 24 horas desde la última adquisición válida. «Actualizar datos ahora» permite una consulta manual. Cambiar de página no dispara descargas si no corresponde. Los tres modelos CPU se cargan en caché y sus pesos, escaladores y cuantiles permanecen intactos.

La política está en [configs/app.json](configs/app.json): frecuencia inicial de 24 h, reintento de fallos a los 15 min y comprobación de sesión cada 300 s. `BRENT_REFRESH_HOURS` permite cambiar la frecuencia; `BRENT_AUTO_REFRESH=0` desactiva consultas automáticas y `BRENT_LIVE_DIR` elige el directorio escribible. El botón manual conserva un antirrebote de 60 s. En un alojamiento suspendido no se ejecutan consultas de fondo: se comprueba de nuevo al arrancar.

Los snapshots en `data/live/` son inmutables. Un bloqueo nativo no bloqueante coordina sesiones y procesos; un puntero JSON se publica atómicamente solo tras verificar SHA256, política temporal, diez entradas habilitadas y compatibilidad de inferencia. Ante un fallo de cualquier entrada —incluidos los Yahoo opcionales del descargador— se conserva la última copia completa. ORB queda excluida de esa exigencia por no ser entrada de las redes. En reinicios se recupera la última copia válida; si no existe, se utiliza el paquete congelado. La pérdida del almacenamiento temporal de Spaces implica volver a ese paquete y comprobar las fuentes, sin prometer persistencia no contratada. Los fallos también respetan el intervalo de reintento.

El histórico permite activar acontecimientos y consultar su contexto: máximo y crisis de 2008, exceso de oferta 2014–2016, pandemia de 2020, Ucrania en 2022 y movimientos de 2026. [configs/economic_events.json](configs/economic_events.json) separa hechos documentados, interpretación económica y referencias. Los marcadores se calculan sobre observaciones spot reales del episodio completo antes de filtrar el gráfico, sin inventar precios ni equiparar la fecha del extremo con una causa. Los episodios son contexto retrospectivo, nunca nuevas entradas de las redes. Las explicaciones de 2026 usan el [STEO de EIA del 10/11/2026](https://www.eia.gov/outlooks/steo/report/global_oil.php), con revisión documental fechada; no convierten sus previsiones de 2027 en hechos ni en resultados de nuestros modelos.

La predicción incorpora una explicación determinista según modelo, horizonte y resultado, sin LLM ni API de texto. Contrasta referencia y última cotización, cambio, bandas y MAE de la semilla 42 frente a persistencia; no convierte errores en probabilidades de acierto. Los resultados archivados se describen retrospectivamente, sin introducir cotizaciones actuales como conocidas entonces. La calculadora opcional multiplica un volumen por referencia, persistencia, estimación y límites originales, con redondeo final a céntimos. Son escenarios del componente spot en USD, sin primas, transporte, impuestos ni conversión de divisa; no son recomendaciones de inversión ni costes garantizados.

La fecha de adquisición, la observación económica, el origen y el reajuste lógico se muestran por separado. El checkpoint es histórico y ha vencido su cadencia semestral: consultar datos recientes no lo convierte en un modelo actualizado. Los retardos asumidos no certifican disponibilidad histórica point-in-time; OPEP figura como no disponible en el snapshot congelado.

## Estructura y artefactos

| Ruta | Función | Git |
|---|---|---|
| `app.py`, `.streamlit/config.toml` | Entrada y tema de la aplicación local | Sí |
| `src/oil_forecast/` | Adquisición, datos, redes, entrenamiento, evaluación, incertidumbre e inferencia | Sí |
| `scripts/` | Ejecutores reproducibles y gestión del paquete congelado | Sí |
| `configs/` | Fuentes, protocolo, experimentos y registro del paquete | Sí |
| `tests/` | Contratos temporales, métricas, modelos y artefactos | Sí |
| `docs/assignment/`, `docs/README_profesor.md` | Materiales originales de la asignatura | Sí |
| `docs/PUBLICACION.md` | Inventario de entrega y comandos de continuación | Sí |
| `artifacts/derived/brent-academic-v1/` | Métricas académicas adicionales y manifiesto de R² | Sí |
| `data/` | CSV original, snapshots, datos derivados y cachés locales | No |
| `experiments/` | Historial válido, resultados completos, pesos y gráficos | No |
| `artifacts/releases/`, `artifacts/local/` | Archivo distribuible y paquete instalado | No |
| `docs/auditoria/` | Evidencias y documentación interna de trabajo | No |

El paquete de inferencia incluye los pesos y transformadores de RNN-D, LSTM-B y GRU-C con semilla 42, protocolo, cuantiles, métricas de las tres semillas, histórico Brent, última ventana válida y predicciones de referencia. Sus manifiestos conservan hashes y procedencia. La API sirve LSTM-B/42; la presencia de otros candidatos no cambia la selección.

Para reconstruir el paquete desde los resultados locales originales, sin entrenar ni recalibrar:

```bash
poetry run python scripts/manage_artifacts.py build
```

El constructor utiliza por defecto la evaluación final y el snapshot preparado; crea destinos nuevos y conserva íntegros los archivos de origen. Un ZIP verificado puede trasladarse a otro ordenador. Su publicación como Release requiere una acción posterior autorizada; no se ha subido a servicios externos en este cierre.

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

| Uso | Años de los orígenes | Finalidad |
|---|---|---|
| Desarrollo | 2015, 2017, 2019 | Selección de variantes, épocas y red |
| Calibración | 2021–2022 | Residuos prequential y cuantiles de incertidumbre |
| Test | 2024; objetivos largos durante 2025 | Evaluación final sin modificar decisiones anteriores |

El walk-forward es expansivo, con reajustes semestrales en enero y julio. El corte se sitúa en la decisión del día lunes–viernes anterior y purga ejemplos por maduración conjunta de las tres etiquetas. Los modelos y persistencia se comparan sobre los mismos orígenes y fechas objetivo.

Las redes tienen una capa de 32 unidades, batch 64, Adam, clipping 1 y tres salidas. Las variantes finales conservan learning rate 0,0003 y weight decay 0,0001. La semilla operativa 42 es una convención previa, no la mejor semilla elegida retrospectivamente; 123 y 2026 se usan para evaluar estabilidad.

| Experimento conservado | Ejecutor | Presupuesto y propósito |
|---|---|---|
| Prueba CPU inicial | `run_smoke.py` | 18 ajustes de tres épocas; referencia funcional |
| Comparación A/B | `run_optimization.py` | 36 ajustes iniciales y 36 de estabilidad; validación interior y Early Stopping |
| Inicio persistente y pérdida robusta | `run_refinement.py` | 36 ajustes C/D y 24 adicionales admitidos por el criterio de mejora |
| Evaluación final | `run_final_evaluation.py` | 54 reajustes con configuraciones y épocas congeladas |

La validación interior reserva 252 posiciones de origen con etiquetas maduras y purga train por disponibilidad de sus tres resultados. Sus escaladores solo usan el entrenamiento interior. El checkpoint minimiza MSE interior; paciencia 5 y máximo 20 épocas. La época cero es diagnóstico. No se usa el bloque exterior para elegir época.

La formulación ya era residual antes de las variantes C/D. C cambia realmente el inicio: pesos de salida cero y sesgos `−media_train/escala_train`, de modo que el cambio reconstruido inicial es cero. D mantiene ese inicio y optimiza SmoothL1 normalizada con beta 1. La concentración de residuos de entrenamiento a 63 días justificó esa única alternativa; no se ajustaron pesos por horizonte mirando test.

RNN-D y GRU-C mejoraron frente a sus B en desarrollo; LSTM conservó B. El indicador de selección fue MAE/MAE_persistencia, con igual peso por bloque, horizonte y semilla. LSTM-B obtuvo 1,266271, GRU-C 1,296945 y RNN-D 1,388056. Se eligió LSTM-B antes de consultar calibración o test.

En evaluación final se reajusta el historial maduro completo, con escaladores propios, durante el techo de la mediana de las 18 mejores épocas de desarrollo: **RNN-D 4, LSTM-B 3, GRU-C 5**. Se conserva la última época prefijada, sin Early Stopping ni selección con resultados finales. Esta política utiliza más historia que el entrenamiento interior de selección y está documentada antes de test.

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

## Fechas, vigencia y uso de la aplicación

El checkpoint servido corresponde al **reajuste lógico del backtest del 01/07/2024**, con corte 28/06/2024. Su último origen de entrenamiento es 28/06/2023; la última etiqueta admitida estaba disponible el 28/06/2024. La fecha real de ejecución del entrenamiento no quedó registrada y no se deduce del reajuste lógico ni de la fecha de modificación del archivo.

El snapshot se descargó el 11/11/2026. En la decisión del 11/11/2026, el último Brent admitido por el retardo asumido es el de 28/10/2026, distinto del último Brent observado en el snapshot, 03/11/2026. La API distingue origen, observación de referencia, disponibilidad, actualización de datos, corte y reajuste del modelo.

Una ventana reciente no actualiza los pesos. El servicio informa que ha vencido la cadencia semestral del checkpoint histórico y conserva sus limitaciones. No se ha realizado un reajuste de producción posterior a test ni se presenta el modelo como actualizado a 2026.

La aplicación conserva la referencia ingenua visible y las redes como demostración, sin atribuirles superioridad inexistente. La actualización controlada de datos ya está implementada; un reajuste de producción requiere una fase posterior independiente y no se ejecuta desde la interfaz.

## Cumplimiento del enunciado y aclaración docente

Contraste realizado leyendo íntegramente las cuatro páginas de `docs/assignment/ENUNCIADO.pdf` y las instrucciones originales. Según la aclaración del profesor comunicada en el Prompt 9, actualizar datos es obligatorio, mantener modelos congelados está permitido y un eventual reentrenamiento no puede depender del ordenador del estudiante. Se adopta definitivamente actualización dinámica de datos con pesos congelados; no se implementa un entrenamiento manual de producción.

| Requisito del enunciado | Estado y evidencia |
|---|---|
| Brent objetivo y WTI diarios desde 2000, fuentes oficiales | Cumplido: FRED/EIA, calendario lunes–viernes y última observación publicada por cada fuente; la adquisición no elimina sus retardos. |
| Variables externas mundiales | Parcial: WTI, gas, BCE, dos cambios, dólar, oro, cobre y VIX presentes; ORB sigue sin integración fiable. |
| Ausencias, festivos y precios anómalos | Cumplido: unión por disponibilidad, límites de antigüedad, máscaras y valores originales; se conservan precios negativos de WTI. El futuro WTI negativo no se confunde con Brent spot. |
| Escalado ajustado solo con entrenamiento | Cumplido en el estudio y conservado en los artefactos; la aplicación no ajusta transformadores. |
| Ventanas para redes recurrentes | Cumplido: 60 pasos, diez series y diez máscaras; veinte entradas por paso. |
| RNN, LSTM y GRU con walk-forward y ajuste de hiperparámetros | Cumplido en los experimentos cerrados; no se repiten entrenamientos en esta fase. |
| Comparación con modelo ingenuo | Cumplido: misma referencia disponible, orígenes y objetivos; persistencia sigue siendo referencia principal. |
| Predicciones a 5, 63 y 252 días hábiles | Cumplido: tres salidas directas y calendario idéntico al backend, sin inventar una trayectoria diaria. |
| Rango de incertidumbre por horizonte | Cumplido: cuantiles nominales del 80 % congelados antes de test; sin garantía de cobertura futura. |
| MSE, MAE y RMSE por modelo/horizonte | Cumplido: métricas originales de desarrollo, calibración y test; medias y semilla 42 diferenciadas. |
| Acierto de dirección | Opcional; no se añade una nueva métrica ni se reescribe el estudio. |
| Comparación en condiciones y presupuesto similares | Cumplido: datos, ventanas, particiones y protocolo común; variantes y objetivos de pérdida seleccionados en desarrollo y declarados. |
| Tabla de errores, mejora, parámetros y tiempo | Cumplido en Comparación; vista conjunta de horizontes y recursos por semilla. |
| Modelo elegido y conclusión justificada | Cumplido: LSTM-B/42 elegida previamente entre redes; se informa que persistencia gana en medias de test y de los fallos de la red operativa. |
| Aplicación Python, histórico, horizonte y tendencias | Cumplido localmente en las cinco páginas Streamlit; modelos reales y predicciones archivadas temporalmente compatibles. |
| Precio, variación e intervalo comprensibles | Cumplido; la referencia admisible y la última cotización se distinguen por sus fechas. |
| Explicación breve, fiabilidad y factores invalidantes | Cumplido mediante reglas deterministas, MAE operativo y comparación ingenua; no se presentan probabilidades de acierto. |
| Ciclo económico y acontecimientos mínimos | Cumplido: seis episodios cubren 2008, 2014–2016, 2020, 2022 y 2026, con marcadores de datos reales, activación y fuentes oficiales. |
| Actualización dinámica | Cumplido conforme a la aclaración docente: inicio/24 h, comprobación de sesión, botón manual, última copia válida y modelos congelados. |
| Trabajo individual y transparencia académica | Declarado en el README, incluida asistencia de IA; identidad y declaración final a confirmar por el estudiante. |
| Máximo cinco diapositivas de contenido, sin portadas | Pendiente fuera de esta fase; no se prepara todavía la presentación. |
| Código en GitHub compartido y README de ejecución | Localmente completo; publicación/sincronización del código pendiente de autorización. No se entrega un fichero de código aparte. |
| Aplicación funcional y solución viable, sin complejidad técnica innecesaria | Cumplido localmente en CPU; automatización y cálculos usan el backend existente y biblioteca estándar. Linux/Spaces se verificará posteriormente. |
| Autoría propia y prohibición de plagio | Transparencia documentada; la responsabilidad y validación académica corresponden al estudiante. No se certifica una autoría ficticia. |

Los OHLC y volúmenes de futuros Brent/WTI figuran como complementos en la tabla de fuentes del profesor y permanecen en los materiales originales. La matriz recurrente cerrada utiliza las diez series descritas, sin añadir nuevas entradas ni cambiar el estudio para aprovechar esos complementos.

La búsqueda acotada de ORB confirmó que el [portal de OPEP](https://www.opec.org/opec-basket-price.html) enlaza el mismo XML histórico, pero no validó una descarga diaria completa ni condiciones de redistribución. El rechazo de MIME de la herramienta de consulta no prueba un nuevo HTTP 403: el estado de una adquisición real queda en su propio manifiesto. No se muestra una cotización no verificada ni se sustituye por medias mensuales u otra cesta.

## Reproducción y verificación

`poetry install` utiliza `poetry.lock`; el entorno contrastado es **Python 3.13.7, Poetry 2.5.1, Windows x86-64 y CPU**. Streamlit 1.67.0, Plotly 7.1.0, NumPy 2.5.4 y PyTorch 2.15.0+cpu están bloqueados. Hay 52 entradas en el lock. Matplotlib solo se requiere para regenerar figuras científicas, en un entorno separado; no es una dependencia de inferencia.

Los dos ZIP autorizados y el suplemento académico permiten ejecutar la aplicación sin `data/` ni `experiments/`. Para reproducir íntegramente el estudio y sus pruebas científicas también hacen falta los snapshots y experimentos locales originales, conservados fuera de Git. Su ubicación y recuperación están descritas en la [guía](docs/PUBLICACION.md). Una nueva descarga puede contener revisiones y no garantiza identidad con el estudio cerrado.

Con todos los recursos originales disponibles:

```bash
poetry check --lock
poetry run python -m pip check
poetry run python -B -W error -m unittest discover -s tests -v
```

Con una copia nueva del código y ambos ZIP instalados, sin los recursos de investigación:

```bash
poetry run python -B -W error -m unittest discover -s tests -p test_app.py -v
```

La segunda comprobación requiere también el ZIP original en `artifacts/releases/`, que utiliza una prueba de descarga controlada. Un clon sin esos recursos no acredita la batería completa; omitir pruebas por falta de datos no equivale a superarlas.

Los ejecutores `prepare_data.py`, `run_baseline.py`, `run_smoke.py`, `run_optimization.py`, `run_refinement.py` y `run_final_evaluation.py` conservan el procedimiento histórico, semillas, versiones, hashes y decisiones. No deben ejecutarse sobre los resultados existentes. Cualquier reproducción nueva exige permisos de uso de datos, destinos nuevos y el protocolo original. El script original del profesor se conserva como material docente: ejecutarlo puede sobrescribir su CSV. La aplicación nunca lo ejecuta.

### Comprobaciones del cierre de publicación

La comprobación completa anterior pasó **178/178 pruebas sin omisiones**. En esta fase se vuelve a verificar la batería y, adicionalmente, una instalación nueva de dependencias en un entorno temporal, con los mismos archivos de Poetry y sin reutilizar `.venv`. Los resultados definitivos de esta comprobación se registran en la [guía de publicación](docs/PUBLICACION.md).

La batería cubre las cinco páginas, cuatro modelos por tres horizontes, métricas, errores, disponibilidad temporal, actualización automática/manual, integridad, recuperación y aprovisionamiento. Las pruebas sintéticas de entrenamiento trabajan exclusivamente en temporales; no reentrenan los modelos del proyecto.

Las revisiones visuales previas cubrieron las cinco páginas en escritorio y móvil. Sus capturas y los informes históricos permanecen en `artifacts/local/app-screenshots/` y `docs/auditoria/`, fuera de Git. La versión anterior del README, con el registro cronológico de fases, se conserva localmente en `docs/auditoria/README_pre_publicacion_prompt11.md`; este README se organiza ahora por contenido académico.

### Compatibilidad y límites de la verificación

Las rutas son relativas y se admiten las claves históricas de Windows en POSIX sin reescribir manifiestos. El lock incluye wheel CPU de Linux x86-64 para Python 3.13, pero **Linux y el despliegue remoto aún no se han ejecutado**. Linux debe disponer de las zonas horarias para `Europe/Madrid`; la rama POSIX del bloqueo de actualización sigue pendiente de validación en ese sistema.

Las configuraciones del entorno histórico se preservan en `artifacts/local/research-environment/`. Sus hashes se cotejan con esa copia, sin confundirlos con el entorno ampliado de la aplicación. El código de carga incorpora adaptaciones posteriores de portabilidad y registro de cambios de entorno; no se afirma que todo el código actual sea idéntico al código histórico.

## Publicación, permisos y autoría

GitHub contendrá código, configuraciones, pruebas, lock, documentación docente y las 1.017 métricas agregadas del suplemento de R² con su manifiesto. Permanecen locales los datos, experimentos completos, pesos, transformadores, ZIP, capturas, auditorías y entornos. Los originales no se eliminan.

Los materiales del profesor se mantienen para esta entrega: sus instrucciones indican copiarlos al repositorio personal y subirlos a GitHub. Esa autorización académica no equivale a una licencia general. La licencia del código propio y la identidad académica siguen pendientes de confirmación; no se ha asignado una licencia ficticia.

**Revisión de fuentes del 17/11/2026:** los [términos actuales de FRED](https://fred.stlouisfed.org/legal/) incluyen restricciones sobre desarrollo y entrenamiento de ML; no se ha determinado su versión aplicable al estudio histórico. Esto requiere aclaración antes de nuevas adquisiciones, reproducciones o habilitación pública del servicio. La reutilización permitida por EIA de sus productos propios no resuelve las condiciones de FRED ni de las series de terceros. Los paquetes con datos Yahoo/FRED siguen fuera de publicación hasta resolver sus permisos. Detalles y fuentes en la [guía](docs/PUBLICACION.md).

El aprovisionamiento remoto futuro puede usar `BRENT_INFERENCE_URL` y `BRENT_EVIDENCE_URL`, con HTTPS y revisión fijada, para descargar los ZIP exactos verificados por SHA256. Las URL todavía no están provisionadas. El cargador no autentica Models privados solo por definir `HF_TOKEN`: se requiere aprovisionamiento autenticado externo o una ampliación posterior autorizada. No se incluyen tokens en Git ni en URLs.

La autoría es individual. Se ha utilizado asistencia de IA para programación, revisión técnica y documentación; el estudiante debe confirmar su identificación y declaración académica conforme a las normas aplicables. La responsabilidad académica no se delega en la herramienta.

La presentación de **máximo cinco diapositivas de contenido, sin contar portadas**, queda para una fase posterior. El commit y el push requieren autorización expresa después de revisar el inventario; no se publica ni despliega nada durante esta preparación.
