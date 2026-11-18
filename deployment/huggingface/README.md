---
language:
  - es
library_name: pytorch
tags:
  - time-series
  - forecasting
  - recurrent-neural-network
  - rnn
  - lstm
  - gru
  - brent
  - academic
---

# Brent Forecast: RNN-D, LSTM-B y GRU-C congeladas

Ficha académica del Caso 2 de Aprendizaje Automático de la UAX. El estudio compara tres redes recurrentes y persistencia para predecir el precio spot del Brent a **5, 63 y 252 días lunes–viernes**, en USD/barril.

**Estado: preparación local para un posible repositorio privado; no se han subido los paquetes ni autorizado su redistribución.** El namespace `Juaandeleoon` fue indicado por el estudiante; no se ha verificado una sesión autenticada en ese namespace. Esta ficha no crea un repositorio, no configura su privacidad y no concede una licencia. La identificación académica y la declaración final del estudiante requieren confirmación; se ha empleado asistencia de IA en programación, revisión técnica y documentación.

Nombre propuesto del repositorio de tipo Model: **`brent-forecast-rnn-lstm-gru`**. Destino indicado: **`Juaandeleoon/brent-forecast-rnn-lstm-gru`**. Su visibilidad inicial prevista es **privada**, todavía no configurada remotamente. No hay una revisión remota ni una descarga provisionada.

**No es una herramienta de predicción financiera fiable ni apta para decisiones de inversión.** Es una demostración académica con pesos históricos, errores relevantes y persistencia como referencia principal.

**Resultado principal:** persistencia supera a las tres arquitecturas neuronales al promediar las tres semillas de test, en los tres horizontes. LSTM-B con semilla 42 fue seleccionada antes de consultar test y se conserva como demostración, con sus resultados negativos visibles.

Código y protocolo publicados: [GitHub, revisión 389530b78d42a1f76273c8f2608c08cc5c6805ce](https://github.com/Juandeleongarcia/caso-2-prediccion-petroleo/tree/389530b78d42a1f76273c8f2608c08cc5c6805ce). Las adaptaciones locales posteriores de despliegue deberán vincularse a su revisión real cuando se autorice publicarlas; no se inventa una revisión futura ni una URL de aplicación.

## Contenido previsto y estado de permisos

La estructura propuesta conserva los dos ZIP originales exactos:

```text
README.md
brent-frozen-v1.zip
brent-dashboard-v1.zip
```

**Control del Prompt 13:** ambos ZIP quedan excluidos de cualquier carga por permisos sin resolver. La lista anterior describe los artefactos técnicos originales; no es una lista autorizada de subida. Esta ficha documental es el único candidato actual para una posible carga como `README.md`, y también requiere autorización expresa antes de crear el destino o publicar contenido. No se extraen pesos ni se reconstruye un paquete alternativo.

| Archivo local original | Bytes | SHA256 |
|---|---:|---|
| `artifacts/releases/brent-frozen-v1.zip` | 176139 | `2ccf6f181772420dfbd1650c726b72ac6c936f1e2eb1fed82b3ef99c049f7f2a` |
| `artifacts/releases/brent-dashboard-v1.zip` | 2594014 | `2b26b26d90383c3ab0c27a563714cae0ca668f94538d7e1bfcdbcd2542a45620` |

El paquete de inferencia incluye `checkpoints.pt`, con los tres modelos canónicos de semilla 42 y sus transformadores; `protocol.json`; `intervals.json`; métricas puntuales y de intervalos; metadatos; histórico Brent; última ventana y predicciones de referencia. Su manifiesto verifica nueve archivos internos. No se necesitan pesos adicionales para persistencia.

El paquete de evidencia conserva cinco archivos y su manifiesto: catálogo de experimentos, curvas de entrenamiento, métricas, métricas de intervalos y predicciones archivadas de ejemplo. No contiene otros checkpoints. La evidencia no equivale a todos los datos y experimentos históricos originales.

El suplemento académico `artifacts/derived/brent-academic-v1/` permanece con el código: sus métricas agregadas añaden R² a partir de las predicciones originales sin sustituir métricas, predicciones o paquetes congelados. Es necesario para consultar las cuatro métricas en la aplicación.

**Permisos pendientes para ambos ZIP, incluidos pesos, transformadores y cuantiles derivados.** Los paquetes contienen además datos de terceros; sus hashes no acreditan derechos de uso, alojamiento o redistribución. Los [términos actuales de FRED](https://fred.stlouisfed.org/legal/) y las [condiciones de datos indicadas por yfinance](https://github.com/ranaroussi/yfinance#readme) requieren aclaración. No se determina retrospectivamente qué términos eran aplicables al estudio histórico.

Preparar un destino privado no autoriza subir contenido ni elimina esas restricciones. Tampoco autoriza exponer datos o predicciones archivadas desde una aplicación pública. La revisión de permisos debe cubrir el alojamiento, los derivados, la actualización y las salidas visibles antes de una publicación efectiva. No se asigna MIT ni otra licencia por defecto, ni se marca el contenido como abierto.

## Modelos e inferencia

| Modelo | Variante | Semilla canónica | Épocas del reajuste final | Parámetros |
|---|---|---:|---:|---:|
| RNN | D, inicio persistente y SmoothL1 | 42 | 4 | 1827 |
| LSTM | B, selección previa entre redes | 42 | 3 | 7011 |
| GRU | C, inicio persistente | 42 | 5 | 5283 |
| Persistencia | Último Brent admisible | No aplica | No entrena | 0 |

Las tres redes utilizan una capa recurrente de 32 unidades, ventanas de **60 pasos**, **20 entradas por paso** y tres salidas directas. Las entradas son diez variables económicas y sus diez máscaras de ausencia. La formulación es residual: `precio futuro = Brent disponible en el origen + cambio predicho`. Persistencia fija el cambio a cero.

El entrenamiento usa Adam, batch 64, learning rate 0,0003, weight decay 0,0001 y clipping de gradiente 1. Las semillas 123 y 2026 evalúan estabilidad; la aplicación sirve la convención previa 42, sin escoger después la semilla con mejor test. Los estados de imputación y normalización se ajustaron solo en el entrenamiento permitido de cada fold y se recuperan del checkpoint; inferir no los reajusta.

El código emplea módulos PyTorch propios y su cargador verificado. Estos ZIP no son un modelo compatible automáticamente con `AutoModel`, un pipeline estándar de Transformers ni un endpoint de inferencia contratado. Se ejecutan localmente en CPU con el código del proyecto.

## Datos y disponibilidad temporal

El snapshot del estudio es `2026-11-11-official`, con calendario lunes–viernes desde enero de 2000. FRED aporta Brent y WTI spot, gas natural, tipo del BCE, USD/EUR, CNY/USD y VIX; Yahoo Finance aporta índice del dólar, oro y cobre. ORB de OPEP permanece ausente, como limitación registrada. No se fabrican etiquetas Brent ni se incorporan OHLC o volumen a la matriz final.

Las decisiones se sitúan a las 08:00 de `Europe/Madrid`. Se aplican retardos asumidos de diez días lunes–viernes a spot, gas y cambios, y de dos a BCE, VIX y cierres Yahoo. Se unen covariables hacia atrás según disponibilidad y se controlan su antigüedad y procedencia. No se rellena con observaciones futuras.

Los retardos son supuestos explícitos, no calendarios oficiales certificados de publicación. El vintage descargado puede contener revisiones: **el estudio no es point-in-time certificado**. Las comprobaciones acreditan causalidad bajo la política implementada, sin descartar revisiones o disponibilidad histórica real diferente.

## Selección y evaluación temporal

| Conjunto | Años de los orígenes | Uso |
|---|---|---|
| Desarrollo | 2015, 2017 y 2019 | Variantes, épocas y selección de arquitectura |
| Calibración | 2021–2022 | Residuos prequential para incertidumbre |
| Test | 2024; objetivos largos hasta 2025 | Evaluación final de la política congelada |

El walk-forward es expansivo, con reajustes semestrales planificados. Todos los modelos se comparan sobre los mismos orígenes y fechas objetivo. La construcción del entrenamiento exige madurez conjunta de los tres horizontes y ajusta transformaciones solo con su historia admisible.

La validación interior cronológica y Early Stopping se aplicaron en desarrollo. En la evaluación final se conservaron épocas prefijadas de 4/3/5 para RNN/LSTM/GRU, sin elegir checkpoints mediante test. LSTM-B fue seleccionada en desarrollo; test no cambia esa decisión.

El test es online bajo política congelada. Ningún ejemplo con origen en 2024 entrena las redes; un reajuste posterior puede utilizar como etiquetas de orígenes antiguos precios de 2024 que ya estén publicados antes de su corte. No se presenta como un holdout estático de todos los precios observados en test.

## Resultados reales de test

**238 orígenes por horizonte y semilla.** Las cifras neuronales son medias de métricas de las tres semillas; no son un ensemble ni 714 observaciones independientes. Persistencia tiene un único resultado. MAE y RMSE están en USD/barril, MSE en USD²/barril² y R² es adimensional. La media de RMSE no tiene por qué coincidir con la raíz del MSE medio.

| Horizonte L–V | Modelo | MAE | MSE | RMSE | R² |
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

La mejora respecto a persistencia se define como `100 × (1 − MAE_modelo / MAE_persistencia)`. Las medias neuronales son peores en los tres horizontes. R² puede ser negativo y compara con la media retrospectiva de los objetivos evaluados; no es una probabilidad de acierto ni sustituye al baseline operativo.

### Red seleccionada para la aplicación: LSTM-B/42

| Horizonte L–V | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| 5 | 3,574699 | 21,454060 | 4,631853 | 0,299981 |
| 63 | 5,573487 | 47,873313 | 6,919054 | −0,619356 |
| 252 | 21,525380 | 502,868895 | 22,424738 | −16,927106 |

A 252 días, la semilla operativa 42 empeora el MAE ingenuo de 11,435420 a 21,525380 USD/barril y sobreestima los 238 objetivos de test. Hay elevada variabilidad entre semillas. Escoger retrospectivamente otra semilla por su test alteraría el criterio previo; no se hace.

## Incertidumbre congelada

Los intervalos nominales del 80 % son `predicción ± q`, usando errores absolutos de calibración prequential por modelo, semilla y horizonte, en unidades originales. El cuantil ocupa la posición `ceil((n+1) × 0,8)`, sin interpolación y limitada a `n`.

Cada grupo utiliza **471 orígenes**: se excluye de los tres horizontes el origen 29/12/2022 por madurez de su etiqueta larga justo en la frontera de test. Las métricas puntuales de calibración conservan sus **472 orígenes** originales. Los cuantiles se congelaron antes de test, no se ajustan con él y no se agrupan semillas como observaciones adicionales.

Para LSTM-B/42, las coberturas observadas en test a 5/63/252 días fueron **94,54/100/97,06 %**, con amplitudes medias **18,62/34,18/63,40 USD/barril**. Cobertura alta con bandas amplias no significa calibración exacta al 80 %. GRU-C a 252 días muestra infracobertura media del 63,59 %. Dependencia temporal, horizontes solapados y cambios de régimen impiden prometer cobertura futura exacta.

## Fechas, vigencia y actualización

| Concepto | Valor del checkpoint portable |
|---|---|
| Reajuste lógico del backtest | 01/07/2024, 08:00 Madrid (`2024-07-01T06:00:00+00:00`) |
| Corte de entrenamiento | 28/06/2024, 08:00 Madrid |
| Último origen de entrenamiento | 28/06/2023 |
| Última disponibilidad de etiqueta admitida | 28/06/2024, 08:00 Madrid |
| Ejecución real del entrenamiento | **No registrada**; no se deduce de las fechas lógicas ni del archivo |
| Adquisición del snapshot portable | `2026-11-11T21:08:21.056273+00:00` |

Los pesos permanecen congelados en ese reajuste lógico histórico. Una ventana reciente no los convierte en modelos actualizados a 2026; ha vencido la cadencia semestral del checkpoint. Actualizar datos e inferir no cambia pesos, imputadores, normalizadores ni cuantiles. No hay reentrenamiento automático.

La aplicación distingue adquisición, fecha de observación, disponibilidad, origen y antigüedad de los pesos. La actualización conserva la última copia completa y válida ante fallos de proveedores. Si el alojamiento pierde sus archivos efímeros, necesita recuperar los paquetes autorizados y verificar sus hashes; no depende de que el ordenador del estudiante esté encendido. La compatibilidad del servicio remoto solo podrá declararse después de comprobar un despliegue real autorizado.

## Uso previsto, límites y pasos pendientes

El uso previsto es demostración académica y análisis reproducible del experimento, con persistencia visible como referencia principal. El estudio no acredita superioridad predictiva de las redes, calibración exacta, disponibilidad histórica certificada ni vigencia de los pesos para una decisión económica actual.

Antes de cualquier carga deben confirmarse permisos, alcance privado/público, identificación académica y repositorio destino. Después habrá que fijar el commit completo de los ZIP en Hugging Face y mantener sus SHA256 registrados independientemente en GitHub. El token de lectura se guardará en secretos del servidor; nunca en esta ficha, el código, una URL o un repositorio público.

Esta preparación no modifica los modelos originales, no ejecuta experimentos y no publica archivos. El alojamiento privado y la autenticación son medidas técnicas; no sustituyen la autorización de uso y difusión.
