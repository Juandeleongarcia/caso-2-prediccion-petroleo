# Guía de revisión académica

Estado de cierre: **19/11/2026**. El [README principal](../README.md) describe el estado actual; las guías por fases conservan también decisiones y comprobaciones históricas.

## Recorrido para evaluar el proyecto

1. Leer el [enunciado original](assignment/ENUNCIADO.pdf) y las [instrucciones del profesor](README_profesor.md), conservados íntegros.
2. Revisar [datos, metodología y resultados](../README.md#datos-y-fuentes). Persistencia gana en los errores medios de test; ORB y reentrenamiento periódico siguen pendientes.
3. Abrir la [web pública](https://brent-forecast.vercel.app/): cinco secciones con inferencia real. Los pesos son históricos; la fecha de adquisición de datos se muestra por separado.
4. Consultar el [PDF de enlaces](entrega/Enlaces_Brent_Forecast.pdf). Hugging Face sigue privado; el acceso de la profesora no se ha concedido todavía.
5. Para reproducción local, seguir el [README](../README.md#reproducción-e-instalación-local). Un clon no contiene los artefactos restringidos ni todos los experimentos.

## Documentación técnica

| Documento | Finalidad y alcance |
|---|---|
| [Actualización diaria](ACTUALIZACION_DIARIA.md) | Canal live, validaciones, secretos, recuperación y propuesta separada de reentrenamiento. Incluye el estado inicial anterior a su activación. |
| [Importación de Vercel](VERCEL_IMPORTACION.md) | Configuración del paquete autónomo, diagnóstico y contratos de despliegue. Conserva comprobaciones de fases anteriores. |
| [Plan de despliegue](PLAN_DESPLIEGUE.md) | Decisiones de arquitectura y limitaciones operativas; no sustituye al estado comprobado del README. |
| [Inventario de artefactos](deployment/artifact_inventory.csv) | 92 entradas originales con función, tamaño y SHA256; no es un catálogo de descargas públicas. |
| [Guía de publicación e historial](PUBLICACION.md) | Registro de auditorías, permisos y fases anteriores. Frases como «pendiente», «presentación pausada» o «no desplegado» describen esas fases, no el cierre actual. |
| [Web local](../web/README.md) | Arranque y contratos originales de la migración; la web actual está desplegada. |

## Conservación y entrega

Los directorios `data/`, `experiments/`, `artifacts/releases/`, `artifacts/local/`, `docs/auditoria/` y `presentacion/` se conservan localmente y están excluidos de Git. No se han movido datos, pesos, configuraciones, métricas ni manifiestos. Las duplicaciones del snapshot `deployment/vercel-app/` son necesarias para el Root Directory de Vercel; la demostración histórica limitada se conserva por sus referencias y pruebas.

La presentación definitiva está en `presentacion/entrega_final_2026-11-19/Brent_Forecast_Defensa_Final_UAX.pdf`, junto al PowerPoint original: portada más cinco diapositivas de contenido. Se entrega por separado del PDF de enlaces. Los guiones y borradores no se añaden como páginas de esa presentación. No se prepara un ZIP ni se sube a Moodle desde el repositorio.
