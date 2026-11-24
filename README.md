# Clasificador de enfermedades en hoja de maiz

Clasifica una foto de hoja de maiz en 4 clases: `healthy`, `common_rust`, `gray_leaf_spot` y
`northern_blight`. Esta pensado como **filtro de primera revision, no como reemplazo del
agronomo**: si la confianza es baja, la API lo indica para que una persona revise.

Se comparan dos modelos: una CNN simple entrenada desde cero y una EfficientNet-B0
preentrenada en ImageNet y ajustada (fine-tuning). El modelo final se sirve con FastAPI.

## Estructura

```
.
├── src/
│   ├── config.py            # constantes: clases, tamanos, umbrales
│   ├── data/
│   │   ├── validation.py    # validacion de imagenes, etiquetas y fuga entre conjuntos
│   │   └── preprocessing.py # foto -> 16 recortes de 96 px -> tensor normalizado
│   ├── models/
│   │   ├── loader.py        # arquitectura y carga de pesos (local o S3)
│   │   └── predictor.py     # prediccion promediando recortes
│   ├── api/main.py          # FastAPI: /health y /predict
│   └── monitoring/logger.py # log de predicciones y resumen
├── tests/                   # pytest: datos, preprocesamiento, API, monitoreo
├── notebooks/               # EDA y entrenamiento (Colab / SageMaker)
├── data/                    # datos locales (fuera de git)
├── models/                  # pesos .pt (fuera de git)
├── .github/workflows/ci.yml # GitHub Actions
└── requirements.txt         # dependencias de ejecucion
```

## Datos

Dataset: subconjunto de maiz de [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
(3,852 imagenes de 256x256). Se usa la variante `segmented` (fondo negro) y se entrena con
recortes de 96 px sin fondo negro. El split 70/15/15 se hizo **por grupos de duplicados** y
**antes** de aumentar datos, para evitar fuga entre train, val y test.

Los datos y pesos no estan en git. Se guardan en S3:

```
s3://sagemaker-maize-disease-248471876373-us-east-1/maize/raw_segmented/{clase}/      imagenes
s3://sagemaker-maize-disease-248471876373-us-east-1/maize/splits/split_v1.csv         split train/val/test
s3://sagemaker-maize-disease-248471876373-us-east-1/maize/class_mapping.json          orden de clases
s3://sagemaker-maize-disease-248471876373-us-east-1/maize/models/                     pesos entrenados
```

Para validar el split desde la terminal:

```bash
python -m src.data.validation --csv data/split_v1.csv --root data/raw_segmented
```

## Modelos

El entrenamiento esta en `notebooks/`:

- `efficientnet_b0_v3.pt`: modelo final que usa la API.
- `class_mapping.json`: orden de clases.

Si el archivo no esta, la API lo descarga de `MODEL_S3_URI` (por ejemplo
`s3://sagemaker-maize-disease-248471876373-us-east-1/maize/models/efficientnet_b0_v3.pt`).

## API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

| Ruta | Descripcion |
|---|---|
| `GET /health` | `{"status": "ok", "model_loaded": true}` |
| `POST /predict` | recibe una imagen (`file`) y devuelve la clase, la probabilidad de cada clase y si necesita revision |


```json
{
  "predicted_class": "common_rust",
  "confidence": 0.93,
  "probabilities": {"common_rust": 0.93, "gray_leaf_spot": 0.03, "healthy": 0.01, "northern_blight": 0.03},
  "needs_review": false,
  "n_patches": 8
}
```

- Si `confidence < 0.70` (`LOW_CONFIDENCE`), `needs_review` es `true` y se agrega un mensaje.
- Archivo que no es imagen valida: `400` con el motivo. Sin modelo cargado: `503`.
- Mas de 10 MB: `413`.
- La imagen se procesa con el mismo método que se usó para medir el modelo en fotos externas
  (16 recortes de 96 px, promedio de probabilidades), para que el servicio se comporte como
  en la evaluación.

Variables de entorno: `MODELS_DIR`, `MODEL_FILE`, `MODEL_S3_URI`, `LOW_CONFIDENCE`,
`PREDICTION_LOG`.

## Monitoreo

Cada prediccion se guarda en `logs/predictions.jsonl` (fecha, clase, confianza, si necesito
revision, latencia). No se guarda la imagen. `src.monitoring.logger.summarize()` da el total,
el porcentaje enviado a revision, la confianza media y el conteo por clase; un aumento de
revisiones o un cambio en la mezcla de clases es senal de que las fotos reales difieren del
entrenamiento. `upload_log_to_s3()` sube el log a S3.


## Arquitectura en AWS

| Servicio | Uso |
|---|---|
| Amazon S3 | datos, modelos y logs de prediccion |
| Amazon SageMaker (notebook) | EDA y procesamiento en una instancia chica |
| Amazon EC2 | corre la API FastAPI, que baja el modelo desde S3 |

## Resultados y limitaciones

| Modelo | Accuracy (test) | F1 macro (test) |
|---|---|---|
| Random Forest (histogramas HSV) | 0.903 | 0.861 |
| CNN simple | 0.952 | 0.936 |
| CNN simple (color) | 0.966 | 0.951 |
| EfficientNet-B0 v1 | 0.950 | 0.937 |
| EfficientNet-B0 v2 (160 px) | 0.937 | 0.922 |
| EfficientNet-B0 v3 (final) | 0.962 | 0.949 |

- En PlantVillage, EfficientNet-B0 v3 y la CNN simple de color rinden prácticamente igual
  (F1 0.949 y 0.951); la diferencia es menor que el ruido de un conjunto de test de este
  tamaño. El error principal es confundir `gray_leaf_spot` con `northern_blight`.
- PlantVillage tiene fondos controlados (casi toda la roya tiene fondo negro). El rendimiento
  en test esta inflado respecto a fotos reales: en 20 fotos externas se obtuvo entre 75% y 90%
  segun el modelo.
