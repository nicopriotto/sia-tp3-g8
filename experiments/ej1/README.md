# Ejercicio 1 — Knowledge Distillation de BigModel a TinyModel

## Contexto

CompanyX usa un modelo caro (**BigModel**) para estimar la probabilidad de fraude de cada
transacción. Nos piden entrenar un **TinyModel** (perceptrón simple) que aproxime la salida
de BigModel y sea barato de inferir y almacenar.

El target de entrenamiento es la probabilidad continua que da BigModel
(`big_model_fraud_probability` ∈ [0, 1]) — **no** la columna `flagged_fraud`, que es ground
truth y solo se usa para evaluación final y para recomendar el umbral.

## Dataset

Archivo: `data/fraud_dataset.csv` (7500 filas, sin nulls).

| Columna | Tipo | Rol | Rango observado |
|---|---|---|---|
| `timestamp` | int (epoch s) | feature | ~1.7·10⁹ |
| `amount_usd` | float | feature | 1 – 2000 |
| `quantity_purchased` | int | feature | 1 – 24 |
| `session_duration_seconds` | float | feature | 5 – 727 |
| `days_since_last_purchase` | float | feature | 0 – 142 |
| `account_age_days` | int | feature | 1 – 3649 |
| `device_screen_resolution` | int (px²) | feature | 10⁶ – 8·10⁶ |
| `time_since_last_login_s` | float | feature | 10 – 40160 |
| `items_viewed_before_purchase` | int | feature | 1 – 29 |
| `big_model_fraud_probability` | float [0,1] | **target de entrenamiento** | 0.001 – 1.0 |
| `flagged_fraud` | int {0,1} | **ground truth (no usar para entrenar)** | 11.6% positivos |

Observaciones clave:
- Las features tienen escalas que difieren en hasta **9 órdenes de magnitud**
  (`timestamp ~10⁹` vs `quantity_purchased ~1`). Sin normalización, el net `w·x` saturaría
  cualquier activación no lineal (overflow en `exp` para sigmoid/tanh).
- `flagged_fraud` está desbalanceado (11.6% positivo) → para evaluar como clasificador,
  accuracy es engañoso; conviene precision/recall/F1 y curva PR.
- `device_screen_resolution` es semi-categórica (pocos valores frecuentes
  correspondientes a resoluciones comunes), pero la tratamos como numérica y la
  normalizamos.
- `timestamp` es valor absoluto epoch — probablemente no es informativo per se. Lo dejamos
  por ahora normalizado, y evaluamos descartarlo si el peso aprendido es despreciable.

## Plan de acción

### Fase 1 — Exploración (EDA)

1. Histogramas y boxplots de cada feature para detectar outliers y distribución.
2. Correlación de Pearson de cada feature con `big_model_fraud_probability`.
3. Distribución del target (histograma de la probabilidad de BigModel).
4. Confirmar ausencia de nulos y consistencia de tipos.

**Salida**: una notebook o script `eda.py` que genere estos plots en `results/ej1/eda/`.

### Fase 2 — Preprocesamiento

1. **Selección de features**: empezar con las 9 numéricas. Dejar `timestamp` por ahora.
2. **Normalización**: z-score (media 0, desvío 1) por columna, calculada sobre el subset de
   train y aplicada a val/test sin leakage. Para `device_screen_resolution` y `timestamp`,
   z-score sobre el rango completo es suficiente para evitar overflow.
3. **Split**: train / validation / test (por ej. 60 / 20 / 20) con seed fija.

**Salida**: módulo `data_loader.py` que entregue `(X_train, y_train, X_val, y_val,
X_test, y_test)` ya normalizados, más una función para guardar la media/desvío usados
(necesarios para inferir sobre nuevos datos).

### Fase 3 — Entrenamiento de las dos variantes

Reutilizamos la librería `perceptron/` con dos configs JSON:

| Variante | `activation` | `loss` | Justificación |
|---|---|---|---|
| **Lineal** | `identity` | `mse` | baseline, predice cualquier valor real |
| **No lineal** | `sigmoid` | `mse` o `binary_cross_entropy` | la salida vive en [0,1] como el target |

Entrenamos **con todas las muestras del dataset** (lo aclara el enunciado para esta primera
comparación), guardando la curva de loss por época.

**Salida**: configs en `experiments/ej1/configs/{linear,nonlinear}.json`, scripts
`train_linear.py` / `train_nonlinear.py` (o uno común parametrizado) que vuelquen
resultados en `results/ej1/<variant>/`.

### Fase 4 — Comparación de aprendizaje

Responder a las tres preguntas del enunciado mirando las curvas:

- **(a) Underfitting**: ¿el lineal queda con loss alta y plateau temprano? El perceptrón
  lineal solo puede modelar `y = w·x + b`, así que esperamos que sí cuando el target tenga
  componente no lineal.
- **(b) Saturación**: ¿el no lineal (sigmoid) queda atascado por gradientes muy pequeños
  cuando el net es grande en magnitud? Inspeccionar la distribución del net en la
  inicialización y a lo largo del training.
- **(c) Cuál usamos para generalización**: justificar con base en los puntos anteriores.
  Hipótesis a confirmar: el no lineal con sigmoid debería ganar porque el target está en
  [0,1] y la sigmoid lo respeta.

**Salida**: gráfico comparativo `loss_lineal_vs_nolineal.png` y un breve análisis en
`results/ej1/comparison.md`.

### Fase 5 — Estudio de generalización

Con el perceptrón seleccionado:

1. **Métricas elegidas y por qué**:
   - **Regresión** vs probabilidad de BigModel: MSE, MAE.
   - **Clasificación** vs `flagged_fraud` (ground truth) tras umbralizar:
     - Precision, recall, F1 (clases desbalanceadas → no usar accuracy como métrica
       principal).
     - Curva precision-recall y AUC-PR.
2. **Estrategia de manipulación de datos**:
   - K-fold cross-validation (k=5) sobre train+val para elegir el mejor modelo / hiper.
   - Test set guardado, solo se mira al final.
   - Reportar varianza entre folds, no solo el mejor.
3. **Recomendación de umbral**:
   - Barrer umbrales en [0, 1] y graficar precision/recall/F1 vs umbral usando
     `flagged_fraud` como ground truth.
   - Recomendar el umbral que maximiza F1 (o el que cumpla un nivel mínimo de recall si
     CompanyX prefiere capturar la mayor cantidad de fraudes a costa de más
     falsos positivos — discutir el trade-off en el reporte).

**Salida**: `results/ej1/generalization/` con:
- `metrics_per_fold.json`
- curvas PR y ROC contra `flagged_fraud`
- `threshold_analysis.png` y umbral recomendado en `final_report.md`

## Estructura prevista de la carpeta

```
experiments/ej1/
├── README.md                 ← este archivo
├── eda.py                    ← Fase 1
├── data_loader.py            ← Fase 2
├── configs/
│   ├── linear.json
│   └── nonlinear.json
├── train.py                  ← Fases 3 y 5
├── compare.py                ← Fase 4
└── threshold_analysis.py     ← Fase 5
```

Resultados van todos a `results/ej1/` (fuera del código, igual que en `validation/`).

## Aclaraciones del enunciado a respetar

- ❌ **No usar `flagged_fraud` para entrenar.** Solo para evaluar y elegir umbral.
- ✅ Para la primera comparación (lineal vs no lineal) **se usan todas las muestras**.
- ⚠️ Los opcionales (ReLU, feature engineering, calibración) quedan **fuera de esta primera
  iteración**.
