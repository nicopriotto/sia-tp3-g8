# SIA · TP3 · Grupo 8 — Perceptrón Simple y Multicapa

Trabajo práctico de Sistemas de Inteligencia Artificial (ITBA, 1Q 2026). El enunciado pide
implementar las cuatro variantes de perceptrón (escalón, lineal, no lineal y multicapa) y
aplicarlas a tres ejercicios: detección de fraude por destilación de un modelo grande
(ej. 1) y clasificación de dígitos manuscritos (ej. 2 y 3).

## Estructura del repositorio

```
.
├── perceptron/         ← librería core, reutilizada por todos los experimentos
│   ├── activations.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── persistence.py
│   ├── preprocessing.py            ← StandardScaler numpy-only
│   ├── utils.py
│   ├── models/                     ← SimplePerceptron + MLPPerceptron (He / Xavier init)
│   └── training/                   ← Trainer + Optimizers (SGD / Momentum / Adam con
│                                     weight_decay) + lr_schedules (none / step /
│                                     exponential / cosine)
├── experiments/        ← un sub-paquete por ejercicio
│   ├── _common/        ← evaluation y plots compartidos entre ej2 y ej3
│   ├── validation/     ← AND, regresión y=x, regresión y=tanh(x), XOR, weight_decay
│   ├── ej1/            ← Knowledge distillation de BigModel a TinyModel
│   ├── ej2/            ← Clasificación de dígitos · sweep sobre digits.csv
│   └── ej3/            ← Iteración con more_digits.csv · target accuracy ≥ 0.98
├── data/               ← datasets provistos por la cátedra
├── requirements.txt
└── README.md
```

### `perceptron/` — la librería

Implementa los modelos y toda la infraestructura de entrenamiento, separada por
responsabilidades:

- **modelos** — un perceptrón de una sola neurona que cubre los casos escalón / lineal /
  no lineal según la activación que se le pase, y un perceptrón multicapa con
  backpropagation que soporta arquitecturas arbitrarias, entrenamiento online /
  mini-batch / full-batch, y tres estrategias de inicialización de pesos
  (`uniform`, `xavier`, `he`).
- **activaciones** — `step`, `identity`, `tanh`, `sigmoid`, `relu`, `softmax`. Cada una
  expone `forward` y `derivative`, y se obtiene por nombre vía un registry.
- **entrenamiento** — un `Trainer` que orquesta el loop por épocas, registra la historia
  (loss, accuracy, métricas auxiliares), soporta early stopping con `restore_best_weights`,
  acepta `lr_schedule` (`none`, `step`, `exponential`, `cosine`) y delega los updates en un
  optimizador intercambiable. Los tres optimizadores (`SGD`, `Momentum`, `Adam`) aceptan
  `weight_decay` (regularización L2) vía `optimizer_params`.
- **preprocesamiento** — `StandardScaler` numpy-only con `fit` / `transform` /
  `fit_transform` / `inverse_transform` y serialización a/desde dict, pensado para
  pipelines anti-leakage (fit sobre train, transform sobre val/test).
- **configuración y persistencia** — `ExperimentConfig` (dataclass serializable a JSON)
  para fijar hiperparámetros (incluye campos opcionales `weight_init`, `lr_schedule`,
  `lr_schedule_params`), y utilidades para guardar y levantar los pesos junto con su
  config, de modo de poder retomar entrenamientos sin arrancar de cero.
- **métricas y datos** — funciones para cargar datasets en JSON o CSV y computar accuracy,
  MSE, precision / recall / F1, matrices de confusión multiclase, etc.

La idea es que la librería no sepa nada de los datasets concretos: los experimentos arman
sus configs y la consumen.

### `experiments/` — un sub-paquete por ejercicio

Cada ejercicio vive en su propia carpeta y es autónomo: tiene sus configs JSON, los
scripts que arman datos y entrenan modelos, y un README con el plan de trabajo. Los
resultados (pesos, métricas por época, plots) se vuelcan a `results/<exp>/` fuera del
código fuente, lo que mantiene los experimentos reproducibles y separa claramente las
piezas que generan datos de las que los analizan.

Sub-paquetes:

- **`_common/`** — utilidades genéricas reusadas por ej2 y ej3 (`evaluate_multiclass`,
  `save_evaluation`, `plot_loss_curve`, `plot_confusion_matrix`, etc.). ej2 y ej3 las
  reexportan desde sus propios `evaluation.py` / `plots.py` para preservar las APIs
  históricas.
- **`validation/`** — los cuatro ejercicios de validación opcionales del enunciado
  (AND con escalón, regresión lineal y=x, regresión no lineal y=tanh(x), XOR con MLP en
  arquitecturas `[2,2,1]` y `[2,3,2,1]`) más un test de `weight_decay` en los tres
  optimizadores. Sirven como sanity check de la librería.
- **`ej1/`** — destilación de BigModel a TinyModel para detección de fraude. EDA, k-fold
  cross-validation, learning curve, análisis de umbral. Plan en
  [experiments/ej1/README.md](experiments/ej1/README.md).
- **`ej2/`** — clasificación de dígitos manuscritos con perceptrón multicapa.
  Sweep de arquitectura, learning rate y optimizador sobre `digits.csv`. Plan
  en [experiments/ej2/README.md](experiments/ej2/README.md).
- **`ej3/`** — segunda iteración del problema de dígitos: alcanzar accuracy
  ≥ 98 % incorporando el dataset adicional `more_digits.csv`. Estrategia de datos
  (`only_more` vs `combined`), ablation por datos, sweep de configs (arquitectura,
  optimizador, init, weight_decay, lr_schedule). Plan en
  [experiments/ej3/README.md](experiments/ej3/README.md).

### `data/`

Datasets provistos por la cátedra (`fraud_dataset.csv`, `digits.csv`, `digits_test.csv`,
`more_digits.csv`) y su documentación oficial. No se modifican.

## Setup

```bash
pip install -r requirements.txt
```

Dependencias: `numpy`, `pandas`, `matplotlib`, `scikit-learn` (Python ≥ 3.10).

## Cómo correr los experimentos de validación

```bash
python -m experiments.validation.ex_0_1_and
python -m experiments.validation.ex_0_2_linear --config experiments/validation/configs/linear_clean.json
python -m experiments.validation.ex_0_3_nonlinear --config experiments/validation/configs/nonlinear_clean.json
python -m experiments.validation.ex_0_4_mlp --config experiments/validation/configs/mlp_xor_2_2_1.json
python -m experiments.validation.ex_0_4_mlp --config experiments/validation/configs/mlp_xor_2_3_2_1.json
python -m experiments.validation.ex_0_6_weight_decay
```

Cada corrida deja en `results/validation/<nombre>/`: pesos `.npz`, `config.json`,
`history.json`, `evaluation.json` y plots PNG (curva de loss, frontera de decisión cuando
aplica).

## Cómo correr los ejercicios

Cada ejercicio sigue el mismo patrón: EDA → entrenamiento (uno o varios configs) →
consolidación → evaluación final una sola vez sobre el test.

```bash
# Ejercicio 1 — knowledge distillation
python -m experiments.ej1.eda
python -m experiments.ej1.train --config experiments/ej1/configs/linear.json
python -m experiments.ej1.train --config experiments/ej1/configs/nonlinear_sigmoid_mse.json
python -m experiments.ej1.compare_learning
python -m experiments.ej1.generalization

# Ejercicio 2 — dígitos manuscritos (sweep base)
python -m experiments.ej2.eda
python -m experiments.ej2.train --config experiments/ej2/configs/<config>.json
python -m experiments.ej2.summarize
python -m experiments.ej2.evaluate_final

# Ejercicio 3 — dígitos manuscritos con datos extendidos
python -m experiments.ej3.eda
python -m experiments.ej3.run_all
python -m experiments.ej3.summarize
python -m experiments.ej3.evaluate_final
```

`digits_test.csv` se carga **una sola vez por ejercicio**, en `evaluate_final`. Toda
selección de hiperparámetros se hace mirando solo train / validation.
