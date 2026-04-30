# SIA · TP3 · Grupo 8 — Perceptrón Simple y Multicapa

Trabajo práctico de Sistemas de Inteligencia Artificial (ITBA, 1Q 2026). El enunciado pide
implementar las cuatro variantes de perceptrón (escalón, lineal, no lineal y multicapa) y
aplicarlas a tres ejercicios: detección de fraude por destilación de un modelo grande
(ej. 1) y clasificación de dígitos manuscritos (ej. 2 y 3).

## Estructura del repositorio

```
.
├── perceptron/         ← librería core, reutilizada por todos los experimentos
├── experiments/        ← un sub-paquete por ejercicio
│   ├── validation/     ← AND, regresión y=x, regresión y=tanh(x), XOR
│   └── ej1/            ← Knowledge distillation de BigModel a TinyModel (en curso)
├── data/               ← datasets provistos por la cátedra
├── requirements.txt
└── README.md
```

### `perceptron/` — la librería

Implementa los modelos y toda la infraestructura de entrenamiento, separada por
responsabilidades:

- **modelos** — un perceptrón de una sola neurona que cubre los casos escalón / lineal /
  no lineal según la activación que se le pase, y un perceptrón multicapa con
  backpropagation que soporta arquitecturas arbitrarias y entrenamiento online,
  mini-batch o full-batch.
- **activaciones** — `step`, `identity`, `tanh`, `sigmoid`, `relu`, `softmax`. Cada una
  expone `forward` y `derivative`, y se obtiene por nombre vía un registry.
- **entrenamiento** — un `Trainer` que orquesta el loop por épocas, registra la historia
  (loss, accuracy, métricas auxiliares), soporta early stopping y delega los updates en un
  optimizador intercambiable (SGD, Momentum, Adam).
- **configuración y persistencia** — `ExperimentConfig` (dataclass serializable a JSON)
  para fijar hiperparámetros, y utilidades para guardar y levantar los pesos junto con su
  config, de modo de poder retomar entrenamientos sin arrancar de cero.
- **métricas y datos** — funciones para cargar datasets en JSON o CSV y computar accuracy,
  MSE, precision/recall/F1, etc.

La idea es que la librería no sepa nada de los datasets concretos: los experimentos arman
sus configs y la consumen.

### `experiments/` — un sub-paquete por ejercicio

Cada ejercicio vive en su propia carpeta y es autónomo: tiene sus configs JSON, los
scripts que arman datos y entrenan modelos, y un README con el plan de trabajo. Los
resultados (pesos, métricas por época, plots) se vuelcan a `results/<exp>/` fuera del
código fuente, lo que mantiene los experimentos reproducibles y separa claramente las
piezas que generan datos de las que los analizan.

Hoy hay dos sub-paquetes:

- **`validation/`** — los cuatro ejercicios de validación opcionales del enunciado
  (AND con escalón, regresión lineal y=x, regresión no lineal y=tanh(x), y XOR con MLP en
  arquitecturas `[2,2,1]` y `[2,3,2,1]`). Sirven como sanity check de la librería.
- **`ej1/`** — destilación de BigModel a TinyModel para detección de fraude. El plan
  detallado está en [experiments/ej1/README.md](experiments/ej1/README.md).

### `data/`

Datasets provistos por la cátedra (`fraud_dataset.csv`, `digits.csv`, etc.) y su
documentación oficial. No se modifican.

## Setup

```bash
pip install -r requirements.txt
```

Dependencias: `numpy`, `pandas`, `matplotlib` (Python ≥ 3.10).

## Cómo correr los experimentos de validación

```bash
python -m experiments.validation.ex_0_1_and
python -m experiments.validation.ex_0_2_linear --config experiments/validation/configs/linear_clean.json
python -m experiments.validation.ex_0_3_nonlinear --config experiments/validation/configs/nonlinear_clean.json
python -m experiments.validation.ex_0_4_mlp --config experiments/validation/configs/mlp_xor_2_2_1.json
python -m experiments.validation.ex_0_4_mlp --config experiments/validation/configs/mlp_xor_2_3_2_1.json
```

Cada corrida deja en `results/validation/<nombre>/`: pesos `.npz`, `config.json`,
`history.json`, `evaluation.json` y plots PNG (curva de loss, frontera de decisión cuando
aplica).
