# SIA · TP3 · Grupo 8 — Perceptrón simple y multicapa

Trabajo práctico de Sistemas de Inteligencia Artificial (ITBA, 1Q 2026).

El repo contiene:

- una librería propia de perceptrones en `perceptron/`,
- scripts de validación y sanity checks en `experiments/validation/`,
- tres ejercicios del TP en `experiments/ej1`, `experiments/ej2` y `experiments/ej3`,
- utilidades auxiliares de interpretabilidad en `experiments/extras/`.

## Estructura del repositorio

```text
.
├── data/                       ← datasets provistos por la cátedra
├── perceptron/                 ← librería core reutilizada por todos los experimentos
│   ├── activations.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── persistence.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── models/
│   └── training/
├── experiments/
│   ├── _common/                ← evaluation/plots compartidos por ej2 y ej3
│   ├── validation/             ← AND, regresión, XOR, smoke tests y checks de weight decay
│   ├── ej1/                    ← distillation de BigModel a TinyModel sobre fraude
│   ├── ej2/                    ← clasificación de dígitos con digits.csv
│   │   ├── configs/            ← configs oficiales del flujo base
│   │   ├── configs_grid/       ← grilla auxiliar de barridos
│   │   └── experiments/        ← barridos auxiliares adicionales
│   ├── ej3/                    ← clasificación de dígitos con datos extendidos
│   └── extras/                 ← análisis auxiliares fuera del flujo oficial
├── requirements.txt
└── README.md
```

## Qué hay en cada paquete

### `perceptron/`

La librería implementa los modelos y la infraestructura común:

- `models/`
  `SimplePerceptron` para los casos escalón, lineal y no lineal de una sola neurona, y `MLPPerceptron` para arquitecturas multicapa arbitrarias.
- `activations.py`
  Activaciones `step`, `identity`, `tanh`, `sigmoid`, `relu`, `softmax`. Las activaciones diferenciables exponen derivada; `step` no, y por eso usa regla del perceptrón en vez de backprop.
- `training/`
  `Trainer`, optimizadores `sgd` / `momentum` / `adam`, regularización L2 vía `weight_decay`, y schedules `step`, `exponential` y `cosine`.
- `config.py`
  `ExperimentConfig` serializable a JSON, usado por todos los scripts.
- `persistence.py`
  Guarda y reconstruye modelos desde `weights.npz` + `config.json`.
- `data.py`
  Helpers genéricos para CSV/JSON, one-hot, splits train/validation y standardization básica.
- `preprocessing.py`
  `StandardScaler` numpy-only, usado en pipelines anti-leakage.
- `metrics.py`
  Accuracy, MSE, MAE, precision/recall/F1, matrices de confusión y threshold sweeps.

### `experiments/validation/`

Contiene seis scripts cortos para validar la infraestructura:

- `ex_0_1_and.py`
- `ex_0_2_linear.py`
- `ex_0_3_nonlinear.py`
- `ex_0_4_mlp.py`
- `ex_0_5_core_smoke.py`
- `ex_0_6_weight_decay.py`

Los cuatro primeros entrenan modelos y guardan artefactos en `results/validation/...`.
`ex_0_5_core_smoke.py` y `ex_0_6_weight_decay.py` son checks tipo smoke/unit test: verifican invariantes y terminan imprimiendo éxito si todo está bien.

### `experiments/ej1/`

Knowledge distillation para aproximar `big_model_fraud_probability` con un perceptrón simple.

Puntos importantes del flujo real:

- el target de entrenamiento es `big_model_fraud_probability`,
- `flagged_fraud` se usa para evaluación como clasificación y análisis de umbral,
- `prepare_data()` hace split interno train/validation/test sobre `fraud_dataset.csv`,
- no existe `evaluate_final.py`: tanto `train.py` como `generalization.py` reportan métricas sobre el test hold-out del dataset de fraude.

Más detalle en [experiments/ej1/README.md](experiments/ej1/README.md).

### `experiments/ej2/`

Clasificación de dígitos manuscritos con `digits.csv`.

Flujo oficial:

- `eda.py`
- `train.py`
- `run_all.py`
- `compare_experiments.py`
- `summarize.py`
- `evaluate_final.py`

Restricción importante:

- `digits_test.csv` se reserva para `evaluate_final.py`.
- La selección de modelo se hace solo con train/validation.

Limitación estructural del dataset:

- `digits.csv` no contiene la clase `8`.
- Por eso los reportes finales de ej2 remarcan explícitamente que esa clase queda fuera de distribución.

Además del flujo oficial, `ej2` incluye assets auxiliares:

- `config_base.json`
  plantilla base para barridos adicionales,
- `configs_grid/`
  grilla auxiliar de configs,
- `experiments/`
  barridos adicionales por activación, loss, batch size, weight init, etc.

Esos barridos auxiliares escriben bajo `results/ej2/comparasion/` con esa ortografía, porque así quedó fijado en el código. El flujo oficial base usa `results/ej2/comparisons/`.

Más detalle en [experiments/ej2/README.md](experiments/ej2/README.md).

### `experiments/ej3/`

Segunda iteración del problema de dígitos, ahora usando `more_digits.csv` además de `digits.csv`.

Flujo oficial:

- `eda.py`
- `train.py`
- `run_all.py`
- `summarize.py`
- `evaluate_final.py`

Puntos importantes:

- la estrategia de datos se controla con `config.extra.data_strategy`,
- `run_all.py` corre por default el sweep principal; `ablation_data_only.json` se corre por separado si se quiere repetir esa ablación puntual,
- `digits_test.csv` también se toca solo en `evaluate_final.py`.

Más detalle en [experiments/ej3/README.md](experiments/ej3/README.md).

### `experiments/extras/`

Espacio aislado para análisis auxiliares que no forman parte del flujo oficial de selección de modelos.

Hoy contiene `ej2_interpretability/`, que analiza runs ya entrenados de ej2 y genera reportes de interpretabilidad en `results/ej2/interpretability/`.

## Datasets

Bajo `data/` están los archivos provistos por la cátedra:

- `fraud_dataset.csv`
- `digits.csv`
- `digits_test.csv`
- `more_digits.csv`
- `fraud_dataset_documentation.pdf`

También hay un helper local, `digit_dataset_loader.py`, usado como utilidad ad hoc sobre el dataset de dígitos.

## Setup

Requisitos:

- Python 3.10 o superior
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Instalación:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cómo correr los checks de validación

```bash
python3 -m experiments.validation.ex_0_1_and
python3 -m experiments.validation.ex_0_2_linear --config experiments/validation/configs/linear_clean.json
python3 -m experiments.validation.ex_0_3_nonlinear --config experiments/validation/configs/nonlinear_clean.json
python3 -m experiments.validation.ex_0_4_mlp --config experiments/validation/configs/mlp_xor_2_2_1.json
python3 -m experiments.validation.ex_0_4_mlp --config experiments/validation/configs/mlp_xor_2_3_2_1.json
python3 -m experiments.validation.ex_0_5_core_smoke
python3 -m experiments.validation.ex_0_6_weight_decay
```

Artefactos:

- `ex_0_1` a `ex_0_4` guardan pesos, config, history, evaluation y plots en `results/validation/...`.
- `ex_0_5` y `ex_0_6` no generan carpetas de resultados: solo ejecutan asserts y terminan con un mensaje de éxito si todo está bien.

## Cómo correr el ejercicio 1

```bash
python3 -m experiments.ej1.eda
python3 -m experiments.ej1.train --config experiments/ej1/configs/linear.json
python3 -m experiments.ej1.train --config experiments/ej1/configs/nonlinear_sigmoid_mse.json
python3 -m experiments.ej1.train --config experiments/ej1/configs/nonlinear_sigmoid_bce.json
python3 -m experiments.ej1.compare_learning
python3 -m experiments.ej1.generalization --config experiments/ej1/configs/nonlinear_sigmoid_mse.json
```

Outputs típicos:

- `results/ej1/eda/`
- `results/ej1/training/<run_id>/`
- `results/ej1/comparison/`
- `results/ej1/generalization/`

## Cómo correr el ejercicio 2

Corrida individual:

```bash
python3 -m experiments.ej2.eda
python3 -m experiments.ej2.train --config experiments/ej2/configs/baseline_tanh_sgd.json
```

Sweep oficial base:

```bash
python3 -m experiments.ej2.run_all --group all --seeds 42 123 2026
python3 -m experiments.ej2.compare_experiments --group all --seeds 42 123 2026
python3 -m experiments.ej2.summarize
python3 -m experiments.ej2.evaluate_final
```

También se puede correr por grupo:

```bash
python3 -m experiments.ej2.run_all --group learning_rate --seeds 42 123 2026
python3 -m experiments.ej2.run_all --group architecture --seeds 42 123 2026
python3 -m experiments.ej2.run_all --group optimizer --seeds 42 123 2026
python3 -m experiments.ej2.compare_experiments --group learning_rate --seeds 42 123 2026
python3 -m experiments.ej2.compare_experiments --group architecture --seeds 42 123 2026
python3 -m experiments.ej2.compare_experiments --group optimizer --seeds 42 123 2026
```

`evaluate_final.py` también acepta un run explícito:

```bash
python3 -m experiments.ej2.evaluate_final --run-dir results/ej2/training/<run_id>
```

Outputs típicos:

- `results/ej2/eda/`
- `results/ej2/training/<run_id>/`
- `results/ej2/comparisons/<group>/`
- `results/ej2/summary.csv`
- `results/ej2/summary.md`
- `results/ej2/selection/`
- `results/ej2/final_test/`

Barridos auxiliares extra:

```bash
python3 -m experiments.ej2.experiments.run_all --group all --seeds 42 123 2026
```

Esos scripts no reemplazan el flujo oficial anterior; generan sus propios resultados bajo `results/ej2/comparasion/`.

## Cómo correr el ejercicio 3

Corrida individual:

```bash
python3 -m experiments.ej3.eda
python3 -m experiments.ej3.train --config experiments/ej3/configs/arch_128_64_combined.json
```

Sweep principal:

```bash
python3 -m experiments.ej3.run_all
python3 -m experiments.ej3.summarize
python3 -m experiments.ej3.evaluate_final
```

Si se quiere correr solo algunos configs:

```bash
python3 -m experiments.ej3.run_all --only data_only_more.json opt_momentum_combined.json
```

Y si se quiere repetir una ablación puntual fuera del set default de `run_all.py`:

```bash
python3 -m experiments.ej3.train --config experiments/ej3/configs/ablation_data_only.json
```

`evaluate_final.py` también acepta un run explícito:

```bash
python3 -m experiments.ej3.evaluate_final --run-dir results/ej3/training/<run_id>
```

Outputs típicos:

- `results/ej3/eda/`
- `results/ej3/training/<run_id>/`
- `results/ej3/summary.csv`
- `results/ej3/summary.md`
- `results/ej3/selection/`
- `results/ej3/final_test/`

## Cómo correr la interpretabilidad auxiliar de ej2

Sobre un run ya entrenado de ej2:

```bash
python3 -m experiments.extras.ej2_interpretability.run_analysis \
  --run-dir results/ej2/training/<run_id>
```

Salida por default:

- `results/ej2/interpretability/<run_id>/`

con manifest, summary, reporte markdown, figuras y casos individuales.
