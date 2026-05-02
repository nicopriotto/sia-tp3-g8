# Ejercicio 2 - Clasificacion de digitos manuscritos

Este paquete contiene los scripts y configuraciones para estudiar el problema de
clasificacion de digitos manuscritos entre `0` y `9` con un perceptron multicapa.

El codigo del experimento debe mantenerse separado de la libreria core. Toda construccion
de modelos debe reutilizar `perceptron/`, en particular `perceptron.models.factory.build_model`;
no se debe implementar un MLP paralelo dentro de `experiments/ej2/`.

## Datos

- `data/digits.csv`: se usa para entrenar, validar y seleccionar hiperparametros.
- `data/digits_test.csv`: se reserva para la evaluacion final de generalizacion, como si
  fueran datos de produccion.
- `data/more_digits.csv`: pertenece al Ejercicio 3 y no se usa en este ejercicio.

El flujo de seleccion debe usar un split interno de `digits.csv` para train/validation.
`digits_test.csv` no se usa para elegir arquitectura, learning rate, optimizador, epocas,
early stopping ni ninguna otra decision de hiperparametros.

## Resultados

Los scripts deben crear sus salidas bajo `results/ej2/` cuando se ejecuten. No se guardan
resultados manuales dentro de este paquete.

Convenciones previstas:

- `results/ej2/eda/`: exploracion inicial del dataset de aprendizaje.
- `results/ej2/training/<run_id>/`: pesos, config, historia, metricas y plots de cada corrida.
- `results/ej2/comparisons/<group>/`: graficos y reportes comparativos por parametro.
- `results/ej2/selection/`: resumen y candidato seleccionado usando solo validation.
- `results/ej2/final_test/`: evaluacion final en `digits_test.csv`.

## Comandos principales

EDA y una corrida individual:

```bash
python3 -m experiments.ej2.eda
python3 -m experiments.ej2.train --config experiments/ej2/configs/baseline_tanh_sgd.json
```

Serie experimental minima del Ejercicio 2, con tres seeds por variante:

```bash
python3 -m experiments.ej2.run_all --group all --seeds 42 123 2026
python3 -m experiments.ej2.compare_experiments --group all --seeds 42 123 2026
```

Tambien se puede correr por grupo:

```bash
python3 -m experiments.ej2.run_all --group learning_rate --seeds 42 123 2026
python3 -m experiments.ej2.run_all --group architecture --seeds 42 123 2026
python3 -m experiments.ej2.run_all --group optimizer --seeds 42 123 2026
python3 -m experiments.ej2.compare_experiments --group learning_rate --seeds 42 123 2026
```

Resumen y evaluacion final:

```bash
python3 -m experiments.ej2.summarize
python3 -m experiments.ej2.evaluate_final --run-dir results/ej2/training/<run_id>
```

## Experimentos comparativos

La serie minima del Ejercicio 2 estudia:

- `learning_rate`: `0.0001`, `0.001`, `0.01`, `0.1`, `1.0`, `3.0`, manteniendo arquitectura `[784, 64, 10]` y optimizador `sgd`.
- `architecture`: `[784, 32, 10]`, `[784, 64, 10]`, `[784, 128, 10]`, `[784, 64, 32, 10]`, manteniendo `learning_rate=0.05` y `sgd`.
- `optimizer`: `sgd@0.05`, `momentum@0.05`, `adam@0.001`, manteniendo arquitectura `[784, 128, 10]`.

Cada comparacion genera:

- `loss_by_epoch.png`
- `accuracy_by_epoch.png`
- `macro_f1_by_epoch.png`
- `final_metrics.png`
- `summary.csv`
- `aggregate_summary.csv`
- `report.md`

Los graficos comparativos arrancan en `epoch=0`, antes de aplicar actualizaciones de
pesos. Usan media entre seeds y banda de error estandar. El reporte explica
como evaluar el sistema, que variantes se probaron y cual fue la mejor alternativa por
`val_macro_f1`.

## Learning rate fijo

Para este ejercicio no se usan schedules ni decays de learning rate. Los scripts de barrido
rechazan cualquier config con `lr_schedule` distinto de `null`, ausente o `"none"`.

## Estructura inicial

```text
experiments/ej2/
├── __init__.py
├── README.md
└── configs/
```
