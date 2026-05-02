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
- `results/ej2/selection/`: resumen y candidato seleccionado usando solo validation.
- `results/ej2/final_test/`: evaluacion final en `digits_test.csv`.

## Comandos previstos

Estos comandos documentan la interfaz esperada por las siguientes tareas. Algunos modulos
se agregaran en tareas posteriores.

```bash
python3 -m experiments.ej2.eda
python3 -m experiments.ej2.train --config experiments/ej2/configs/baseline_tanh_sgd.json
python3 -m experiments.ej2.summarize
python3 -m experiments.ej2.evaluate_final --run-dir results/ej2/training/<run_id>
```

## Estructura inicial

```text
experiments/ej2/
├── __init__.py
├── README.md
└── configs/
```
