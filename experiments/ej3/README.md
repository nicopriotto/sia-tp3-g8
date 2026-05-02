# Ejercicio 3 - Clasificacion de digitos manuscritos con datos extendidos

Este paquete contiene los scripts y configuraciones para el Ejercicio 3 del
TP3. La empresa CompanyX volvio a contactar al equipo porque la primera
iteracion (Ejercicio 2) no fue satisfactoria: pidieron alcanzar **accuracy >= 98%** en `digits_test.csv`. Para lograrlo, ahora disponen de un dataset
adicional, `more_digits.csv`, que se debe combinar con el original.

El codigo del experimento se mantiene separado de la libreria core. Toda
construccion de modelos reutiliza `perceptron/`, en particular
`perceptron.models.factory.build_model`; no se reimplementa MLP dentro de
`experiments/ej3/`.

## Datos

Tres archivos involucrados, todos con formato `label,image` (28x28 pixeles
ya en `[0, 1]`):

| Archivo | Filas | Clases presentes | Comentarios |
|---|---:|---|---|
| `data/digits.csv` | 12,449 | 0,1,2,3,4,5,6,7,9 | **Clase `8` ausente.** Clase `5` muy subrepresentada (~271 vs ~1500 del resto). Es el dataset original del Ejercicio 2. |
| `data/more_digits.csv` | 15,741 | 0..9 | Datos nuevos. **Incluye la clase `8`** con 585 muestras y refuerza la `5` con 542. Resto de clases entre 1700 y 2000. |
| `data/digits_test.csv` | 2,497 | 0..9 | Balanceado en las 10 clases (incluye 8). **Solo se usa en `evaluate_final.py`** (una sola vez), como simulacion de produccion. |

Aclaracion: el archivo que el PDF llama `more_data_digits.csv` es
`data/more_digits.csv` en este repo (mismo formato y contenido).

### Implicancia estructural

El `digits.csv` original no contiene la clase 8, por lo que cualquier modelo
de Ejercicio 2 entrenado solo con ese dataset esta topado por construccion
en `accuracy ≈ (samples_test_no_clase_8) / total_samples_test ≈ 0.9027`
sobre `digits_test.csv`. Esto es **un factor estructural** que va a explicar
parte importante del salto de accuracy en Ejercicio 3 — clave para
responder la pregunta (c) del enunciado.

## Estrategia de datos

Se comparan **dos variantes** controladas via config (`extra.data_strategy`):

- **`only_more`**: entrenar solo con `more_digits.csv`. Util para aislar
  cuanto aporta el dataset nuevo por si solo.
- **`combined`**: concatenar `digits.csv + more_digits.csv` (28,190 filas).
  Maximiza volumen y cubre las 10 clases.

El split train/validation se hace internamente sobre la fuente elegida con
estratificacion. `digits_test.csv` nunca participa de la seleccion de
hiperparametros.

## Resultados

Los scripts crean sus salidas bajo `results/ej3/`:

- `results/ej3/eda/`: exploracion comparativa de `digits.csv` vs
  `more_digits.csv`.
- `results/ej3/training/<run_id>/`: pesos, config, historia, metricas y
  plots de cada corrida.
- `results/ej3/ablation/`: comparacion del ganador de ej2 entrenado sobre
  los datos combinados — cuantifica la mejora atribuible solo a los datos.
- `results/ej3/selection/`: resumen y candidato seleccionado usando solo
  validation.
- `results/ej3/final_test/`: evaluacion final en `digits_test.csv`.
- `results/ej3/report/`: reporte de cierre que responde las preguntas
  (a)/(b)/(c) del enunciado.

## Estructura prevista del paquete

```text
experiments/ej3/
├── __init__.py
├── README.md
├── configs/             ← un JSON por experimento (sweep + ablation)
├── eda.py               ← T02
├── data_pipeline.py     ← T07 (multi-source: only_more / combined)
├── train.py             ← T08
├── evaluation.py        ← T08 (probablemente compartido con ej2 via _common/)
├── plots.py             ← T08 (idem)
├── run_all.py           ← T11 (corre todos los configs del sweep secuencialmente)
├── summarize.py         ← T11 (consolida training runs + selecciona)
└── evaluate_final.py    ← T12 (UNICA carga de digits_test.csv)
```

## Comandos previstos

```bash
python3 -m experiments.ej3.eda
python3 -m experiments.ej3.train --config experiments/ej3/configs/<config>.json
python3 -m experiments.ej3.run_all       # corre todo el sweep (skipea fallos, log en results/ej3/training/run_all.log)
python3 -m experiments.ej3.summarize
python3 -m experiments.ej3.evaluate_final
```

## Configs

Los configs en `experiments/ej3/configs/` cubren cinco dimensiones del sweep
(fuente de datos, arquitectura, optimizador/LR, regularizacion, inicializacion).
Salvo `data_only_more.json`, todos usan `extra.data_strategy = "combined"`.

Grupo 1 - Fuente de datos (ablations limpias)

- `ablation_data_only.json`: baseline ej2 (`[784, 64, 10]`, tanh, sgd lr=0.01) entrenado sobre datos combinados; mide la mejora atribuible solo a sumar `more_digits.csv`.
- `data_only_more.json`: misma arquitectura del baseline pero entrenando solo con `more_digits.csv` (`data_strategy = only_more`); aisla el aporte del dataset nuevo por si solo.

Grupo 2 - Arquitectura

- `arch_128_64_combined.json`: MLP `[784, 128, 64, 10]`, tanh + softmax, adam lr=0.001, batch=64, 50 epochs con early stopping (patience=10).
- `arch_256_128_combined.json`: MLP `[784, 256, 128, 10]`, tanh + softmax, mismas hiperparametros que el anterior; mide el efecto de mayor capacidad.
- `arch_relu_128_combined.json`: MLP `[784, 128, 10]` con activation ReLU, weight_init=he, adam lr=0.001; compara ReLU vs tanh.

Grupo 3 - Optimizador y learning rate

- `opt_adam_lr_decay_combined.json`: adam lr=0.005 con `lr_schedule = exponential` (gamma=0.95), 60 epochs; evalua decaimiento de LR sobre adam.
- `opt_momentum_combined.json`: momentum=0.9, lr=0.05, 80 epochs; punto de comparacion no-adaptativo con lr alto.
- `opt_adam_cosine_combined.json`: adam lr=0.001 con `lr_schedule = cosine` (T_max=60); annealing suave hasta `lr_min=0`.

Grupo 4 - Regularizacion (L2 weight decay dentro de `optimizer_params`)

- `reg_l2_low_combined.json`: adam lr=0.001 con `weight_decay=1e-4`; regularizacion suave para combatir overfitting moderado.
- `reg_l2_high_combined.json`: idem pero `weight_decay=1e-3`; regularizacion mas agresiva.

Grupo 5 - Inicializacion de pesos

- `init_he_relu_combined.json`: arquitectura `[784, 256, 128, 10]` con activation ReLU y `weight_init = he`; combinacion teoricamente optima para ReLU.
- `init_xavier_tanh_combined.json`: misma arquitectura con activation tanh y `weight_init = xavier`; combinacion teoricamente optima para tanh.

## Aclaraciones

- **`digits_test.csv` se toca exactamente una vez**, en `evaluate_final.py`.
  Ningun otro modulo de este paquete debe cargarlo. Esto es un invariante
  duro: si un agente ve un `import` o lectura de ese archivo fuera de
  `evaluate_final.py`, es un bug.
- Todas las decisiones de arquitectura, learning rate, optimizador,
  regularizacion, etc. se toman mirando solo `train`/`validation` del
  dataset combinado.
- Los opcionales del enunciado (robustez al ruido, interpretabilidad) estan
  fuera del alcance de esta iteracion, igual que en ej1 y ej2.
