"""Model factory from experiment configuration."""
from __future__ import annotations

from ..activations import get_activation
from ..config import ExperimentConfig
from .mlp import MLPPerceptron
from .simple import SimplePerceptron


def build_model(config: ExperimentConfig, n_features: int):
    activation = get_activation(config.activation, **config.activation_params)

    if config.model_type == "mlp":
        if not config.architecture:
            raise ValueError("config.architecture is required for model_type='mlp'.")
        output_activation = (
            get_activation(config.output_activation, **config.output_activation_params)
            if config.output_activation
            else None
        )
        return MLPPerceptron(
            n_features=n_features,
            architecture=config.architecture,
            activation=activation,
            output_activation=output_activation,
            seed=config.seed,
            batch_size=config.batch_size,
            loss=config.loss,
        )

    if config.model_type in {"step", "linear", "nonlinear"}:
        return SimplePerceptron(
            n_features=n_features,
            activation=activation,
            seed=config.seed,
            loss=config.loss,
        )

    raise ValueError("Unknown model_type '{}'. Available: step, linear, nonlinear, mlp".format(config.model_type))
