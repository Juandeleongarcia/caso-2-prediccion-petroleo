"""Transformaciones ajustadas únicamente con las filas de entrenamiento recibidas.

El consumidor debe pasar las filas temporales únicas de las ventanas de train;
estas clases no seleccionan filas ni aprenden durante ``transform``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _numeric(values, *, ndim: int | None = None, features: int | None = None,
             allow_nan: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or (ndim is not None and array.ndim != ndim):
        raise ValueError("Forma de matriz no válida.")
    if features is not None and array.shape[-1] != features:
        raise ValueError("Número de variables distinto al ajustado.")
    if np.isinf(array).any() or (not allow_nan and np.isnan(array).any()):
        raise ValueError("Se requieren valores finitos.")
    return array


def _statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("El ajuste necesita filas y variables de entrenamiento.")
    with np.errstate(over="raise", invalid="raise"):
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    return mean, scale


def _feature_names(values, supplied: Sequence[str] | None, count: int) -> tuple[str, ...]:
    if supplied is None:
        supplied = list(values.columns) if hasattr(values, "columns") else [f"feature_{i}" for i in range(count)]
    names = tuple(supplied)
    if len(names) != count or len(set(names)) != count or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Los nombres de variables deben ser únicos y corresponder a las columnas.")
    if hasattr(values, "columns") and tuple(values.columns) != names:
        raise ValueError("El orden de columnas no coincide con los nombres de variables.")
    return names


class InputTransformer:
    """Mediana train, estandarización y un indicador de ausencia por variable.

    Una columna enteramente ausente en train usa cero, media cero y escala uno;
    se registra ``all_missing_`` y no se inventa una cotización histórica.
    Las entradas demasiado antiguas ya deben llegar como NaN desde el pipeline.
    """

    def fit(self, values_2d, feature_names: Sequence[str] | None = None) -> InputTransformer:
        values = _numeric(values_2d, ndim=2, allow_nan=True)
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("El ajuste necesita filas y variables de entrenamiento.")
        names = _feature_names(values_2d, feature_names, values.shape[1])
        all_missing = np.isnan(values).all(axis=0)
        median = np.zeros(values.shape[1], dtype=np.float64)
        for column in np.flatnonzero(~all_missing):
            observed = values[:, column]
            median[column] = np.median(observed[~np.isnan(observed)])
        filled = np.where(np.isnan(values), median, values)
        mean, scale = _statistics(filled)
        self.feature_names_ = names
        self.median_, self.mean_, self.scale_ = median, mean, scale
        self.all_missing_ = all_missing
        self.n_fit_rows_ = len(values)
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "median_"):
            raise ValueError("El transformador de entradas todavía no está ajustado.")

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        return self.feature_names_ + tuple(f"{name}__missing" for name in self.feature_names_)

    def transform(self, values) -> np.ndarray:
        self._require_fitted()
        array = _numeric(values, features=len(self.feature_names_), allow_nan=True)
        if hasattr(values, "columns") and tuple(values.columns) != self.feature_names_:
            raise ValueError("El orden de columnas no coincide con el entrenamiento.")
        missing = np.isnan(array)
        filled = np.where(missing, self.median_, array)
        scaled = (filled - self.mean_) / self.scale_
        return np.concatenate([scaled, missing.astype(np.float64)], axis=-1)

    def to_dict(self) -> dict:
        self._require_fitted()
        return {"schema_version": 1, "kind": "input_median_standard",
                "feature_names": list(self.feature_names_),
                "output_feature_names": list(self.output_feature_names),
                "median": self.median_.tolist(), "mean": self.mean_.tolist(),
                "scale": self.scale_.tolist(), "all_missing": self.all_missing_.tolist(),
                "n_fit_rows": self.n_fit_rows_, "ddof": 0}

    @classmethod
    def from_dict(cls, state: Mapping) -> InputTransformer:
        if state.get("schema_version") != 1 or state.get("kind") != "input_median_standard":
            raise ValueError("Formato de transformador de entradas no reconocido.")
        result = cls()
        names = tuple(state["feature_names"])
        _feature_names(np.empty((1, len(names))), names, len(names))
        if not names or int(state["n_fit_rows"]) <= 0 or state.get("ddof") != 0:
            raise ValueError("Estado de ajuste no válido.")
        result.feature_names_ = names
        for key in ["median", "mean", "scale"]:
            setattr(result, f"{key}_", _numeric(state[key], ndim=1, features=len(names)).copy())
        flags = state["all_missing"]
        if len(flags) != len(names) or any(type(flag) is not bool for flag in flags) or (result.scale_ <= 0).any():
            raise ValueError("Parámetros de imputación/escalado no válidos.")
        result.all_missing_ = np.asarray(flags, dtype=bool)
        result.n_fit_rows_ = int(state["n_fit_rows"])
        return result


class TargetTransformer:
    """Estandariza los tres cambios en USD/barril usando solo etiquetas train."""

    def fit(self, y_changes_2d) -> TargetTransformer:
        values = _numeric(y_changes_2d, ndim=2, features=3)
        self.mean_, self.scale_ = _statistics(values)
        self.n_fit_rows_ = len(values)
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "mean_"):
            raise ValueError("El transformador de objetivos todavía no está ajustado.")

    def transform(self, y_changes) -> np.ndarray:
        self._require_fitted()
        return (_numeric(y_changes, features=3) - self.mean_) / self.scale_

    def inverse_transform(self, scaled_changes) -> np.ndarray:
        self._require_fitted()
        return _numeric(scaled_changes, features=3) * self.scale_ + self.mean_

    def to_dict(self) -> dict:
        self._require_fitted()
        return {"schema_version": 1, "kind": "target_changes_standard",
                "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
                "n_fit_rows": self.n_fit_rows_, "ddof": 0}

    @classmethod
    def from_dict(cls, state: Mapping) -> TargetTransformer:
        if state.get("schema_version") != 1 or state.get("kind") != "target_changes_standard":
            raise ValueError("Formato de transformador de objetivos no reconocido.")
        result = cls()
        result.mean_ = _numeric(state["mean"], ndim=1, features=3).copy()
        result.scale_ = _numeric(state["scale"], ndim=1, features=3).copy()
        if (result.scale_ <= 0).any() or int(state["n_fit_rows"]) <= 0 or state.get("ddof") != 0:
            raise ValueError("Parámetros de escalado no válidos.")
        result.n_fit_rows_ = int(state["n_fit_rows"])
        return result
