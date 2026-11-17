"""Calendario L–V, disponibilidad explícita y objetivo sin rellenado.

Los retardos son supuestos conservadores identificados, no fechas históricas
de publicación. Este módulo no transforma un snapshot revisado en point-in-time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

TIMEZONE = "Europe/Madrid"
HORIZONS = (5, 63, 252)


def decision_times(dates: Sequence, hour: int = 8) -> pd.DatetimeIndex:
    """Instantes de decisión en UTC; el calendario de fechas es local a Madrid."""
    if not isinstance(hour, int) or not 0 <= hour <= 23:
        raise ValueError("La hora de decisión debe estar entre 0 y 23.")
    local = pd.DatetimeIndex(dates).normalize() + pd.Timedelta(hours=hour)
    if local.tz is not None:
        raise ValueError("Las fechas del calendario deben carecer de zona horaria.")
    return local.tz_localize(TIMEZONE).tz_convert("UTC").as_unit("ns")


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Un instante de disponibilidad/corte debe incluir zona horaria.")
    return timestamp.tz_convert("UTC")


def apply_availability(raw: pd.DataFrame, policies: Mapping) -> pd.DataFrame:
    """Asigna disponibilidad observada o supuesta y conserva valores originales.

    Políticas por variable: enabled, lag_business_days, max_age_business_days.
    Un available_at explícito debe ser timezone-aware. Nunca se interpreta una
    fecha sin hora como una publicación oficial. Variables deshabilitadas salen.
    """
    required = {"observation_date", "variable", "value", "source"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(raw.columns))}")
    data = raw.copy().reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(data["observation_date"], errors="raise"))
    if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
        raise ValueError("observation_date debe contener fechas locales sin hora ni zona.")
    data["observation_date"] = dates.as_unit("ns")
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if np.isinf(data["value"].to_numpy(dtype=float)).any():
        raise ValueError("No se admiten valores infinitos.")
    if data.duplicated(["variable", "observation_date"]).any():
        raise ValueError("Observaciones duplicadas por variable y fecha.")
    if set(data["variable"]) - set(policies):
        raise ValueError("Falta una política explícita para alguna variable.")
    frames = []
    for variable, group in data.groupby("variable", sort=True):
        policy = policies[variable]
        if not policy.get("enabled", True):
            continue
        lag, max_age = policy["lag_business_days"], policy["max_age_business_days"]
        if type(lag) is not int or lag < 1 or type(max_age) is not int or max_age < lag:
            raise ValueError("Retardo entero >=1 y antigüedad máxima >= retardo requeridos.")
        group = group.copy()
        provided = group.get("available_at", pd.Series(pd.NaT, index=group.index))
        assumed = provided.isna()
        available = pd.Series(pd.NaT, index=group.index, dtype="datetime64[ns, UTC]")
        bases = pd.Series("assumed_business_lag", index=group.index)
        if assumed.any():
            dates = pd.DatetimeIndex(group.loc[assumed, "observation_date"]) + pd.offsets.BDay(lag)
            available.loc[assumed] = decision_times(dates)
        # Las publicaciones explícitas suelen ser pocas; su procedencia se
        # valida individualmente sin iterar las decenas de miles de supuestos.
        for index, row in group.loc[~assumed].iterrows():
            instant = _utc(provided.loc[index])
            if instant.tz_convert(TIMEZONE).date() < row["observation_date"].date():
                raise ValueError("Disponibilidad anterior a la observación.")
            basis = row.get("availability_basis")
            if not isinstance(basis, str) or not basis or basis == "assumed_business_lag":
                raise ValueError("La disponibilidad explícita necesita procedencia verificable.")
            available.loc[index] = instant
            bases.loc[index] = basis
        group["available_at"] = available
        group["availability_basis"] = bases
        group["max_age_business_days"] = max_age
        frames.append(group)
    if not frames:
        raise ValueError("No hay variables habilitadas.")
    return pd.concat(frames, ignore_index=True).sort_values(["variable", "observation_date"])


def build_features(raw: pd.DataFrame, start: str, end: str, decision_hour: int = 8) -> pd.DataFrame:
    """Construye entradas as-of; los valores demasiado antiguos quedan nulos.

    is_observed_today significa observación original fechada en la decisión;
    is_carried significa valor de una observación anterior, no un dato original
    nuevo. Repetir el mismo precio nunca permite inferir estas máscaras.
    """
    dates = pd.bdate_range(start, end).as_unit("ns")
    output = pd.DataFrame({"decision_date": dates, "decision_at": decision_times(dates, decision_hour)})
    blocks = [output]
    for variable, group in raw.groupby("variable", sort=True):
        group = group.loc[group["value"].notna()].sort_values(["available_at", "observation_date"])
        group = group.drop_duplicates("available_at", keep="last")
        # Una publicación tardía de una observación antigua no hace retroceder
        # la observación más reciente ya conocida.
        group = group.loc[group["observation_date"].ge(group["observation_date"].cummax())]
        columns = ["available_at", "observation_date", "value", "availability_basis"]
        if group.empty:
            joined = pd.DataFrame(index=output.index, columns=columns)
            joined["available_at"] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
            joined["observation_date"] = pd.NaT
        else:
            joined = pd.merge_asof(output[["decision_at"]], group[columns],
                                   left_on="decision_at", right_on="available_at", direction="backward")
        observed_date = pd.to_datetime(joined["observation_date"])
        known = observed_date.notna()
        if (observed_date[known].to_numpy() > dates[known].to_numpy()).any():
            raise ValueError("Una entrada contiene una observación futura.")
        ages = np.full(len(dates), np.nan)
        ages[known] = np.busday_count(observed_date[known].to_numpy(dtype="datetime64[D]"),
                                     dates[known].to_numpy(dtype="datetime64[D]"))
        max_age = int(raw.loc[raw["variable"].eq(variable), "max_age_business_days"].iloc[0])
        stale = known & (ages > max_age)
        usable = known & ~stale
        blocks.append(pd.DataFrame({
            variable: pd.to_numeric(joined["value"], errors="raise").where(usable),
            f"{variable}__observation_date": observed_date,
            f"{variable}__available_at": joined["available_at"],
            f"{variable}__availability_basis": joined["availability_basis"],
            f"{variable}__age_business_days": ages,
            f"{variable}__is_available": usable,
            f"{variable}__is_stale": stale,
            f"{variable}__is_observed_today": usable & observed_date.eq(output["decision_date"]),
            f"{variable}__is_carried": usable & observed_date.lt(output["decision_date"]),
        }))
    return pd.concat(blocks, axis=1)


def build_target(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Brent original exacto en cada fecha L–V; nunca rellena una etiqueta."""
    brent = raw.loc[raw["variable"].eq("brent"),
                    ["observation_date", "value", "available_at", "availability_basis"]]
    if brent.duplicated("observation_date").any():
        raise ValueError("Fechas de Brent duplicadas.")
    dates = pd.DataFrame({"observation_date": pd.bdate_range(start, end).as_unit("ns")})
    result = dates.merge(brent, on="observation_date", how="left", validate="one_to_one")
    result = result.rename(columns={"value": "y_brent", "available_at": "label_available_at"})
    result["observed"] = result["y_brent"].notna()
    result.loc[~result["observed"], "label_available_at"] = pd.NaT
    return result


def eligible_training_labels(targets: pd.DataFrame, cutoff: object) -> pd.Series:
    """Máscara por etiqueta; un ejemplo multihorizonte debe cumplirlas TODAS."""
    available = pd.to_datetime(targets["label_available_at"], utc=True)
    return targets["observed"] & targets["y_brent"].notna() & available.notna() & available.le(_utc(cutoff))


def target_dates(origin: object, horizons: Sequence[int] = HORIZONS) -> dict[int, pd.Timestamp]:
    """Fechas futuras por pasos L–V, sin eliminar antes los festivos sin precio."""
    date = pd.Timestamp(origin)
    if date.tzinfo is not None or date != date.normalize() or date.day_of_week > 4:
        raise ValueError("El origen debe ser una fecha L–V sin hora ni zona.")
    if any(type(h) is not int or h <= 0 for h in horizons):
        raise ValueError("Los horizontes deben ser enteros positivos.")
    return {h: date + pd.offsets.BDay(h) for h in horizons}


def evaluation_schedule(config: Mapping) -> pd.DataFrame:
    """Plan congelado de reajustes semestrales expansivos; no entrena modelos."""
    evaluation = config["evaluation"]
    months = evaluation.get("refit_months", [1, 7])
    if months != [1, 7]:
        raise ValueError("Este protocolo implementa reajuste semestral en enero y julio.")
    result = []
    blocks = [("development", "development_origin_years"),
              ("calibration", "calibration_origin_years"), ("test", "test_origin_years")]
    for phase, key in blocks:
        for year in evaluation[key]:
            for month in months:
                begin = pd.Timestamp(year=year, month=month, day=1)
                finish = pd.Timestamp(year=year + (month == 7), month=7 if month == 1 else 1, day=1)
                dates = pd.bdate_range(begin, finish - pd.Timedelta(days=1))
                first = dates[0]
                result.append({"phase": phase, "refit_at": decision_times([first])[0],
                               "training_cutoff": decision_times([first - pd.offsets.BDay(1)])[0],
                               "origin_start": first, "origin_end": dates[-1]})
    return pd.DataFrame(result).sort_values(["phase", "origin_start"]).reset_index(drop=True)
