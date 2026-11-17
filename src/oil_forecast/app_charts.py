"""Interactive, original-unit charts for the frozen forecasting application.

This module only presents supplied observations and model outputs. It neither
fits models nor fills missing prices, creates intermediate forecasts or chooses
an architecture from test results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from html import escape
import textwrap
import plotly.graph_objects as go
from plotly.subplots import make_subplots


NAVY = "#081321"
CARD = "#102333"
GOLD = "#D9B970"
TEAL = "#4ED3C1"
WHITE = "#F1F5F9"
MUTED = "#B1C3D6"
CORAL = "#EB9C90"
COLORS = {"persistence": "#A9B8CA", "RNN": TEAL, "LSTM": GOLD, "GRU": "#7CA7E5"}
ORDER = ("persistence", "RNN", "LSTM", "GRU")


def _series_color(rows: pd.DataFrame) -> str:
    if "model" in rows and rows["model"].nunique() == 1:
        return COLORS.get(str(rows["model"].iloc[0]), GOLD)
    return GOLD


def _tint(color: str, opacity: float) -> str:
    return f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{opacity})"


def _model_label(model: str, variant: str | None = None) -> str:
    if str(model).lower() in {"persistence", "persistencia"}:
        return "Persistencia"
    return f"{model}-{variant}" if variant and variant != "baseline" else str(model)


def _base(figure: go.Figure | None = None, *, height: int = 390,
          ylabel: str = "USD/barril") -> go.Figure:
    figure = go.Figure() if figure is None else figure
    figure.update_layout(
        height=height, autosize=True, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", colorway=list(COLORS.values()),
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": WHITE, "size": 13},
        margin={"l": 14, "r": 20, "t": 30, "b": 110},
        hovermode="x unified", hoverlabel={"bgcolor": CARD, "font_color": WHITE},
        legend={"orientation": "h", "x": 0, "y": -.20, "xanchor": "left", "yanchor": "top",
                "entrywidthmode": "fraction", "entrywidth": .5,
                "font": {"size": 12}, "bgcolor": "rgba(0,0,0,0)"},
    )
    figure.update_xaxes(showgrid=False, zeroline=False, color=MUTED,
                        showline=False, automargin=True)
    figure.update_yaxes(title_text=ylabel, gridcolor="rgba(145,169,193,0.12)",
                        zeroline=False, color=MUTED, automargin=True)
    return figure


def _empty(message: str, *, height: int = 330) -> go.Figure:
    figure = _base(height=height)
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font={"color": MUTED, "size": 14})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _observed(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["observation_date", "y_brent"])
    if not {"observation_date", "y_brent"}.issubset(history):
        raise ValueError("Brent history needs observation_date and y_brent")
    rows = history.copy()
    rows["observation_date"] = pd.to_datetime(rows["observation_date"], errors="raise")
    rows["y_brent"] = pd.to_numeric(rows["y_brent"], errors="raise")
    if "observed" in rows:
        rows = rows.loc[rows["observed"].isin([True, "True", "true", 1])]
    rows = rows.loc[rows["y_brent"].notna()].sort_values("observation_date")
    if not np.isfinite(rows["y_brent"]).all() or rows["observation_date"].isna().any():
        raise ValueError("Observed Brent prices and dates must be finite")
    return rows


def history_chart(history: pd.DataFrame, context_events: bool = True, events: pd.DataFrame | None = None) -> go.Figure:
    """Plot actual observations, with event markers on actual local extrema."""
    rows = _observed(history)
    if rows.empty:
        return _empty("No hay observaciones Brent disponibles para este periodo")
    figure = _base(height=375)
    figure.update_layout(margin_b=75)
    figure.add_trace(go.Scatter(x=rows["observation_date"], y=rows["y_brent"],
        name="Brent observado", mode="lines", line={"color": GOLD, "width": 2},
        hovertemplate="%{x|%d/%m/%Y}<br>Brent: $%{y:.2f}/barril<extra></extra>"))
    if context_events and events is not None and not events.empty:
        markers = events.loc[pd.to_datetime(events["observation_date"]).between(rows["observation_date"].min(), rows["observation_date"].max())]
        if not markers.empty:
            detail = [[item["marker_basis"], "<br>".join(escape(line) for line in textwrap.wrap(item["documented_fact"], 54)),
                       escape(item["sources"][0]["title"])] for item in markers.to_dict("records")]
            figure.add_trace(go.Scatter(x=markers["observation_date"], y=markers["price"],
                text=markers["title"], customdata=detail, name="Acontecimientos", mode="markers",
                marker={"color": TEAL, "size": 8, "symbol": "diamond", "line": {"width": 1, "color": NAVY}},
                hovertemplate="%{text}<br>%{x|%d/%m/%Y} · $%{y:.2f}/barril<br>%{customdata[0]}<br>%{customdata[1]}<br>Fuente: %{customdata[2]}<extra></extra>"))
    figure.update_xaxes(type="date", rangeslider_visible=False)
    return figure


def _interval(row: dict, point: float, lower: str = "lower", upper: str = "upper") -> tuple[float, float] | None:
    lo, hi = row.get(lower), row.get(upper)
    present = [value is not None and pd.notna(value) for value in (lo, hi)]
    if not any(present):
        return None
    if not all(present) or not np.isfinite([lo, point, hi]).all() or not lo <= point <= hi:
        raise ValueError("Forecast interval endpoints are incomplete or incoherent")
    return float(lo), float(hi)


def forecast_chart(history: pd.DataFrame, forecast: dict, horizon: int) -> go.Figure:
    """Show one direct future output and its band, never a daily forecast path."""
    matches = [row for row in forecast["forecasts"] if int(row["horizon"]) == int(horizon)]
    if len(matches) != 1:
        raise ValueError("A forecast chart needs exactly one direct output for the selected horizon")
    result = matches[0]
    origin = pd.Timestamp(forecast["origin_date"])
    target = pd.Timestamp(result["target_date"])
    price = float(result["price"])
    if pd.isna(target) or pd.isna(origin) or target <= origin or not np.isfinite(price):
        raise ValueError("Forecast target and price are incoherent")
    interval = _interval(result, price)
    rows = _observed(history)
    rows = rows.loc[rows["observation_date"].between(origin - pd.Timedelta(days=150), origin)]
    figure = _base(height=430)
    figure.add_trace(go.Scatter(x=rows["observation_date"], y=rows["y_brent"], name="Observado · contexto",
        mode="lines", line={"color": MUTED, "width": 1.8},
        hovertemplate="%{x|%d/%m/%Y}<br>Observado: $%{y:.2f}<br>Contexto · puede diferir de la referencia<extra></extra>"))
    model = str(forecast["model"])
    model_color = COLORS.get(model, GOLD)
    if model.lower() not in {"persistence", "persistencia"} and pd.notna(result.get("persistence_price")):
        baseline = float(result["persistence_price"])
        baseline_interval = _interval(result, baseline, "persistence_lower", "persistence_upper")
        baseline_details = {} if baseline_interval is None else {"customdata": [list(baseline_interval)], "error_y": {
            "type": "data", "symmetric": False, "array": [baseline_interval[1] - baseline],
            "arrayminus": [baseline - baseline_interval[0]],
            "color": COLORS["persistence"], "thickness": 1, "width": 5}}
        baseline_hover = "%{x|%d/%m/%Y}<br>Persistencia: $%{y:.2f}"
        if baseline_interval is not None:
            baseline_hover += "<br>Intervalo 80 %: $%{customdata[0]:.2f}–$%{customdata[1]:.2f}"
        figure.add_trace(go.Scatter(x=[target], y=[baseline], name="Persistencia",
            mode="markers", marker={"size": 11, "symbol": "circle-open", "color": COLORS["persistence"]},
            hovertemplate=baseline_hover + "<extra></extra>", **baseline_details))
    details = {} if interval is None else {"customdata": [[interval[0], interval[1]]], "error_y": {
        "type": "data", "symmetric": False, "array": [interval[1] - price],
        "arrayminus": [price - interval[0]], "color": model_color, "thickness": 3, "width": 10}}
    hover = "%{x|%d/%m/%Y}<br>Estimación: $%{y:.2f}"
    if interval is not None:
        hover += "<br>Intervalo: $%{customdata[0]:.2f}–$%{customdata[1]:.2f}"
    figure.add_trace(go.Scatter(x=[target], y=[price],
        name=_model_label(model, forecast.get("variant")) + (" · 80 %" if interval else ""), mode="markers",
        marker={"size": 14, "symbol": "diamond", "color": model_color, "line": {"color": NAVY, "width": 1}},
        hovertemplate=hover + "<extra></extra>", **details))
    reference_date = pd.Timestamp(forecast.get("reference_observation_date", origin))
    if pd.notna(reference_date):
        figure.add_trace(go.Scatter(x=[reference_date], y=[forecast["reference_brent"]],
            name="Referencia", mode="markers",
            marker={"color": WHITE, "size": 9, "symbol": "square-open", "line": {"width": 2}},
            hovertemplate="%{x|%d/%m/%Y}<br>Referencia: $%{y:.2f}<extra></extra>"))
    figure.add_shape(type="line", x0=origin, x1=origin, y0=0, y1=1, yref="paper",
                      line={"color": GOLD, "width": 1.4, "dash": "dot"})
    figure.add_annotation(x=origin, y=1, yref="paper", text="Origen", showarrow=False,
                          xanchor="left", xshift=7, yshift=8, font={"size": 12, "color": GOLD})
    figure.update_xaxes(type="date")
    return figure


def metrics_chart(aggregated: pd.DataFrame, metric: str = "mae") -> go.Figure:
    """Compare one horizon, keeping seed dispersion distinct from uncertainty."""
    if aggregated.empty:
        return _empty("No hay métricas disponibles para esta selección")
    if metric not in aggregated or "model" not in aggregated:
        raise ValueError("The chosen metric and model identifiers are required")
    rows = aggregated.copy()
    if rows["model"].duplicated().any():
        raise ValueError("Select one horizon and one aggregate per model before plotting metrics")
    rows["_order"] = rows["model"].map({model: index for index, model in enumerate(ORDER)}).fillna(9)
    rows = rows.sort_values("_order")
    labels = [_model_label(str(row["model"]), row.get("variant")) for row in rows.to_dict(orient="records")]
    values = pd.to_numeric(rows[metric], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Chart metrics must be finite")
    deviation = rows.get(f"{metric}_std", pd.Series(0., index=rows.index)).fillna(0).to_numpy(dtype=float)
    unit = {"mse": "USD²/barril²", "r2": "R² · sin unidad", "parameters": "Parámetros entrenables",
            "training_seconds": "Segundos de entrenamiento", "skill_mae_pct": "Mejora MAE (%)"}.get(metric, "USD/barril")
    figure = _base(height=345, ylabel=unit)
    figure.update_layout(margin_b=60)
    best = values.max() if metric in {"skill_mae_pct", "r2"} else values.min()
    winners = np.isclose(values, best, rtol=1e-12, atol=1e-12)
    figure.add_trace(go.Bar(x=labels, y=values, marker_color=[COLORS.get(model, TEAL) for model in rows["model"]],
        marker_line_width=[2 if winner else 0 for winner in winners],
        marker_line_color=[WHITE if winner else NAVY for winner in winners],
        width=.58, showlegend=False,
        error_y={"type": "data", "array": deviation, "color": MUTED, "thickness": 1.4, "width": 5},
        customdata=np.column_stack([deviation, rows.get("n_seeds", pd.Series(1, index=rows.index))]),
        hovertemplate="%{x}<br>Valor: %{y:.3f}<br>Desv. entre semillas: %{customdata[0]:.3f}<br>Semillas: %{customdata[1]:.0f}<extra></extra>"))
    padding = max(float(np.max(np.abs(values) + deviation)) * .035, 1e-6)
    for label, value, spread, winner in zip(labels, values, deviation, winners):
        # Place values above the complete error bar, rather than over its cap.
        figure.add_annotation(x=label, y=float(value + spread + padding),
            text=f"<b>{value:,.2f}</b>" if winner else f"{value:,.2f}",
            showarrow=False, yanchor="bottom", font={"size": 13, "color": WHITE})
    figure.update_yaxes(rangemode="tozero")
    if metric == "r2":
        figure.add_hline(y=0, line_dash="dot", line_color=MUTED)
    if (values >= 0).all():
        figure.update_yaxes(range=[0, max(float((values + deviation).max()) * 1.25, 1e-6)])
    return figure


def loss_chart(history: pd.DataFrame, model: str, variant: str, seed: int, block: str) -> go.Figure:
    """Draw one actual run; an unavailable validation series is never invented."""
    required = {"model", "variant", "seed", "block_id", "epoch", "train_loss"}
    if not required.issubset(history):
        return _empty("Este experimento no incluye curvas de pérdida")
    rows = history.loc[history["model"].eq(model) & history["variant"].eq(variant)
                       & history["seed"].eq(seed) & history["block_id"].eq(block)].sort_values("epoch")
    if rows.empty:
        return _empty("No hay una curva guardada para esta ejecución")
    if rows["epoch"].duplicated().any():
        raise ValueError("Select a single experiment before plotting its loss")
    figure = _base(height=330, ylabel="MSE normalizado")
    for name, label, color, dash in [("train_loss", "Entrenamiento", TEAL, "solid"),
                                    ("validation_loss", "Validación", GOLD, "dash")]:
        if name in rows and rows[name].notna().any():
            figure.add_trace(go.Scatter(x=rows["epoch"], y=rows[name], name=label,
                mode="lines+markers", line={"color": color, "width": 2, "dash": dash}, marker_size=5,
                hovertemplate="Época %{x}<br>MSE: %{y:.4f}<extra></extra>"))
    figure.update_xaxes(title_text=None, dtick=1)
    figure.update_yaxes(rangemode="tozero")
    if "validation_loss" not in rows or not rows["validation_loss"].notna().any():
        figure.add_annotation(text="Épocas prefijadas · sin selección con test", x=0, y=1.08,
                              xref="paper", yref="paper", showarrow=False, xanchor="left",
                              font={"color": MUTED, "size": 12})
    return figure


def _prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"target_date", "y_true", "y_pred"}
    if not required.issubset(predictions):
        raise ValueError("Stored predictions require target_date, y_true and y_pred")
    rows = predictions.copy()
    for name in ["model", "variant", "seed", "horizon"]:
        if name in rows and rows[name].nunique() > 1:
            raise ValueError("Select one model, variant, seed and horizon before plotting forecasts")
    rows["target_date"] = pd.to_datetime(rows["target_date"], errors="raise")
    if rows["target_date"].duplicated().any():
        raise ValueError("Stored forecasts must have unique target dates")
    return rows.sort_values("target_date")


def prediction_chart(predictions: pd.DataFrame) -> go.Figure:
    """Display a sequence of saved forecasts aligned to their actual targets."""
    if predictions.empty:
        return _empty("No hay predicciones guardadas para esta selección")
    rows = _prediction_rows(predictions)
    figure = _base(height=360)
    color = _series_color(rows)
    if {"lower", "upper"}.issubset(rows) and rows[["lower", "upper"]].notna().all().all():
        figure.add_trace(go.Scatter(x=rows["target_date"], y=rows["upper"], mode="lines", line_width=0,
                                    hoverinfo="skip", showlegend=False))
        figure.add_trace(go.Scatter(x=rows["target_date"], y=rows["lower"], mode="lines", line_width=0,
            fill="tonexty", fillcolor=_tint(color, .15), name="Intervalo 80 %", hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=rows["target_date"], y=rows["y_true"], name="Observado", mode="lines",
        line={"color": WHITE, "width": 2}, hovertemplate="%{x|%d/%m/%Y}<br>Observado: $%{y:.2f}<extra></extra>"))
    figure.add_trace(go.Scatter(x=rows["target_date"], y=rows["y_pred"], name="Predicción", mode="lines",
        line={"color": color, "width": 2, "dash": "dot"},
        hovertemplate="%{x|%d/%m/%Y}<br>Predicción: $%{y:.2f}<extra></extra>"))
    figure.update_xaxes(type="date", title_text=None)
    return figure


def errors_chart(predictions: pd.DataFrame) -> go.Figure:
    """Signed original-unit error, from saved prices rather than rounded KPIs."""
    if predictions.empty:
        return _empty("No hay errores guardados para esta selección")
    rows = _prediction_rows(predictions)
    error = rows["y_pred"] - rows["y_true"]
    color = _series_color(rows)
    figure = _base(height=300, ylabel="Error · USD/barril")
    figure.update_layout(margin_b=65, showlegend=False)
    figure.add_trace(go.Scatter(x=rows["target_date"], y=error, name="Error de predicción", mode="lines",
        line={"color": color, "width": 1.8}, fill="tozeroy", fillcolor=_tint(color, .10),
        hovertemplate="%{x|%d/%m/%Y}<br>Predicho − observado: $%{y:.2f}<extra></extra>"))
    figure.add_hline(y=0, line={"color": MUTED, "width": 1, "dash": "dot"})
    figure.update_xaxes(type="date", title_text="Fecha objetivo")
    return figure


def coverage_chart(aggregated: pd.DataFrame) -> go.Figure:
    """Show empirical coverage and width in separate panels with explicit units."""
    if aggregated.empty:
        return _empty("No hay intervalos evaluados para esta selección")
    rows = aggregated.copy()
    column = "coverage_pct" if "coverage_pct" in rows else "coverage"
    if not {"model", column, "mean_width"}.issubset(rows):
        raise ValueError("Interval charts require model, empirical coverage and mean_width")
    coverage = rows[column].to_numpy(dtype=float) * (1 if column == "coverage_pct" else 100)
    width = rows["mean_width"].to_numpy(dtype=float)
    if not np.isfinite(np.r_[coverage, width]).all() or (coverage < 0).any() or (coverage > 100).any() or (width < 0).any():
        raise ValueError("Interval coverage and widths must be finite and coherent")
    labels = [_model_label(str(row["model"]), row.get("variant"))
              + (f"<br>{int(row['horizon'])} días" if rows["model"].duplicated().any() and "horizon" in row else "")
              for row in rows.to_dict(orient="records")]
    colors = [COLORS.get(model, TEAL) for model in rows["model"]]
    figure = make_subplots(rows=1, cols=2, horizontal_spacing=.15,
                          subplot_titles=("Cobertura empírica", "Amplitud media"))
    figure.add_trace(go.Bar(x=labels, y=coverage, marker_color=colors, showlegend=False,
        text=[f"{value:.1f}%" for value in coverage], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>Cobertura: %{y:.2f}%<extra></extra>"), row=1, col=1)
    figure.add_trace(go.Bar(x=labels, y=width, marker_color=colors, showlegend=False,
        text=[f"${value:.2f}" for value in width], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>Amplitud: $%{y:.2f}/barril<extra></extra>"), row=1, col=2)
    figure = _base(figure, height=340)
    figure.update_layout(margin_b=65, margin_t=45)
    figure.update_annotations(font_size=14)
    figure.update_xaxes(tickangle=-30, tickfont_size=12)
    figure.add_hline(y=80, row=1, col=1, line={"color": GOLD, "width": 1, "dash": "dot"})
    figure.update_yaxes(title_text="Cobertura (%) · nominal 80 %", range=[0, 112], row=1, col=1)
    figure.update_yaxes(title_text="USD/barril", range=[0, max(float(width.max()) * 1.25, 1)], row=1, col=2)
    return figure
