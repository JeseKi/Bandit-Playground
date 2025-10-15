from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING
import math

from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objects as go  # type: ignore

from bandit_lib.utils.viz.constants import _METRIC_LABELS
from bandit_lib.utils.viz.utils import (
    determine_layout,
    compute_convergence_rate_series,
    compute_mean_and_ci,
    hex_to_rgba,
    get_color_from_name,
    z_value_for_confidence,
)


if TYPE_CHECKING:
    from bandit_lib.agents.base import BaseAgent
    from bandit_lib.agents.schemas import Metrics


def _validate_inputs_for_history(
    metrics_history: Sequence["Metrics"],
    agents: Sequence["BaseAgent"],
    metrics_to_plot: List[str],
) -> Tuple[List[str], List[float]]:
    """Validate inputs and normalize metrics list; return steps."""
    if not metrics_history or not agents:
        raise ValueError("metrics_history or agents is empty")
    normalized_metrics = metrics_to_plot or _METRIC_LABELS.copy()
    steps = [metric.current_step for metric in metrics_history]
    return normalized_metrics, steps


def _create_subplots(metrics_to_plot: List[str]) -> Tuple[go.Figure, int, int]:
    """Create subplot figure and return figure with (rows, cols)."""
    rows, cols = determine_layout(len(metrics_to_plot))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=metrics_to_plot,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    return fig, rows, cols


def _compute_raw_values_for_metric(
    metric_key: str,
    steps: List[float],
    metrics_history: Sequence["Metrics"],
    agents: Sequence["BaseAgent"],
) -> List[float]:
    """Compute the y-series for a metric.

    - For "convergence_rate": uses agents to compute a proportion series
    - Otherwise: extracts the metric attribute from metrics_history
    """
    if metric_key == "convergence_rate":
        return compute_convergence_rate_series(steps, agents)
    return [getattr(metric, metric_key) for metric in metrics_history]


def _add_history_series_trace(
    fig: go.Figure,
    row: int,
    col: int,
    steps: List[float],
    raw_values: List[float],
    metric_key: str,
) -> None:
    """Add the main line+markers trace for a single metric."""
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=raw_values,
            mode="lines+markers",
            name=metric_key,
            marker=dict(size=6),
        ),
        row=row,
        col=col,
    )


def _add_credibility_band_for_metric(
    fig: go.Figure,
    row: int,
    col: int,
    metric_key: str,
    steps: List[float],
    raw_values: List[float],
    agents: Sequence["BaseAgent"],
    credibility_confidence: float,
) -> None:
    """Draw statistical credibility band for a metric when enabled.

    - For convergence_rate: use binomial approximation p ± z*sqrt(p(1-p)/n)
    - For others: aggregate agents' metrics time series and use compute_mean_and_ci
    """
    color = get_color_from_name(metric_key)
    if metric_key == "convergence_rate":
        n = len(agents) if agents else 0
        if n <= 0:
            return
        z = z_value_for_confidence(credibility_confidence)
        ci_lower: List[float] = []
        ci_upper: List[float] = []
        for p in raw_values:
            se = math.sqrt(max(0.0, p * (1.0 - p)) / n)
            margin = z * se
            ci_lower.append(max(0.0, p - margin))
            ci_upper.append(min(1.0, p + margin))
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=ci_lower,
                mode="lines",
                line=dict(color=color, width=0),
                name=f"{metric_key} CI lower",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=ci_upper,
                mode="lines",
                line=dict(color=color, width=0),
                fill="tonexty",
                fillcolor=hex_to_rgba(color, 0.18),
                name=f"{metric_key} {int(credibility_confidence * 100)}% CI",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        return

    # Aggregate across agents for non-convergence metrics
    if agents:
        series_list: List[List[float]] = []
        min_len = None
        for ag in agents:
            seq = [getattr(m, metric_key) for m in getattr(ag, "metrics", [])]
            if not seq:
                continue
            if min_len is None:
                min_len = len(seq)
            else:
                min_len = min(min_len, len(seq))
            series_list.append([float(v) for v in seq])
        if series_list and len(series_list) >= 2:
            mean_v, lower_v, upper_v = compute_mean_and_ci(
                [s[:min_len] for s in series_list], confidence=credibility_confidence
            )
            x_steps = steps[: len(mean_v)]
            fig.add_trace(
                go.Scatter(
                    x=x_steps,
                    y=lower_v,
                    mode="lines",
                    line=dict(color=color, width=0),
                    name=f"{metric_key} CI lower",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_steps,
                    y=upper_v,
                    mode="lines",
                    line=dict(color=color, width=0),
                    fill="tonexty",
                    fillcolor=hex_to_rgba(color, 0.18),
                    name=f"{metric_key} {int(credibility_confidence * 100)}% CI",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )


def _add_final_value_hline(
    fig: go.Figure, row: int, col: int, raw_values: List[float]
) -> None:
    """Add a dashed horizontal line at the final non-None value for context."""
    final_value = next(
        (value for value in reversed(raw_values) if value is not None), None
    )
    if final_value is None:
        return
    fig.add_hline(
        y=final_value,
        row=row,  # type: ignore
        col=col,  # type: ignore
        line_dash="dash",
        line_color="red",
        annotation_text=f"Final value: {final_value:.4f}",
        annotation_position="top left",
        annotation_font=dict(size=10),
    )


def _configure_axes(
    fig: go.Figure, row: int, col: int, x_log: bool, metric_key: str
) -> None:
    """Configure axes titles and optional logarithmic scale."""
    xaxis_kwargs: Dict[str, Any] = {"title": "Steps"}
    if x_log:
        xaxis_kwargs["type"] = "log"
    fig.update_xaxes(row=row, col=col, **xaxis_kwargs)
    fig.update_yaxes(row=row, col=col, title=metric_key)


def _apply_layout_history(
    fig: go.Figure, agent_name: str, rows: int, cols: int
) -> None:
    """Apply layout with auto dimensions based on subplot grid."""
    title = f'"{agent_name}" metrics history' if agent_name else "Metrics history"
    fig.update_layout(
        title=title,
        height=max(400, rows * 350),
        width=max(600, cols * 500),
        showlegend=False,
        template="plotly_white",
    )


def _export_figure_history(
    fig: go.Figure, file_name: Path, width: int, height: int, scale: int
) -> Path:
    """Export figure to HTML or image, with HTML fallback when engine missing."""
    output_path = Path(file_name)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".html")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    try:
        if suffix in {".html", ".htm"}:
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), width=width, height=height, scale=scale)
    except ValueError:
        print("Error writing image, falling back to HTML")
        fallback_path = output_path.with_suffix(".html")
        fig.write_html(str(fallback_path))
        output_path = fallback_path
    return output_path


def plot_metrics_history(
    metrics_history: Sequence["Metrics"],
    agent_name: str,
    file_name: Path,
    agents: Sequence["BaseAgent"],
    x_log: bool = False,
    metrics_to_plot: List[str] = _METRIC_LABELS,
    width: int = 1500,
    height: int = 1500,
    scale: int = 2,
    enable_statistical_credibility: bool = False,
    credibility_confidence: float = 0.95,
) -> go.Figure:
    """Plot the metrics history with plotly."""
    metrics_to_plot, steps = _validate_inputs_for_history(
        metrics_history, agents, metrics_to_plot
    )
    fig, rows, cols = _create_subplots(metrics_to_plot)

    for idx, metric_key in enumerate(metrics_to_plot):
        row = idx // cols + 1
        col = idx % cols + 1

        raw_values = _compute_raw_values_for_metric(
            metric_key, steps, metrics_history, agents
        )

        _add_history_series_trace(fig, row, col, steps, raw_values, metric_key)

        if enable_statistical_credibility:
            _add_credibility_band_for_metric(
                fig,
                row,
                col,
                metric_key,
                steps,
                raw_values,
                agents,
                credibility_confidence,
            )

        _add_final_value_hline(fig, row, col, raw_values)
        _configure_axes(fig, row, col, x_log, metric_key)

    _apply_layout_history(fig, agent_name, rows, cols)
    _export_figure_history(fig, file_name, width, height, scale)
    return fig
