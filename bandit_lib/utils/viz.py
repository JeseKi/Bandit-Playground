from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING
import math

from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objects as go  # type: ignore

if TYPE_CHECKING:
    from bandit_lib.agents.base import BaseAgent
    from bandit_lib.agents.schemas import Metrics

_METRIC_LABELS: List[str] = [
    "regret_rate",
    "regret",
    "reward_rate",
    "reward",
    "optimal_arm_rate",
    "convergence_rate",
    "sliding_window_reward_rate",
]


def _compute_convergence_rate_series(
    steps: Sequence[float], agents: Sequence["BaseAgent"]
) -> List[float]:
    """Compute the convergence rate series (the proportion of agents that have converged within a given time step)"""
    if not agents:
        raise ValueError(
            "`agents` cannot be empty, otherwise the convergence rate cannot be calculated"
        )

    total = len(agents)
    rates: List[float] = []
    for step in steps:
        converged = 0
        for agent in agents:
            convergence_step = getattr(agent, "convergence_step", None)
            if convergence_step is None or convergence_step <= 0:
                continue
            if convergence_step <= step:
                converged += 1
        rates.append(converged / total)
    return rates


def _determine_layout(metric_count: int) -> Tuple[int, int]:
    """Determine the layout of the subplots (rows, cols) based on the number of metrics"""
    if metric_count <= 0:
        return 1, 1
    cols = 3 if metric_count > 2 else 3
    rows = math.ceil(metric_count / cols)
    return rows, cols


def plot_metrics_history(
    metrics_history: Sequence["Metrics"],
    agent_name: str,
    file_name: Path,
    agents: Sequence["BaseAgent"],
    x_log: bool = False,
    width: int = 1500,
    height: int = 1500,
    scale: int = 2,
) -> go.Figure:
    """Plot the metrics history with plotly

    Args:
        metrics_history: The metrics history to plot.
        agent_name: The name of the agent.
        file_name: The name of the file to save the plot.
        agents: The agents to plot, for calculating the static convergence rate.
        x_log: Whether to use a logarithmic x-axis.
        width: The width of the image.
        height: The height of the image.
        scale: The scale of the image.
    """
    if not metrics_history or not agents:
        raise ValueError("metrics_history or agents is empty")

    steps = [metric.current_step for metric in metrics_history]

    canonical_metrics = _METRIC_LABELS
    if agents[0].env.config.enable_dynamic:
        canonical_metrics.remove("convergence_rate")

    rows, cols = _determine_layout(len(canonical_metrics))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=canonical_metrics,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, metric_key in enumerate(canonical_metrics):
        row = idx // cols + 1
        col = idx % cols + 1

        if metric_key == "convergence_rate":
            raw_values = _compute_convergence_rate_series(steps, agents)
        else:
            raw_values = [getattr(metric, metric_key) for metric in metrics_history]

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

        final_value = next(
            (value for value in reversed(raw_values) if value is not None), None
        )
        if final_value is not None:
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

        xaxis_kwargs: Dict[str, Any] = {"title": "Steps"}
        if x_log:
            xaxis_kwargs["type"] = "log"
        fig.update_xaxes(row=row, col=col, **xaxis_kwargs)
        fig.update_yaxes(row=row, col=col, title=metric_key)

    title = f'"{agent_name}" metrics history' if agent_name else "Metrics history"
    fig.update_layout(
        title=title,
        height=max(400, rows * 350),
        width=max(600, cols * 500),
        showlegend=False,
        template="plotly_white",
    )

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

    return fig
