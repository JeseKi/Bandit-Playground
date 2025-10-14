from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, TYPE_CHECKING

from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objects as go  # type: ignore

from bandit_lib.utils.viz.constants import _METRIC_LABELS
from bandit_lib.utils.viz.utils import determine_layout, compute_convergence_rate_series


if TYPE_CHECKING:
    from bandit_lib.agents.base import BaseAgent
    from bandit_lib.agents.schemas import Metrics


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
) -> go.Figure:
    """Plot the metrics history with plotly

    Args:
        metrics_history: The metrics history to plot.
        agent_name: The name of the agent.
        file_name: The name of the file to save the plot.
        agents: The agents to plot, for calculating the static convergence rate.
        x_log: Whether to use a logarithmic x-axis.
        metrics_to_plot: The metrics to plot.
        width: The width of the image.
        height: The height of the image.
        scale: The scale of the image.
    """
    if not metrics_history or not agents:
        raise ValueError("metrics_history or agents is empty")

    steps = [metric.current_step for metric in metrics_history]

    rows, cols = determine_layout(len(metrics_to_plot))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=metrics_to_plot,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, metric_key in enumerate(metrics_to_plot):
        row = idx // cols + 1
        col = idx % cols + 1

        if metric_key == "convergence_rate":
            raw_values = compute_convergence_rate_series(steps, agents)
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
