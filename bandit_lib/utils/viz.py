from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING
import math
import hashlib

from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objects as go  # type: ignore

from bandit_lib.utils.schemas import ProcessDataDump


if TYPE_CHECKING:
    from bandit_lib.agents.base import BaseAgent, Agent_T
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

    rows, cols = _determine_layout(len(metrics_to_plot))

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


def _get_color_from_name(name: str) -> str:
    """Generate a stable hex color from a string name using hash

    Args:
        name: The string to generate color from (e.g., agent name)

    Returns:
        A hex color string like "#a1c3ef"
    """
    # Create MD5 hash of the name
    hash_obj = hashlib.md5(name.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()

    # Take first 6 characters of hash and add # prefix
    color_hex = hash_hex[:6]
    return f"#{color_hex}"


def _find_intersections(
    x1: List[float], y1: List[float], x2: List[float], y2: List[float]
) -> List[Tuple[float, float]]:
    """Find intersection points between two curves using linear interpolation

    Args:
        x1, y1: Data points for the first curve
        x2, y2: Data points for the second curve

    Returns:
        List of (x, y) intersection points
    """
    intersections: List[Tuple[float, float]] = []

    if len(x1) < 2 or len(x2) < 2:
        return intersections

    # For each segment in curve 1, check for intersections with curve 2
    for i in range(len(x1) - 1):
        x1_start, x1_end = x1[i], x1[i + 1]
        y1_start, y1_end = y1[i], y1[i + 1]

        for j in range(len(x2) - 1):
            x2_start, x2_end = x2[j], x2[j + 1]
            y2_start, y2_end = y2[j], y2[j + 1]

            x_overlap_start = max(x1_start, x2_start)
            x_overlap_end = min(x1_end, x2_end)

            if x_overlap_start >= x_overlap_end:
                continue

            # Linear interpolation for curve 1 in overlap region
            t1 = (
                (x_overlap_start - x1_start) / (x1_end - x1_start)
                if x1_end != x1_start
                else 0
            )
            t2 = (
                (x_overlap_end - x1_start) / (x1_end - x1_start)
                if x1_end != x1_start
                else 1
            )

            y1_overlap_start = y1_start + t1 * (y1_end - y1_start)
            y1_overlap_end = y1_start + t2 * (y1_end - y1_start)

            # Linear interpolation for curve 2 in overlap region
            t3 = (
                (x_overlap_start - x2_start) / (x2_end - x2_start)
                if x2_end != x2_start
                else 0
            )
            t4 = (
                (x_overlap_end - x2_start) / (x2_end - x2_start)
                if x2_end != x2_start
                else 1
            )

            y2_overlap_start = y2_start + t3 * (y2_end - y2_start)
            y2_overlap_end = y2_start + t4 * (y2_end - y2_start)

            # Check for sign changes (crossings) in the overlap region
            if (y1_overlap_start - y2_overlap_start) * (
                y1_overlap_end - y2_overlap_end
            ) < 0:
                # Calculate intersection point using linear interpolation
                # Find the exact x where y1(x) = y2(x)
                denom = (y1_overlap_end - y2_overlap_end) - (
                    y1_overlap_start - y2_overlap_start
                )
                if abs(denom) > 1e-10:  # Avoid division by zero
                    t = (y2_overlap_start - y1_overlap_start) / denom
                    x_intersect = x_overlap_start + t * (
                        x_overlap_end - x_overlap_start
                    )

                    # Interpolate y value
                    y_intersect = y1_overlap_start + t * (
                        y1_overlap_end - y1_overlap_start
                    )
                    intersections.append((x_intersect, y_intersect))

    return intersections


def plot_comparison(
    runs_data: List[Tuple[ProcessDataDump, List["Agent_T"]]],
    file_name: Path,
    x_log: bool = False,
    metrics_to_plot: List[str] = _METRIC_LABELS,
    width: int = 2000,
    height: int = 1000,
    scale: int = 2,
    show_intersections: bool = True,
    intersection_marker_size: int = 8,
    intersection_marker_color: str = "red",
) -> go.Figure:
    """Plot comparison of multiple experiment runs with intersection points marked

    Args:
        runs_data: List of tuples containing (ProcessDataDump, List[Agent_T])
                  for each experiment run to compare
        file_name: The name of the file to save the plot
        x_log: Whether to use a logarithmic x-axis
        metrics_to_plot: The metrics to plot
        width: The width of the image
        height: The height of the image
        scale: The scale of the image
        show_intersections: Whether to show intersection points between curves
        intersection_marker_size: Size of intersection markers
        intersection_marker_color: Color of intersection markers

    Returns:
        The plotly figure object
    """
    if not runs_data:
        raise ValueError("runs_data cannot be empty")
    if not metrics_to_plot:
        metrics_to_plot = _METRIC_LABELS.copy()

    # layout
    rows, cols = _determine_layout(len(metrics_to_plot))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=metrics_to_plot,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # pre-process
    run_labels: List[str] = []
    run_colors: List[str] = []
    for run_idx, (process_dump, _agents) in enumerate(runs_data):
        if _agents and getattr(_agents[0], "name", None):
            label = str(_agents[0].name)
        else:
            # fallback to run_id
            label = (
                process_dump.run_id
                if getattr(process_dump, "run_id", None)
                else f"run_{run_idx + 1}"
            )
        run_labels.append(label)
        run_colors.append(_get_color_from_name(label))

    # plot each metric
    for idx, metric_key in enumerate(metrics_to_plot):
        row = idx // cols + 1
        col = idx % cols + 1

        all_x_series: List[List[float]] = []
        all_y_series: List[List[float]] = []

        label_to_series: Dict[str, List[Tuple[List[float], List[float]]]] = {}
        for run_idx, (process_dump, agents) in enumerate(runs_data):
            history = process_dump.metrics_history_avg
            if not history:
                if (
                    process_dump.metrics_history
                    and len(process_dump.metrics_history[0]) > 0
                ):
                    history = process_dump.metrics_history[0]
                else:
                    continue

            steps = [float(m.current_step) for m in history]
            if metric_key == "convergence_rate":
                if not agents:
                    continue
                y_values = _compute_convergence_rate_series(steps, agents)  # type: ignore[arg-type]
            else:
                y_values = [float(getattr(m, metric_key)) for m in history]

            label = run_labels[run_idx]
            label_to_series.setdefault(label, []).append((steps, y_values))

        for label_idx, (label, series_list) in enumerate(label_to_series.items()):
            if not series_list:
                continue
            base_steps = series_list[0][0]
            min_len = min(len(s[0]) for s in series_list)
            aligned_steps = base_steps[:min_len]
            avg_values: List[float] = []
            for idx_step in range(min_len):
                vals = [s[1][idx_step] for s in series_list if len(s[1]) > idx_step]
                if not vals:
                    avg_values.append(float("nan"))
                else:
                    avg_values.append(float(sum(vals) / len(vals)))

            all_x_series.append(list(map(float, aligned_steps)))
            all_y_series.append(list(map(float, avg_values)))

            color = _get_color_from_name(label)
            fig.add_trace(
                go.Scatter(
                    x=aligned_steps,
                    y=avg_values,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    legendgroup=label,
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )

        if show_intersections and len(all_x_series) >= 2:
            curve_count = len(all_x_series)
            for i in range(curve_count - 1):
                for j in range(i + 1, curve_count):
                    intersections = _find_intersections(
                        all_x_series[i],
                        all_y_series[i],
                        all_x_series[j],
                        all_y_series[j],
                    )
                    if not intersections:
                        continue
                    xs = [pt[0] for pt in intersections]
                    ys = [pt[1] for pt in intersections]
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="markers",
                            name=f"Intersection: {i + 1} vs {j + 1} ({metric_key})",
                            marker=dict(
                                size=intersection_marker_size,
                                color=intersection_marker_color,
                                symbol="x",
                            ),
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

        xaxis_kwargs: Dict[str, Any] = {"title": "Steps"}
        if x_log:
            xaxis_kwargs["type"] = "log"
        fig.update_xaxes(row=row, col=col, **xaxis_kwargs)
        fig.update_yaxes(row=row, col=col, title=metric_key)

    final_height = height if height and height > 0 else max(400, rows * 500)
    final_width = width if width and width > 0 else max(600, cols * 500)
    fig.update_layout(
        title="Comparison across runs",
        height=final_height,
        width=final_width,
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            tracegroupgap=8,
        ),
        margin=dict(l=60, r=120, t=60, b=50),
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
            fig.write_image(
                str(output_path), width=final_width, height=final_height, scale=scale
            )
    except ValueError:
        print("Error writing image, falling back to HTML")
        fallback_path = output_path.with_suffix(".html")
        fig.write_html(str(fallback_path))
        output_path = fallback_path

    return fig


def get_metric_labels() -> List[str]:
    return _METRIC_LABELS.copy()
