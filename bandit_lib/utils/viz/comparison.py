from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import math

from plotly.subplots import make_subplots  # type: ignore
import plotly.graph_objects as go  # type: ignore

from bandit_lib.utils.schemas import ProcessDataDump
from bandit_lib.utils.viz.constants import _METRIC_LABELS
from bandit_lib.utils.viz.utils import (
    determine_layout,
    get_color_from_name,
    compute_mean_and_ci,
    hex_to_rgba,
    z_value_for_confidence,
)
from bandit_lib.utils.viz.history import compute_convergence_rate_series


if TYPE_CHECKING:
    from bandit_lib.agents.base import Agent_T


def _validate_inputs_for_comparison(
    runs_data: List[Tuple[ProcessDataDump, List["Agent_T"]]],
    metrics_to_plot: List[str],
) -> List[str]:
    """Validate inputs and normalize metrics list.

    Returns a non-empty list of metrics to plot.
    """
    if not runs_data:
        raise ValueError("runs_data cannot be empty")
    if not metrics_to_plot:
        return _METRIC_LABELS.copy()
    return metrics_to_plot


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


def _build_run_labels_and_colors(
    runs_data: List[Tuple[ProcessDataDump, List["Agent_T"]]],
) -> List[str]:
    """Build per-run labels and stable colors from agent name or run_id."""
    run_labels: List[str] = []
    for run_idx, (process_dump, _agents) in enumerate(runs_data):
        if _agents and getattr(_agents[0], "name", None):
            label = str(_agents[0].name)
        else:
            label = (
                process_dump.run_id
                if getattr(process_dump, "run_id", None)
                else f"run_{run_idx + 1}"
            )
        run_labels.append(label)
    return run_labels


def _extract_history_for_run(process_dump: ProcessDataDump) -> List[Any]:
    """Extract a representative metrics history sequence from a process dump."""
    history = process_dump.metrics_history_avg
    if not history:
        if process_dump.metrics_history and len(process_dump.metrics_history[0]) > 0:
            history = process_dump.metrics_history[0]
        else:
            return []
    return history


def _collect_metric_series_for_runs(
    runs_data: List[Tuple[ProcessDataDump, List["Agent_T"]]],
    run_labels: List[str],
    metric_key: str,
    enable_statistical_credibility: bool,
) -> Tuple[
    Dict[str, List[Tuple[List[float], List[float]]]],
    Dict[str, List[List[float]]],
    Dict[str, List[float]],
    Dict[str, int],
]:
    """Collect time series per label and optional replication series for CI."""
    label_to_series: Dict[str, List[Tuple[List[float], List[float]]]] = {}
    label_to_rep_series: Dict[str, List[List[float]]] = {}
    label_to_rep_steps: Dict[str, List[float]] = {}
    label_to_agents_count: Dict[str, int] = {}

    for run_idx, (process_dump, agents) in enumerate(runs_data):
        history = _extract_history_for_run(process_dump)
        if not history:
            continue

        steps = [float(m.current_step) for m in history]
        if metric_key == "convergence_rate":
            if not agents:
                continue
            y_values = compute_convergence_rate_series(steps, agents)  # type: ignore[arg-type]
            label_to_agents_count[run_labels[run_idx]] = len(agents)
        else:
            y_values = [float(getattr(m, metric_key)) for m in history]

        label = run_labels[run_idx]
        label_to_series.setdefault(label, []).append((steps, y_values))

        # metrics_history: List[List[Metrics]], each agent has a complete history
        if enable_statistical_credibility and process_dump.metrics_history:
            rep_series: List[List[float]] = []
            min_len_rep = None
            for seq in process_dump.metrics_history:
                vals = [float(getattr(m, metric_key)) for m in seq]
                if not vals:
                    continue
                if min_len_rep is None:
                    min_len_rep = len(vals)
                else:
                    min_len_rep = min(min_len_rep, len(vals))
                rep_series.append(vals)
            if rep_series and min_len_rep:
                # Align to the same length across replications
                rep_series = [arr[:min_len_rep] for arr in rep_series]
                label_to_rep_series.setdefault(label, []).extend(rep_series)
                label_to_rep_steps[label] = steps[:min_len_rep]

    return (
        label_to_series,  # Series data for every label
        label_to_rep_series,  # Replication series per label (for statistical confidence intervals)
        label_to_rep_steps,  # Aligned step timestamps corresponding to replication series.
        label_to_agents_count,  # Number of agents per label (only populated for convergence_rate metric)
    )


def _align_and_average_series(
    series_list: List[Tuple[List[float], List[float]]],
) -> Tuple[List[float], List[float]]:
    """Align series by step index and compute the mean value across series."""
    if not series_list:
        return [], []
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
    return list(map(float, aligned_steps)), list(map(float, avg_values))


def _add_series_trace(
    fig: go.Figure,
    row: int,
    col: int,
    label: str,
    aligned_steps: List[float],
    avg_values: List[float],
    metric_block_index: int,
) -> None:
    """Add a line+markers trace for the averaged series of a label."""
    color = get_color_from_name(label)
    fig.add_trace(
        go.Scatter(
            x=aligned_steps,
            y=avg_values,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            legendgroup=label,
            showlegend=(metric_block_index == 0),
        ),
        row=row,
        col=col,
    )


def _add_ci_band_for_label(
    fig: go.Figure,
    row: int,
    col: int,
    label: str,
    metric_key: str,
    aligned_steps: List[float],
    avg_values: List[float],
    label_to_rep_series: Dict[str, List[List[float]]],
    label_to_rep_steps: Dict[str, List[float]],
    label_to_agents_count: Dict[str, int],
    credibility_confidence: float,
) -> None:
    """Draw statistical credibility band for a label when enabled."""
    color = get_color_from_name(label)
    if metric_key == "convergence_rate":
        n_agents = label_to_agents_count.get(label, 0)
        if n_agents <= 0:
            return
        z = z_value_for_confidence(credibility_confidence)
        # Binomial approximation: p ± z * sqrt(p(1-p)/n)
        ci_lower: List[float] = []
        ci_upper: List[float] = []
        for p in avg_values:
            se = math.sqrt(max(0.0, p * (1.0 - p)) / n_agents)
            margin = z * se
            ci_lower.append(max(0.0, p - margin))
            ci_upper.append(min(1.0, p + margin))
        fig.add_trace(
            go.Scatter(
                x=aligned_steps,
                y=ci_lower,
                mode="lines",
                line=dict(color=color, width=0),
                name=f"{label} CI lower",
                hoverinfo="skip",
                legendgroup=label,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=aligned_steps,
                y=ci_upper,
                mode="lines",
                line=dict(color=color, width=0),
                fill="tonexty",
                fillcolor=hex_to_rgba(color, 0.18),
                name=f"{label} {int(credibility_confidence * 100)}% CI",
                hoverinfo="skip",
                legendgroup=label,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    else:
        series_for_ci = label_to_rep_series.get(label)
        x_for_ci = label_to_rep_steps.get(label)
        if series_for_ci and x_for_ci and len(series_for_ci) >= 2:
            mean_v, lower_v, upper_v = compute_mean_and_ci(
                series_for_ci, confidence=credibility_confidence
            )
            x_ci = x_for_ci[: len(mean_v)]
            fig.add_trace(
                go.Scatter(
                    x=x_ci,
                    y=lower_v,
                    mode="lines",
                    line=dict(color=color, width=0),
                    name=f"{label} CI lower",
                    hoverinfo="skip",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_ci,
                    y=upper_v,
                    mode="lines",
                    line=dict(color=color, width=0),
                    fill="tonexty",
                    fillcolor=hex_to_rgba(color, 0.18),
                    name=f"{label} {int(credibility_confidence * 100)}% CI",
                    hoverinfo="skip",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )


def _add_intersections(
    fig: go.Figure,
    row: int,
    col: int,
    all_x_series: List[List[float]],
    all_y_series: List[List[float]],
    metric_key: str,
    intersection_marker_size: int,
    intersection_marker_color: str,
) -> None:
    """Compute pairwise intersections and add marker traces if any."""
    if len(all_x_series) < 2:
        return
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


def _configure_axes(
    fig: go.Figure, row: int, col: int, x_log: bool, metric_key: str
) -> None:
    """Configure axes titles and optional logarithmic scale."""
    xaxis_kwargs: Dict[str, Any] = {"title": "Steps"}
    if x_log:
        xaxis_kwargs["type"] = "log"
    fig.update_xaxes(row=row, col=col, **xaxis_kwargs)
    fig.update_yaxes(row=row, col=col, title=metric_key)


def _compute_final_dimensions(
    height: int, width: int, rows: int, cols: int
) -> Tuple[int, int]:
    """Compute final figure dimensions with fallbacks."""
    final_height = height if height and height > 0 else max(400, rows * 500)
    final_width = width if width and width > 0 else max(600, cols * 500)
    return final_height, final_width


def _apply_layout(fig: go.Figure, final_height: int, final_width: int) -> None:
    """Apply common layout settings to the figure."""
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


def _export_figure(
    fig: go.Figure,
    file_name: Path,
    final_width: int,
    final_height: int,
    scale: int,
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
            fig.write_image(
                str(output_path), width=final_width, height=final_height, scale=scale
            )
    except ValueError:
        print("Error writing image, falling back to HTML")
        fallback_path = output_path.with_suffix(".html")
        fig.write_html(str(fallback_path))
        output_path = fallback_path
    return output_path


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
    enable_statistical_credibility: bool = False,
    credibility_confidence: float = 0.95,
) -> go.Figure:
    """Plot comparison of multiple experiment runs with optional intersections.

    Args:
        runs_data: List of (process_dump, agents) tuples, process_dump is the process data dump of the run, agents is used for statistical credibility and convergence rate.
        file_name: Path to save the figure
        x_log: Whether to use logarithmic scale for the x-axis
        metrics_to_plot: List of metrics to plot
        width: Width of the figure
        height: Height of the figure
        scale: Scale of the figure
        show_intersections: Whether to show intersections
        intersection_marker_size: Size of the intersection marker
        intersection_marker_color: Color of the intersection marker
        enable_statistical_credibility: Whether to enable statistical credibility
        credibility_confidence: Confidence level for the statistical credibility

    Returns:
        Plotly figure
    """
    metrics_to_plot = _validate_inputs_for_comparison(runs_data, metrics_to_plot)
    fig, rows, cols = _create_subplots(metrics_to_plot)

    run_labels = _build_run_labels_and_colors(runs_data)

    # plot each metric
    for metric_block_index, metric_key in enumerate(metrics_to_plot):
        row = metric_block_index // cols + 1
        col = metric_block_index % cols + 1

        all_x_series: List[List[float]] = []
        all_y_series: List[List[float]] = []

        (
            label_to_series,
            label_to_rep_series,
            label_to_rep_steps,
            label_to_agents_count,
        ) = _collect_metric_series_for_runs(
            runs_data,
            run_labels,
            metric_key,
            enable_statistical_credibility,
        )

        for label, series_list in label_to_series.items():
            if not series_list:
                continue
            aligned_steps, avg_values = _align_and_average_series(series_list)
            if not aligned_steps:
                continue
            all_x_series.append(aligned_steps)
            all_y_series.append(avg_values)

            _add_series_trace(
                fig,
                row,
                col,
                label,
                aligned_steps,
                avg_values,
                metric_block_index,
            )

            if enable_statistical_credibility:
                _add_ci_band_for_label(
                    fig,
                    row,
                    col,
                    label,
                    metric_key,
                    aligned_steps,
                    avg_values,
                    label_to_rep_series,
                    label_to_rep_steps,
                    label_to_agents_count,
                    credibility_confidence,
                )

        if show_intersections:
            _add_intersections(
                fig,
                row,
                col,
                all_x_series,
                all_y_series,
                metric_key,
                intersection_marker_size,
                intersection_marker_color,
            )

        _configure_axes(fig, row, col, x_log, metric_key)

    final_height, final_width = _compute_final_dimensions(height, width, rows, cols)
    _apply_layout(fig, final_height, final_width)
    _export_figure(fig, file_name, final_width, final_height, scale)
    return fig
