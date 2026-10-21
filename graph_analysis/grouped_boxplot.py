"""Grouped morphology boxplots from morphology_spores.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .colors import get_sample_color, register_samples

# Contour rows are the primary morphometry population (see bees/morphology.py).
PRIMARY_SOURCE = "contour"

DEFAULT_METRICS = (
    ("major_px", "Major axis (px)"),
    ("minor_px", "Minor axis (px)"),
    ("axis_ratio", "Axis ratio (minor / major)"),
    ("eccentricity", "Eccentricity"),
)


def _safe_slug(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _probe_order(probes: Iterable[str]) -> list[str]:
    probes = sorted(str(p) for p in probes)
    preferred_first = ["control"]
    head = [p for p in preferred_first if p in probes]
    tail = [p for p in probes if p not in head]
    return head + tail


def load_morphology_spores(
    csv_path: Path,
    *,
    source: Optional[str] = PRIMARY_SOURCE,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load spore table; keep contour rows by default."""
    columns = list(
        usecols
        or [
            "type_name",
            "probe_name",
            "sample_name",
            "major_px",
            "minor_px",
            "axis_ratio",
            "eccentricity",
            "area_px",
            "confidence",
            "source",
        ]
    )
    df = pd.read_csv(csv_path, usecols=columns)
    if source is not None and "source" in df.columns:
        df = df[df["source"] == source]
    return df.reset_index(drop=True)


def plot_metric_by_probe(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    type_name: str,
    out_path: Path,
    *,
    show_fliers: bool = False,
) -> Path:
    """One boxplot panel: metric distribution per probe for a single type."""
    subset = df[df["type_name"] == type_name]
    if subset.empty:
        raise ValueError(f"No rows for type {type_name!r}")

    probes = _probe_order(subset["probe_name"].unique())
    register_samples(probes)
    data = [subset.loc[subset["probe_name"] == p, metric].dropna().to_numpy() for p in probes]
    colors = [get_sample_color(p) for p in probes]

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(probes)), 5.5))
    bp = ax.boxplot(
        data,
        labels=probes,
        patch_artist=True,
        showfliers=show_fliers,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(f"{ylabel}\n{type_name}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Probe")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_metrics_overview(
    df: pd.DataFrame,
    type_name: str,
    out_path: Path,
    metrics: Sequence[tuple[str, str]] = DEFAULT_METRICS,
    *,
    show_fliers: bool = False,
) -> Path:
    """2x2 overview of key morphometry metrics for one type."""
    subset = df[df["type_name"] == type_name]
    if subset.empty:
        raise ValueError(f"No rows for type {type_name!r}")

    probes = _probe_order(subset["probe_name"].unique())
    register_samples(probes)
    colors = [get_sample_color(p) for p in probes]

    n = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), squeeze=False)

    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        data = [subset.loc[subset["probe_name"] == p, metric].dropna().to_numpy() for p in probes]
        bp = ax.boxplot(
            data,
            labels=probes,
            patch_artist=True,
            showfliers=show_fliers,
            medianprops={"color": "black", "linewidth": 1.1},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(f"Morphometry overview — {type_name}", y=1.01)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_major_vs_minor(
    df: pd.DataFrame,
    type_name: str,
    out_path: Path,
    *,
    max_points_per_probe: int = 2500,
    seed: int = 0,
) -> Path:
    """Scatter of major vs minor axis, subsampled per probe for readability."""
    subset = df[df["type_name"] == type_name]
    if subset.empty:
        raise ValueError(f"No rows for type {type_name!r}")

    probes = _probe_order(subset["probe_name"].unique())
    register_samples(probes)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    for probe in probes:
        part = subset[subset["probe_name"] == probe][["major_px", "minor_px"]].dropna()
        if len(part) > max_points_per_probe:
            idx = rng.choice(len(part), size=max_points_per_probe, replace=False)
            part = part.iloc[idx]
        ax.scatter(
            part["major_px"],
            part["minor_px"],
            s=8,
            alpha=0.35,
            color=get_sample_color(probe),
            label=probe,
            edgecolors="none",
        )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, color="0.5", linestyle="--", linewidth=1, label="major = minor")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Major axis (px)")
    ax.set_ylabel("Minor axis (px)")
    ax.set_title(f"Major vs minor — {type_name}")
    ax.legend(markerscale=2, fontsize=8, frameon=False)
    ax.grid(linestyle=":", alpha=0.45)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_morphology_plots(csv_path: Path, plots_dir: Path) -> list[Path]:
    """Generate morphology charts for every type present in the CSV."""
    df = load_morphology_spores(csv_path)
    if df.empty:
        return []

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for type_name in sorted(df["type_name"].dropna().unique()):
        slug = _safe_slug(type_name)
        written.append(
            plot_metrics_overview(
                df,
                type_name,
                plots_dir / f"morphology_overview_{slug}.png",
            )
        )
        for metric, ylabel in DEFAULT_METRICS:
            written.append(
                plot_metric_by_probe(
                    df,
                    metric,
                    ylabel,
                    type_name,
                    plots_dir / f"morphology_{metric}_{slug}.png",
                )
            )
        written.append(
            plot_major_vs_minor(
                df,
                type_name,
                plots_dir / f"morphology_major_vs_minor_{slug}.png",
            )
        )
    return written
