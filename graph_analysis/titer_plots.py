"""Titer bar charts from hierarchical_analysis.xlsx / analysis_summary.md."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .colors import get_sample_color, register_samples


def _safe_slug(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _probe_order(probes: list[str]) -> list[str]:
    preferred_first = ["control"]
    head = [p for p in preferred_first if p in probes]
    tail = sorted(p for p in probes if p not in head)
    return head + tail


def _parse_mean_std(value) -> tuple[float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), 0.0
    text = str(value)
    match = re.match(r"\s*([+-]?\d+(?:\.\d+)?)\s*±\s*([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"Cannot parse titer ± std from {value!r}")
    return float(match.group(1)), float(match.group(2))


def load_titer_tables(xlsx_path: Path) -> dict[str, pd.DataFrame]:
    """
    Read each sheet of hierarchical_analysis.xlsx into a tidy table:
    probe, n, titer, std, p_value.
    """
    xlsx_path = Path(xlsx_path)
    book = pd.ExcelFile(xlsx_path)
    tables: dict[str, pd.DataFrame] = {}

    for sheet in book.sheet_names:
        raw = pd.read_excel(book, sheet_name=sheet, header=None)
        # Row 0 is "Вид (Type): ...", row 2 is header, data starts at row 3.
        title = str(raw.iloc[0, 0]) if len(raw) else sheet
        type_match = re.search(r"Вид\s*\(Type\):\s*(.+)$", title)
        type_name = type_match.group(1).strip() if type_match else sheet

        header_row = None
        for i in range(min(6, len(raw))):
            cell = str(raw.iloc[i, 0]).strip().lower()
            if cell == "probe":
                header_row = i
                break
        if header_row is None:
            continue

        body = raw.iloc[header_row + 1 :].copy()
        body = body.dropna(how="all")
        if body.empty:
            continue

        rows = []
        for _, row in body.iterrows():
            probe = row.iloc[0]
            if probe is None or (isinstance(probe, float) and np.isnan(probe)):
                continue
            probe = str(probe).strip()
            if not probe or probe.lower() == "probe":
                continue
            n = int(row.iloc[1])
            titer, std = _parse_mean_std(row.iloc[2])
            p_raw = row.iloc[3]
            if p_raw is None or (isinstance(p_raw, float) and np.isnan(p_raw)):
                p_value = np.nan
            else:
                try:
                    p_value = float(p_raw)
                except (TypeError, ValueError):
                    p_value = np.nan if str(p_raw).upper() == "N/A" else np.nan
            rows.append(
                {
                    "type_name": type_name,
                    "probe": probe,
                    "n": n,
                    "titer": titer,
                    "std": std,
                    "p_value": p_value,
                }
            )
        if rows:
            tables[type_name] = pd.DataFrame(rows)
    return tables


def _significance_label(p_value: Optional[float], probe: str) -> str:
    if probe.lower() == "control" or p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_titer_bars(table: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart of mean titer ± std with Welch p-value markers vs control."""
    if table.empty:
        raise ValueError("Empty titer table")

    type_name = str(table["type_name"].iloc[0])
    order = _probe_order(table["probe"].tolist())
    table = table.set_index("probe").loc[order].reset_index()
    register_samples(order)

    x = np.arange(len(table))
    heights = table["titer"].to_numpy(dtype=float)
    errors = table["std"].to_numpy(dtype=float)
    colors = [get_sample_color(p) for p in table["probe"]]

    fig, ax = plt.subplots(figsize=(max(8, 1.15 * len(table)), 5.5))
    bars = ax.bar(
        x,
        heights,
        yerr=errors,
        color=colors,
        alpha=0.85,
        edgecolor="0.25",
        linewidth=0.8,
        capsize=4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )

    ymax = float(np.nanmax(heights + errors)) if len(heights) else 1.0
    for i, (_, row) in enumerate(table.iterrows()):
        label = _significance_label(row["p_value"], str(row["probe"]))
        if not label:
            continue
        top = heights[i] + errors[i]
        ax.text(i, top + 0.03 * ymax, label, ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(table["probe"], rotation=35, ha="right")
    ax.set_ylabel("Titer (10⁶ spores / ml)")
    ax.set_xlabel("Probe")
    ax.set_title(f"Mean titer ± SD vs control\n{type_name}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Compact legend for significance
    ax.text(
        0.99,
        0.98,
        "Welch t-test vs control\n* p<0.05  ** p<0.01  *** p<0.001",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    # Keep a handle so linters know bars is used intentionally
    _ = bars
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_titer_comparison(tables: dict[str, pd.DataFrame], out_path: Path) -> Optional[Path]:
    """
    Side-by-side mean titers for CV vs HandAnalysis when both exist,
    matching probes by stripping `_hand_analysis`.
    """
    cv_keys = [k for k in tables if k.endswith("_CV") and "independent" not in k and "Hand" not in k]
    hand_keys = [k for k in tables if "HandAnalysis" in k or k.endswith("_Hand_analysis")]
    # Prefer exact pair from this dataset
    cv_name = next((k for k in tables if k == "Vairimorpha ceranae_CV"), cv_keys[0] if cv_keys else None)
    hand_name = next(
        (k for k in tables if k == "Vairimorpha ceranae_HandAnalysis"),
        hand_keys[0] if hand_keys else None,
    )
    if cv_name is None or hand_name is None:
        return None

    cv = tables[cv_name].copy()
    hand = tables[hand_name].copy()
    hand["base_probe"] = hand["probe"].str.replace(r"_hand_analysis$", "", regex=True)
    cv["base_probe"] = cv["probe"]

    merged = cv.merge(
        hand,
        on="base_probe",
        suffixes=("_cv", "_hand"),
        how="inner",
    )
    if merged.empty:
        return None

    order = _probe_order(merged["base_probe"].tolist())
    merged = merged.set_index("base_probe").loc[order].reset_index()
    register_samples(order)

    x = np.arange(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(merged)), 5.5))
    ax.bar(
        x - width / 2,
        merged["titer_cv"],
        width,
        yerr=merged["std_cv"],
        label="CV (YOLO)",
        color="#1f77b4",
        alpha=0.85,
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        merged["titer_hand"],
        width,
        yerr=merged["std_hand"],
        label="Hand analysis",
        color="#ff7f0e",
        alpha=0.85,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(merged["base_probe"], rotation=35, ha="right")
    ax.set_ylabel("Titer (10⁶ spores / ml)")
    ax.set_title("CV vs HandAnalysis mean titer ± SD")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_titer_plots(xlsx_path: Path, plots_dir: Path) -> list[Path]:
    tables = load_titer_tables(xlsx_path)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for type_name, table in sorted(tables.items()):
        path = plot_titer_bars(
            table,
            plots_dir / f"titer_bars_{_safe_slug(type_name)}.png",
        )
        written.append(path)

    comparison = plot_titer_comparison(tables, plots_dir / "titer_cv_vs_hand.png")
    if comparison is not None:
        written.append(comparison)
    return written
