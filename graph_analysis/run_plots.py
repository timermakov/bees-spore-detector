#!/usr/bin/env python3
"""
Build matplotlib charts for a hierarchical analysis result folder.

Default layout (recommended):
  results/<run>/                 ← CSV / XLSX / markdown from the pipeline
  results/<run>/plots/           ← PNG charts for that run

Scripts live in graph_analysis/; images stay next to the data they describe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python graph_analysis/run_plots.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graph_analysis.grouped_boxplot import write_morphology_plots
from graph_analysis.titer_plots import write_titer_plots


def default_results_dir() -> Path:
    return _REPO_ROOT / "results" / "cermorfh"


def resolve_plots_dir(results_dir: Path, plots_dir: Path | None) -> Path:
    if plots_dir is not None:
        return plots_dir
    return results_dir / "plots"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir(),
        help="Folder with hierarchical_analysis.xlsx and morphology_spores.csv",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Output folder for PNGs (default: <results-dir>/plots)",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir.resolve()
    plots_dir = resolve_plots_dir(results_dir, args.plots_dir).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    titer_xlsx = results_dir / "hierarchical_analysis.xlsx"
    if titer_xlsx.is_file():
        written.extend(write_titer_plots(titer_xlsx, plots_dir))
        print(f"titer plots from {titer_xlsx}")
    else:
        print(f"skip titer: missing {titer_xlsx}")

    morph_csv = results_dir / "morphology_spores.csv"
    if morph_csv.is_file():
        written.extend(write_morphology_plots(morph_csv, plots_dir))
        print(f"morphology plots from {morph_csv}")
    else:
        print(f"skip morphology: missing {morph_csv}")

    if not written:
        print("No plots written.", file=sys.stderr)
        return 1

    print(f"Wrote {len(written)} files → {plots_dir}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
