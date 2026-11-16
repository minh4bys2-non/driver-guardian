"""Aggregate validation_metrics across TrainingSystem output versions into output/summary/."""

import argparse
import json
import re
from pathlib import Path

ALGORITHM_ORDER = ("cmdpsofs", "nspsofs", "ibgwo4", "mohho")
METHOD_ORDER = ("cooccurrence", "noisy_or", "weighted_mean", "physical_quadratic")
METRIC_KEY = "validation_metrics"


def _ordered_keys(keys, preferred_order):
    """Return keys ordered by preferred_order first, then alphabetically."""
    preferred = [k for k in preferred_order if k in keys]
    remaining = sorted(k for k in keys if k not in preferred_order)
    return preferred + remaining


def parse_run_version(run_dir_name: str, algorithm: str, method: str) -> int | None:
    """Extract version number (>= 1) from a run directory name, or None if unrelated."""
    prefix = f"{algorithm}_{method}"
    if run_dir_name == prefix:
        return 1
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", run_dir_name)
    if match:
        version = int(match.group(1))
        return version if version >= 2 else None
    return None


def summary_filename(version: int) -> str:
    """Return output summary filename for the given version number."""
    if version < 1:
        raise ValueError(f"Version must be >= 1, got {version}")
    return "summary.json" if version == 1 else f"summary_{version}.json"


def collect_summaries(output_dir: Path, metric_key: str = METRIC_KEY) -> dict[int, dict]:
    """Scan output_dir and group extracted evaluation metrics by run version."""
    output_dir = Path(output_dir)
    raw_by_version: dict[int, dict[str, dict[str, dict]]] = {}

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    algo_dirs = [
        d.name
        for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ("summary", "__pycache__") and not d.name.startswith(".")
    ]
    for algorithm in _ordered_keys(algo_dirs, ALGORITHM_ORDER):
        algo_path = output_dir / algorithm
        for run_dir in sorted(algo_path.iterdir()):
            if not run_dir.is_dir():
                continue
            eval_path = run_dir / "evaluation.json"
            if not eval_path.is_file():
                continue
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            method = evaluation.get("fitness_method")
            if not isinstance(method, str) or not method:
                continue
            version = parse_run_version(run_dir.name, algorithm, method)
            if version is None:
                continue
            if metric_key not in evaluation:
                raise KeyError(f"Missing '{metric_key}' in {eval_path}")
            raw_by_version.setdefault(version, {}).setdefault(algorithm, {})[method] = evaluation[metric_key]

    ordered_by_version: dict[int, dict] = {}
    for version in sorted(raw_by_version):
        algo_map = raw_by_version[version]
        ordered_algos = {}
        for algorithm in _ordered_keys(algo_map.keys(), ALGORITHM_ORDER):
            method_map = algo_map[algorithm]
            ordered_algos[algorithm] = {
                method: method_map[method]
                for method in _ordered_keys(method_map.keys(), METHOD_ORDER)
            }
        ordered_by_version[version] = ordered_algos

    return ordered_by_version


def write_summaries(
    output_dir: Path,
    summary_dir: Path | None = None,
    metric_key: str = METRIC_KEY,
    target_version: int | None = None,
) -> dict[int, Path]:
    """Write per-version summary JSON files to summary_dir and return written paths."""
    output_dir = Path(output_dir)
    summary_dir = Path(summary_dir) if summary_dir is not None else output_dir / "summary"
    summaries = collect_summaries(output_dir, metric_key=metric_key)

    if target_version is not None:
        if target_version not in summaries:
            raise ValueError(f"Version {target_version} not found in {output_dir}")
        summaries = {target_version: summaries[target_version]}

    summary_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    for version, payload in summaries.items():
        out_path = summary_dir / summary_filename(version)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written[version] = out_path
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for aggregating TrainingSystem output summaries."""
    default_output_dir = Path(__file__).resolve().parent / "output"
    parser = argparse.ArgumentParser(
        description="Aggregate TrainingSystem evaluation.json metrics across output versions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Root output directory containing algorithm subdirectories.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Destination directory for summary JSON files (defaults to <output-dir>/summary).",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Generate summary only for a specific version number (1, 2, 3, ...).",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default=METRIC_KEY,
        help=f"Metric field in evaluation.json to extract (default: {METRIC_KEY}).",
    )
    args = parser.parse_args(argv)

    written = write_summaries(
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
        metric_key=args.metric_key,
        target_version=args.version,
    )
    for version, path in written.items():
        print(f"Version {version}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
