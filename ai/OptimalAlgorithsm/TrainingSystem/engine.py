"""Subject-disjoint optimization, validation selection and final test reporting."""

import json
import posixpath
import re
import time
from pathlib import Path, PurePosixPath

import numpy as np

from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import (
    FEATURES,
    Fitness,
    classification_metrics,
    load_jsonl,
    selection_key,
    validate_labels,
)


def split_dataset(records, y, seed=42, group_by="subject"):
    y = validate_labels(y, len(records))
    if group_by not in ("subject", "video"):
        raise ValueError("group_by must be subject or video")
    groups, video_groups = [], {}
    for row in records:
        source = row.get("source_video")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Grouped evaluation requires source_video on every record")
        video = PurePosixPath(posixpath.normpath(source.replace("\\", "/")))
        subject = row.get("subject_id")
        uta_subject = (
            video.parent.name
            if re.fullmatch(r"Fold\d+_part\d+", video.parent.parent.name)
            and video.parent.name.isdigit()
            else None
        )
        uldd_subject = (
            video.parent.parent.name
            if video.parent.name in ("A", "D")
            and re.fullmatch(r"[A-Z]", video.parent.parent.name)
            and re.fullmatch(
                rf"{video.parent.parent.name}_(IR|L3D|R3D)_{video.parent.name}", video.stem
            ) else None
        )
        if group_by == "video":
            group = str(video)
        elif subject is not None:
            if (
                isinstance(subject, bool)
                or not isinstance(subject, (str, int))
                or not str(subject).strip()
            ):
                raise ValueError("subject_id must be a nonempty string or integer")
            group = str(subject).strip()
        elif uta_subject is not None:
            # UTA-RLDD subject IDs are global across fold/part directories.
            group = uta_subject
        elif uldd_subject is not None:
            group = uldd_subject
        else:
            raise ValueError(
                "Provide subject_id for paths outside UTA/ULDD, or explicitly use group_by='video'"
            )
        if group_by == "subject":
            if group.isdecimal():
                group = str(int(group)).zfill(2)
            if uta_subject is not None and group != str(int(uta_subject)).zfill(2):
                raise ValueError(f"subject_id disagrees with UTA video path: {video}")
            if uldd_subject is not None and group != uldd_subject:
                raise ValueError(f"subject_id disagrees with ULDD video path: {video}")
        if str(video) in video_groups and video_groups[str(video)] != group:
            raise ValueError(f"Conflicting subject IDs for video: {video}")
        video_groups[str(video)] = group
        groups.append(group)
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("Metadata must match the sample count")
    unique = np.random.default_rng(seed).permutation(np.unique(groups))
    if len(unique) < 5:
        raise ValueError("Require at least five groups for a 60/20/20 split")
    n_validation = n_test = max(1, int(round(len(unique) * 0.2)))
    partitions = np.split(
        unique, [len(unique) - n_validation - n_test, len(unique) - n_test]
    )
    splits = {}
    for name, partition in zip(("train", "validation", "test"), partitions):
        indices = np.flatnonzero(np.isin(groups, partition))
        if len(np.unique(y[indices])) != 2:
            raise ValueError(
                f"{name} must contain both labels; provide more groups or review the split"
            )
        splits[name] = indices
    return splits, groups


def score_candidates(positions, fitness, y, method):
    results = []
    for candidate in positions:
        mask = candidate[:8] > 0
        if not mask.any():
            continue
        results.append(
            {
                "model": "candidate",
                "candidate": candidate.tolist(),
                "num_features": int(mask.sum()),
                "selected_features": [FEATURES[j] for j in np.flatnonzero(mask)],
                "metrics": classification_metrics(
                    y, fitness.predict(candidate, method)
                ),
            }
        )
    for label in (0, 1):
        results.append(
            {
                "model": f"constant_{label}",
                "constant_label": label,
                "num_features": 0,
                "metrics": classification_metrics(
                    y, np.full(len(y), label, dtype=np.int8)
                ),
            }
        )
    return results


def run_experiment(
    optimizer_type,
    logger_type,
    config,
    data_path,
    output_dir,
    method="noisy_or",
    split_seed=42,
    group_by="subject",
):
    start = time.perf_counter()
    X, y, metadata = load_jsonl(data_path, include_metadata=True)
    splits, groups = split_dataset(metadata["records"], y, split_seed, group_by)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{optimizer_type.__name__.lower()}_{method}"
    run_dir = output_dir / run_name
    attempt = 1
    while True:
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            attempt += 1
            run_dir = output_dir / f"{run_name}_{attempt}"
    manifest = {
        "data_path": str(data_path),
        "sha256": metadata["sha256"],
        "input_schema": metadata["input_schema"],
        "n_samples": len(y),
        "group_by": group_by,
        "split_seed": split_seed,
        "fractions_by_group": [0.6, 0.2, 0.2],
        "splits": {
            name: {
                "n_samples": len(idx),
                "labels": np.bincount(y[idx], minlength=2).tolist(),
                "groups": sorted(set(groups[idx].tolist())),
                "sample_indices": idx.tolist(),
            }
            for name, idx in splits.items()
        },
    }
    (run_dir / "split.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    logger = logger_type(run_dir)
    logger.save_config(config, data_path, method, len(train))
    config_path = run_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "evaluation_scope": "train optimization; validation model selection; held-out test",
            "n_samples_total": len(y),
            "split_manifest": "split.json",
            "input_schema": metadata["input_schema"],
            "selection_rule": "validation balanced_accuracy, then f1; includes constant baselines",
        }
    )
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Samples: train={len(train)}, validation={len(validation)}, test={len(test)}; group_by={group_by}"
    )
    optimizer = optimizer_type(Fitness(X[train]), y[train], method, config, logger)
    archive = optimizer.run()
    results = score_candidates(
        archive.positions, Fitness(X[validation]), y[validation], method
    )
    selected = max(results, key=selection_key)
    with (run_dir / "validation_archive.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(results, key=selection_key, reverse=True):
            f.write(json.dumps(row) + "\n")
    # Freeze the validation choice before accessing test predictions.
    if selected["model"] == "candidate":
        prediction = Fitness(X[test]).predict(selected["candidate"], method)
    else:
        prediction = np.full(len(test), selected["constant_label"], dtype=np.int8)
    best_baseline = max(
        (r for r in results if r["model"] != "candidate"), key=selection_key
    )
    report = {
        "fitness_method": method,
        "input_schema": metadata["input_schema"],
        "selected_model": {k: v for k, v in selected.items() if k != "metrics"},
        "validation_metrics": selected["metrics"],
        "validation_beats_constant_baseline": selection_key(selected)
        > selection_key(best_baseline),
        "test_metrics": classification_metrics(y[test], prediction),
        "test_baselines": {
            f"constant_{label}": classification_metrics(
                y[test], np.full(len(test), label, dtype=np.int8)
            )
            for label in (0, 1)
        },
        "evaluations": optimizer.evaluations,
        "elapsed_seconds": time.perf_counter() - start,
        "output_dir": str(run_dir),
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    m = report["test_metrics"]
    print(
        f"Selected on validation: {selected['model']}; held-out test: "
        f"acc={m['accuracy']:.4f} "
        f"recall={m['recall']:.4f} precision={m['precision']:.4f} "
        f"balanced_acc={m['balanced_accuracy']:.4f} FPR={m['false_positive_rate']:.4f}"
    )
    print(f"Outputs: {run_dir}")
    return report
