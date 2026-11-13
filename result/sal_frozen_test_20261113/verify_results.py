"""Verify frozen SAL test caches without inference or filesystem writes.

Run with the project's Python environment. Results are JSON on stdout;
exit status is zero only when both complete test caches pass every check.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np


FROZEN_SHA256 = "6b05b7406e20b719c7a55ea40c90905fda7f9d4f01ea7d3d50604d7c40c7d4ff"
CONFUSION_KEY = "confusion_true_rows_pred_columns_0_spoof_1_bonafide"
TOLERANCE_PERCENT = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(project, run_path, split, expected_count, resolution):
    status = read_json(run_path / "status.json")
    metrics = read_json(run_path / "metrics.json")
    config = read_json(run_path / "config.json")
    manifest = read_json(project / "code/manifests/reduced" / (split + ".json"))
    require(status["status"] == "complete", "run status is not complete")
    require(status["completed"] == status["total"] == expected_count,
            "status recording count differs")
    require(status["run"] == run_path.name, "status run name differs")
    ids = manifest["utterance_ids"]
    require(ids == sorted(set(ids)), "manifest IDs are duplicated or unsorted")
    require(manifest["split"] == split and manifest["profile"] == "reduced",
            "manifest role/profile differs")
    require(manifest["count"] == len(ids) == expected_count,
            "manifest recording count differs")
    manifest_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    require(manifest["manifest_sha256"] == manifest_sha, "manifest digest differs")
    for name, record in (("metrics", metrics), ("config", config)):
        require(record["checkpoint_sha256"] == FROZEN_SHA256,
                name + " frozen checkpoint digest differs")
        require(record["split"] == split and record["data_profile"] == "reduced",
                name + " role/profile differs")
        require(record["run"] == run_path.name, name + " run name differs")
        require(record["checkpoint_kind"] == "reduced_trained",
                name + " checkpoint kind differs")
        require(record["smoke_test"] is False, name + " is a smoke test")
        require(record["planned_utterances"] == record["total_dataset"] == expected_count,
                name + " planned recording count differs")
        require(record["manifest_sha256"] == record["subset_manifest_sha256"] == manifest_sha,
                name + " manifest digest differs")
        require(record["resolution_seconds"] == resolution,
                name + " label resolution differs")

    db_path = (run_path / "predictions.sqlite").resolve()
    confusion = np.zeros((2, 2), dtype=np.int64)
    devices = Counter()
    observed_ids = []
    frame_count = 0
    audio_seconds = 0.0
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as database:
        database.execute("PRAGMA query_only = ON")
        integrity = [row[0] for row in database.execute("PRAGMA integrity_check")]
        require(integrity == ["ok"], "SQLite integrity check failed: " + repr(integrity))
        for expected_ordinal, row in enumerate(database.execute(
                "SELECT uid, ordinal, logits, labels, seconds, runtime, device "
                "FROM predictions ORDER BY ordinal")):
            uid, ordinal, raw_logits, raw_labels, seconds, runtime, device = row
            require(ordinal == expected_ordinal, "ordinal sequence differs at " + str(uid))
            require(isinstance(raw_labels, bytes) and len(raw_labels) > 0,
                    "empty or invalid label blob: " + str(uid))
            require(isinstance(raw_logits, bytes) and len(raw_logits) == 8 * len(raw_labels),
                    "logits/labels length mismatch: " + str(uid))
            labels = np.frombuffer(raw_labels, dtype=np.uint8)
            logits = np.frombuffer(raw_logits, dtype="<f4").reshape(-1, 2)
            require(np.isin(labels, (0, 1)).all(), "nonbinary labels: " + str(uid))
            require(np.isfinite(logits).all(), "nonfinite logits: " + str(uid))
            require(math.isfinite(seconds) and seconds > 0, "invalid duration: " + str(uid))
            require(math.isfinite(runtime) and runtime >= 0, "invalid runtime: " + str(uid))
            require(device in ("cuda", "cpu", "cpu_oom_fallback"),
                    "unknown prediction device: " + str(device))
            predictions = logits.argmax(axis=1)  # Ties select class 0, as in SAL.
            confusion += np.bincount(labels.astype(np.int64) * 2 + predictions,
                                     minlength=4).reshape(2, 2)
            frame_count += len(labels)
            audio_seconds += seconds
            devices[device] += 1
            observed_ids.append(uid)

    require(len(observed_ids) == expected_count, "cache recording count differs")
    require(len(set(observed_ids)) == expected_count, "cache contains duplicate IDs")
    require(set(observed_ids) == set(ids), "cache and manifest UID sets differ")
    require(observed_ids == ids, "cache order differs from frozen manifest")
    require(metrics["utterance_count"] == expected_count, "metrics recording count differs")
    require(metrics["frame_count"] == frame_count, "metrics frame count differs")
    require(metrics[CONFUSION_KEY] == confusion.tolist(), "confusion matrix differs")
    require(metrics["cpu_fallback_utterances"] == devices["cpu_oom_fallback"],
            "CPU fallback count differs")
    require(math.isclose(metrics["audio_seconds"], audio_seconds, rel_tol=1e-10, abs_tol=1e-6),
            "total audio duration differs")
    fake_correct, fake_as_real, real_as_fake, real_correct = map(int, confusion.ravel())

    def f1(correct):
        denominator = 2 * correct + fake_as_real + real_as_fake
        return 100.0 * 2 * correct / denominator if denominator else 0.0

    recalculated = {
        "accuracy_percent": 100.0 * (fake_correct + real_correct) / frame_count,
        "f1_spoof_percent": f1(fake_correct),
        "f1_bonafide_percent": f1(real_correct),
    }
    differences = {}
    for key, value in recalculated.items():
        saved = metrics[key]
        require(isinstance(saved, (int, float)) and math.isfinite(saved),
                "invalid metric: " + key)
        differences[key] = abs(value - saved)
        require(differences[key] <= TOLERANCE_PERCENT,
                key + " differs by " + str(differences[key]))
    return {
        "passed": True,
        "run_directory": str(run_path),
        "database_integrity": "ok",
        "recordings": len(observed_ids),
        "frames": frame_count,
        "uid_set_and_order_match": True,
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": FROZEN_SHA256,
        "device_recording_counts": dict(sorted(devices.items())),
        "confusion_matrix_true_rows_pred_columns_0_fake_1_real": confusion.tolist(),
        "independent_metrics": recalculated,
        "absolute_difference_percentage_points": differences,
        "reported_eer_percent_not_independently_recomputed": metrics["eer_percent"],
        "audio_seconds": audio_seconds,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--eval-run", default="sal_reduced_eval_20261113")
    parser.add_argument("--lps-run", default="sal_reduced_lps_20261113")
    args = parser.parse_args()
    project = args.project_root.resolve()
    summary = {
        "passed": False,
        "read_only_cache_verification": True,
        "percentage_point_tolerance": TOLERANCE_PERCENT,
        "checkpoint_identity_check": "recorded SHA256 in config and metrics; weights not rehashed",
        "results": {},
    }
    for split, run_name, count, resolution in (
            ("eval", args.eval_run, 3562, 0.16),
            ("lps", args.lps_run, 3800, 0.02)):
        run_path = Path(__file__).resolve().parent / "predictions" / run_name
        try:
            summary["results"][split] = verify(project, run_path, split, count, resolution)
        except Exception as error:
            summary["results"][split] = {
                "passed": False, "run_directory": str(run_path),
                "error": type(error).__name__ + ": " + str(error),
            }
    summary["passed"] = all(result["passed"] for result in summary["results"].values())
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
