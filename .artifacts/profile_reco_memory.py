#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path("/workdir/playground/reco_algorithm_tests")
SRC_DIR = ROOT_DIR / "src"
DEFAULT_FILES = [
    ROOT_DIR / ".data/current/all_rec.root",
]
DEFAULT_EVENT_COUNTS = [0, 1, 2, 5, 10, 25, 50]


def rss_mb() -> float:
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("Could not read VmRSS from /proc/self/status.")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def child_payload(path: Path, event_count: int, profile: str) -> dict:
    sys.path.insert(0, str(SRC_DIR))

    import ROOT as r  # noqa: F401
    from data_io import RecoDataFile, load_pioneer_libraries

    checkpoints: list[dict] = []

    def mark(label: str) -> None:
        checkpoints.append({"step": label, "rss_mb": rss_mb()})

    mark("python_start")
    load_pioneer_libraries()
    mark("after_pioneer_libraries")

    reco_data = RecoDataFile(str(path), profile=profile)
    mark("after_reco_data_open")

    loaded_entries = 0
    for entry_index in range(min(event_count, reco_data.entries)):
        reco_data.load_entry(entry_index)
        loaded_entries += 1
        mark(f"after_load_entry_{entry_index}")

    return {
        "path": str(path),
        "profile": profile,
        "requested_event_count": event_count,
        "loaded_entries": loaded_entries,
        "entries": reco_data.entries,
        "file_size_mb": file_size_mb(path),
        "checkpoints": checkpoints,
    }


def run_child(path: Path, event_count: int, profile: str) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--path",
        str(path),
        "--event-count",
        str(event_count),
        "--profile",
        profile,
    ]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def summarise(payload: dict) -> dict:
    checkpoints = payload["checkpoints"]
    base = checkpoints[0]["rss_mb"]
    latest = checkpoints[-1]["rss_mb"]
    summary = {
        "file_size_mb": payload["file_size_mb"],
        "entries": payload["entries"],
        "loaded_entries": payload["loaded_entries"],
        "rss_start_mb": base,
        "rss_final_mb": latest,
        "delta_total_mb": latest - base,
    }

    for checkpoint in checkpoints[1:]:
        summary[f"delta_{checkpoint['step']}_mb"] = checkpoint["rss_mb"] - base

    if payload["loaded_entries"] > 0:
        open_delta = next(
            checkpoint["rss_mb"] - base
            for checkpoint in checkpoints
            if checkpoint["step"] == "after_reco_data_open"
        )
        post_load_delta = latest - base
        summary["approx_delta_after_first_open_mb"] = open_delta
        summary["approx_delta_after_loaded_entries_mb"] = post_load_delta
        summary["approx_incremental_per_loaded_entry_mb"] = (
            (post_load_delta - open_delta) / payload["loaded_entries"]
        )
    else:
        summary["approx_delta_after_first_open_mb"] = next(
            checkpoint["rss_mb"] - base
            for checkpoint in checkpoints
            if checkpoint["step"] == "after_reco_data_open"
        )
        summary["approx_delta_after_loaded_entries_mb"] = summary["approx_delta_after_first_open_mb"]
        summary["approx_incremental_per_loaded_entry_mb"] = 0.0

    return summary


def print_report(results: list[dict]) -> None:
    print()
    print("Reco memory profiling report")
    print("=" * 80)
    for result in results:
        path = result["path"]
        summary = result["summary"]
        print(f"\nFile: {path}")
        print(f"  size on disk      : {summary['file_size_mb']:.1f} MB")
        print(f"  entries           : {summary['entries']}")
        print(f"  loaded entries    : {summary['loaded_entries']}")
        print(f"  rss start         : {summary['rss_start_mb']:.1f} MB")
        print(f"  rss final         : {summary['rss_final_mb']:.1f} MB")
        print(f"  total delta       : {summary['delta_total_mb']:.1f} MB")
        print(f"  delta after libs  : {summary['delta_after_pioneer_libraries_mb']:.1f} MB")
        print(f"  delta after open  : {summary['delta_after_reco_data_open_mb']:.1f} MB")
        if summary["loaded_entries"] > 0:
            print(
                f"  approx incremental/event after open : "
                f"{summary['approx_incremental_per_loaded_entry_mb']:.2f} MB"
            )
        print("  checkpoints:")
        for checkpoint in result["payload"]["checkpoints"]:
            print(f"    - {checkpoint['step']:<24} {checkpoint['rss_mb']:.1f} MB")


def print_comparison(results: list[dict]) -> None:
    if len(results) < 2:
        return

    print()
    print("Cross-run comparison")
    print("=" * 80)
    deltas = [result["summary"]["approx_incremental_per_loaded_entry_mb"] for result in results]
    print(
        "  incremental/event after open: "
        f"mean={statistics.mean(deltas):.2f} MB, "
        f"min={min(deltas):.2f} MB, max={max(deltas):.2f} MB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile RAM cost of ROOT/PIONEER reco-file access in fresh subprocesses."
    )
    parser.add_argument(
        "--path",
        action="append",
        default=None,
        help="ROOT file to profile. May be passed multiple times.",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        action="append",
        default=None,
        help="Number of entries to load in each clean subprocess. May be passed multiple times.",
    )
    parser.add_argument(
        "--profile",
        default="realistic",
        help="RecoDataFile collection profile name. Default: realistic",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the raw results JSON.",
    )
    parser.add_argument(
        "--child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.child:
        payload = child_payload(Path(args.path[0]), args.event_count[0], args.profile)
        print(json.dumps(payload))
        return

    paths = [Path(p) for p in (args.path or [str(p) for p in DEFAULT_FILES])]
    event_counts = args.event_count or list(DEFAULT_EVENT_COUNTS)

    results: list[dict] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Profile target does not exist: {path}")
        for event_count in event_counts:
            payload = run_child(path, event_count, args.profile)
            results.append(
                {
                    "path": str(path),
                    "event_count": event_count,
                    "payload": payload,
                    "summary": summarise(payload),
                }
            )

    print_report(results)
    print_comparison(results)

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(results, indent=2))
        print()
        print(f"Wrote JSON results to {out_path}")


if __name__ == "__main__":
    main()
