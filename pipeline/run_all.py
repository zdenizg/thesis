"""
Pipeline runner — executes phases 1 → 2 → 3 → 4 → 5 → 6A → 6B in order.

Each phase is a standalone Python script that reads the previous phase's
output from disk and writes its own output to its `data/` subdirectory.
This runner invokes each one as a subprocess and stops on the first
non-zero exit status.

Usage
-----
    python pipeline/run_all.py                # run the full pipeline
    python pipeline/run_all.py --phase 3      # run only phase 3
    python pipeline/run_all.py --phase 6b     # run only phase 6B
    python pipeline/run_all.py --from 4       # run phases 4 → end
    python pipeline/run_all.py --from 6a      # run phases 6A → end

Phase IDs: 1, 2, 3, 4, 5, 6a, 6b
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent

# Ordered list of (phase_id, display_name, script_path).
# Script paths are relative to PIPELINE_DIR.
PHASES: list[tuple[str, str, Path]] = [
    ("1",  "PHASE 1 — Structural Analysis",
     PIPELINE_DIR / "phase1" / "scripts" / "phase1_complete_analysis.py"),
    ("2",  "PHASE 2 — Boilerplate Removal and OCR Normalisation",
     PIPELINE_DIR / "phase2" / "scripts" / "phase2_cleaning.py"),
    ("3",  "PHASE 3 — Line-level Metadata Filtering",
     PIPELINE_DIR / "phase3" / "scripts" / "phase3_line_filtering.py"),
    ("4",  "PHASE 4 — Tokenisation, Stopword Removal and Lemmatisation",
     PIPELINE_DIR / "phase4" / "scripts" / "phase4_modeltext.py"),
    ("5",  "PHASE 5 — Corpus Filtering",
     PIPELINE_DIR / "phase5" / "scripts" / "filter_corpus.py"),
    ("6a", "PHASE 6A — Document Aggregation",
     PIPELINE_DIR / "phase6" / "scripts" / "phase6_aggregation.py"),
    ("6b", "PHASE 6B — Modeling Preparation",
     PIPELINE_DIR / "phase6" / "scripts" / "phase6b_modeling_prep.py"),
]

VALID_IDS = [pid for pid, _, _ in PHASES]
SEPARATOR = "=" * 60


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def normalise_id(raw: str) -> str:
    """Lowercase a phase id and validate it against the known list."""
    pid = raw.strip().lower()
    if pid not in VALID_IDS:
        raise SystemExit(
            f"Unknown phase id: {raw!r}. Valid ids: {', '.join(VALID_IDS)}"
        )
    return pid


def select_phases(only: str | None, start: str | None) -> list[tuple[str, str, Path]]:
    """Return the subset of phases to run based on CLI flags."""
    if only is not None:
        target = normalise_id(only)
        return [p for p in PHASES if p[0] == target]

    if start is not None:
        target = normalise_id(start)
        idx = VALID_IDS.index(target)
        return PHASES[idx:]

    return PHASES


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_phase(phase_id: str, name: str, script: Path) -> int:
    """Invoke a single phase script as a subprocess. Returns its exit code."""
    print(SEPARATOR)
    print(name)
    print(SEPARATOR)

    if not script.exists():
        print(f"  ERROR: script not found: {script}")
        return 127

    # Inherit stdout/stderr so the child's progress bars and summary blocks
    # stream live to the user's terminal.
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=PIPELINE_DIR,
    )
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the JFK preprocessing pipeline (phases 1 → 6B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Phase IDs: " + ", ".join(VALID_IDS),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--phase",
        metavar="N",
        help="Run only the given phase (e.g. 3, 6a, 6b).",
    )
    group.add_argument(
        "--from",
        dest="start",
        metavar="N",
        help="Run from the given phase to the end (e.g. --from 4).",
    )
    args = parser.parse_args()

    selected = select_phases(args.phase, args.start)
    if not selected:
        raise SystemExit("No phases selected — nothing to run.")

    for phase_id, name, script in selected:
        exit_code = run_phase(phase_id, name, script)
        if exit_code != 0:
            print()
            print(SEPARATOR)
            print(f"FAILED: phase {phase_id} exited with status {exit_code}")
            print(f"        script: {script.relative_to(PIPELINE_DIR)}")
            print(SEPARATOR)
            sys.exit(exit_code)

    print()
    print(SEPARATOR)
    if args.phase:
        print(f"DONE — phase {args.phase} completed successfully.")
    elif args.start:
        print(f"DONE — phases {args.start} → end completed successfully.")
    else:
        print("DONE — full pipeline completed successfully.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
