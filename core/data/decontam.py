"""Code-enforced de-contamination: exclude any clip overlapping benchmark/eval sets.

Used by every training/eval phase. A path is contaminated if it lives under a
benchmark/eval directory, has a `test_`-prefixed stem, or is a known same-type
near-duplicate of a benchmark clip.
"""
from __future__ import annotations
from pathlib import Path
from collections.abc import Iterable

CONTAMINATED_DIR_PARTS: tuple[str, ...] = (
    "final_clips",
    "vlm_clips",
    "run_testing",
)

# Same-type near-duplicates of benchmark tricks (different performer/clip,
# same trick TYPE as a POOL-B clip). Keep as explicit substrings on the stem.
NEAR_DUP_STEM_SUBSTRINGS: tuple[str, ...] = (
    # Over-excludes ALL *_in_back_out variants (not only the exact benchmark
    # clip) — intentional, conservative contamination guard.
    "_in_back_out",
    "double_corkscrew_in_back_out",  # already covered by _in_back_out above; kept as explicit benchmark alias
    "tripod_gainer",
)


def is_contaminated(path: Path) -> bool:
    parts = set(path.parts)
    if any(d in parts for d in CONTAMINATED_DIR_PARTS):
        return True
    stem = path.stem
    if stem.startswith("test_"):
        return True
    if any(sub in stem for sub in NEAR_DUP_STEM_SUBSTRINGS):
        return True
    return False


def filter_clean(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if not is_contaminated(p)]


def assert_clean(paths: Iterable[Path]) -> list[Path]:
    """Fail-fast guard: raise if ANY path is contaminated. Use in training/eval
    scripts before they consume a file list."""
    paths = list(paths)
    bad = [str(p) for p in paths if is_contaminated(p)]
    if bad:
        raise AssertionError(
            f"De-contamination violation: {len(bad)} contaminated path(s): {bad[:10]}"
        )
    return paths
