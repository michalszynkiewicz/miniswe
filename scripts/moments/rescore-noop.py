#!/usr/bin/env python3
"""Re-score stored noop-moment samples after the trailing-newline fix in
apply_edit (a byte-identical replace_range whose content ends in "\\n" was
misclassified EDIT-NEW; the real tool's str::lines() treats it as a no-op,
which is exactly why the live guard fired). Patches result files in place."""

import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIX = os.path.join(REPO, "benchmark_results/_fixtures/uds-readloop-121347")

spec = importlib.util.spec_from_file_location("probe", os.path.join(HERE, "tier1-readloop.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

REF = os.path.join(FIX, "workspace-capture/app-with-deps-package/chart/templates/uds-package.yaml")
ref_lines = open(REF).read().split("\n")


def rescore(sample):
    """Returns True if the sample flipped EDIT-NEW -> NOOP-REPLAY."""
    flipped = False
    targets = [sample] + (sample.get("steps") or [])
    for s in targets:
        if s.get("cat") != "EDIT-NEW" or "replace_range" not in s.get("detail", ""):
            continue
        content = s.get("content")
        if content is None or len(content) >= 2000:
            continue  # truncated: cannot re-score safely
        # detail format: "replace_range <path> L<start>-<end>"
        try:
            parts = s["detail"].split()
            rng = parts[2].lstrip("L").split("-")
            args = {"start": int(rng[0]), "end": int(rng[1]), "content": content}
        except (IndexError, ValueError):
            continue
        new = probe.apply_edit(ref_lines, "replace_range", args)
        if new is not None and new == ref_lines:
            s["cat"] = "NOOP-REPLAY"
            flipped = True
    if flipped and sample.get("steps"):
        sample["cat"] = sample["steps"][-1]["cat"]
    return flipped


def main():
    files = sys.argv[1:] or (
        glob.glob(os.path.join(REPO, "benchmark_results/_moments/readloop-v*/results.json"))
        + glob.glob(os.path.join(REPO, "benchmark_results/_moments/readloop-v*/warmcold-noop-*.json"))
    )
    for f in files:
        data = json.load(open(f))
        samples = [s for v in data.values() for s in v] if isinstance(data, dict) else data
        n = sum(rescore(s) for s in samples)
        if n:
            json.dump(data, open(f, "w"), indent=1)
        print(f"{os.path.relpath(f, REPO)}: {n} sample(s) flipped EDIT-NEW -> NOOP-REPLAY")


if __name__ == "__main__":
    main()
