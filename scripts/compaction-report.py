#!/usr/bin/env python3
"""compaction-report.py — turn a compaction-matrix result dir into a blog-ready report.

Read-only on the results: parses each cell's stderr `[compaction]` lines, its
container.log (FINAL score, attempts), and wall_s.txt, then writes:
  <results_dir>/report.md      human-readable comparison + per-event stats
  <results_dir>/events.csv     one row per compaction event (for plotting)

It does NOT run anything or touch the model/server — safe to run anytime after
(or during) the matrix.

Usage:
  scripts/compaction-report.py [RESULTS_DIR]
If RESULTS_DIR is omitted, the newest benchmark_results/compaction_* is used.
"""
import csv
import glob
import os
import re
import statistics as st
import sys

COMPACTION_RE = re.compile(
    r"\[compaction\] strategy=(\S+) before_tokens=(\d+) after_tokens=(\d+) "
    r"elided_tokens=(\d+) msgs_before=(\d+) msgs_after=(\d+)"
)
FINAL_RE = re.compile(r"=== FINAL: (\d+)/(\d+) after (\d+) attempt")


def newest_results_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = sorted(glob.glob(os.path.join(here, "..", "benchmark_results", "compaction_*")))
    if not cands:
        sys.exit("No benchmark_results/compaction_* dir found; pass one explicitly.")
    return os.path.abspath(cands[-1])


def read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def parse_cell(cell_dir):
    """Return dict of metrics for one (strategy, run) cell."""
    # Compaction events live in the per-attempt stderr; ANSI codes may surround
    # the line but the regex matches the substring regardless.
    events = []
    for ef in sorted(glob.glob(os.path.join(cell_dir, "stderr_attempt*.txt"))):
        for m in COMPACTION_RE.finditer(read_text(ef)):
            events.append(
                dict(
                    strategy=m.group(1),
                    before=int(m.group(2)),
                    after=int(m.group(3)),
                    elided=int(m.group(4)),
                    msgs_before=int(m.group(5)),
                    msgs_after=int(m.group(6)),
                )
            )
    clog = read_text(os.path.join(cell_dir, "container.log"))
    fm = list(FINAL_RE.finditer(clog))
    pass_n, total_n, attempts = (None, 6, None)
    if fm:
        pass_n, total_n, attempts = (int(fm[-1].group(1)), int(fm[-1].group(2)), int(fm[-1].group(3)))
    rounds = 0
    for lf in glob.glob(os.path.join(cell_dir, "miniswe_state", "logs", "*.log")):
        rounds += read_text(lf).count("[round ")
    wall = read_text(os.path.join(cell_dir, "wall_s.txt")).strip() or "?"
    return dict(events=events, pass_n=pass_n, total_n=total_n, attempts=attempts,
               rounds=rounds, wall=wall)


def fmt(x, nd=1):
    return round(x, nd) if isinstance(x, float) else x


def main():
    rd = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else newest_results_dir()
    strat_dirs = sorted(
        d for d in glob.glob(os.path.join(rd, "*")) if os.path.isdir(d)
        and glob.glob(os.path.join(d, "run*"))
    )
    if not strat_dirs:
        sys.exit(f"No strategy dirs with run*/ under {rd}")

    all_events = []
    lines = [f"# Conversation-compaction strategy benchmark", "",
             f"Results dir: `{rd}`", "",
             "Each cell fires at the *same* `raw_budget` trigger; only the action differs.",
             "`elided` = tokens removed from in-context history per compaction event.", "",
             "| strategy | runs (PASS/6) | mean | rounds | #compact (total) | elided/event (mean / median) | wall (mean s) |",
             "|---|---|---|---|---|---|---|"]
    json_summary = {}

    for sd in strat_dirs:
        strat = os.path.basename(sd)
        cells = [parse_cell(c) for c in sorted(glob.glob(os.path.join(sd, "run*")))]
        results = [f"{c['pass_n']}/{c['total_n']}" if c["pass_n"] is not None else "?/?" for c in cells]
        passes = [c["pass_n"] for c in cells if c["pass_n"] is not None]
        rounds = [c["rounds"] for c in cells]
        per_cell_events = [len(c["events"]) for c in cells]
        elided = []
        for c in cells:
            elided += [e["elided"] for e in c["events"]]
        for ci, c in enumerate(cells, 1):
            for e in c["events"]:
                all_events.append(dict(strategy=strat, run=ci, **{k: e[k] for k in
                    ("before", "after", "elided", "msgs_before", "msgs_after")}))
        walls = []
        for c in cells:
            try:
                walls.append(float(c["wall"]))
            except ValueError:
                pass
        mean_pass = fmt(st.mean(passes)) if passes else "?"
        mean_rounds = fmt(st.mean(rounds)) if rounds else 0
        total_compact = sum(per_cell_events)
        ev_mean = fmt(st.mean(elided)) if elided else 0
        ev_med = fmt(float(st.median(elided))) if elided else 0
        mean_wall = fmt(st.mean(walls)) if walls else "?"
        lines.append(
            f"| {strat} | {' '.join(results)} | {mean_pass} | {mean_rounds} | "
            f"{total_compact} | {ev_mean} / {ev_med} | {mean_wall} |"
        )
        json_summary[strat] = dict(results=results, mean_pass=mean_pass,
            mean_rounds=mean_rounds, total_compactions=total_compact,
            elided_per_event_mean=ev_mean, elided_per_event_median=ev_med,
            mean_wall_s=mean_wall, compactions_per_run=per_cell_events)

    # Degenerate-arm flags (the plan's "flag null comparisons" requirement).
    notes = []
    for strat, s in json_summary.items():
        if s["total_compactions"] == 0:
            notes.append(f"- ⚠️ **{strat}**: compaction NEVER fired in any run — comparison is null for this arm.")
        if "?/?" in s["results"] or "?" == s["mean_pass"]:
            notes.append(f"- ⚠️ **{strat}**: at least one run has no FINAL score (crash / timeout before validation).")
    if notes:
        lines += ["", "## Caveats", *notes]

    lines += ["", f"_{len(all_events)} compaction events across all cells._", ""]

    report = "\n".join(lines) + "\n"
    with open(os.path.join(rd, "report.md"), "w") as f:
        f.write(report)
    with open(os.path.join(rd, "events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "run", "before", "after",
            "elided", "msgs_before", "msgs_after"])
        w.writeheader()
        w.writerows(all_events)
    print(report)
    print(f"Wrote {rd}/report.md and events.csv")


if __name__ == "__main__":
    main()
