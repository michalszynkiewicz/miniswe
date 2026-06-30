#!/usr/bin/env python3
"""Extract a replay fixture (context + code state) from a benchmark run.

The "before first fix" moment is where the done-gate first rejected a run: the
agent thought it was done, the end-to-end check failed, and it's about to make
its first fix. Replaying from exactly there — same working tree AND same
conversation context — tests whether the agent recovers from a *real* stuck
state (not a synthetic seed).

Two artifacts, both already on disk for every run:
  * context: the exact LLM request at that moment lives in llm_dumps/*.json
    (full `messages` array incl. system + masked history). We pick the first
    dump whose last message is the gate rejection AND whose task message is the
    ORIGINAL task (skipping gate_context_reset's compacted "[Your earlier
    work...]" summaries, which aren't a natural pre-fix context).
  * code state: miniswe's shadow-git commits the working tree per round
    ("round N — before round N"). We map the chosen dump to its round by
    timestamp and `git archive` that commit's tree.

Usage:
  scripts/replay/extract-fixture.py <run>/00_baseline [out_dir]
  # out_dir defaults to /tmp/replay-fixture
"""
import glob
import json
import os
import subprocess
import sys

# The done-gate's signature phrase — unique to the behavioral gate's rejection
# ("A check that exercises the change end-to-end exited non-zero"). Using this
# instead of the looser "do NOT finish yet" avoids matching refactor/tool
# rejections that happen mid-build (those also say "do NOT ...").
GATE_SIGNATURE = "A check that exercises the change end-to-end"
RESET_MARKERS = ("[Your earlier work", "Still need:")


def is_gate_rejection(msg_content: str) -> bool:
    return GATE_SIGNATURE in msg_content


def is_reset_summary(msg_content: str) -> bool:
    return any(m in msg_content for m in RESET_MARKERS)


def find_fixture_dump(dumps_dir: str):
    """First dump that is a CLEAN (pre-reset) first gate rejection."""
    for f in sorted(glob.glob(os.path.join(dumps_dir, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = d.get("messages", [])
        if len(m) < 2:
            continue
        last = m[-1].get("content") or ""
        task = m[1].get("content") or ""
        if is_gate_rejection(last) and not is_reset_summary(task):
            return f, d
    return None, None


def shadow_git_rounds(sg: str):
    """[(commit, ct, round_int)] newest-first from the per-round shadow commits."""
    out = subprocess.run(
        ["git", f"--git-dir={sg}", "log", "--format=%H %ct %s"],
        capture_output=True, text=True,
    ).stdout
    rounds = []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        commit, ct, subj = parts
        rnd = None
        for tok in subj.replace("—", " ").split():
            if tok.isdigit():
                rnd = int(tok)
                break
        rounds.append((commit, int(ct), rnd))
    return rounds


def map_messages_to_commit(messages, rounds):
    """Map the captured context to its shadow-git tree by ROUND, not time.

    miniswe bumps its round counter once per main-agent turn, and shadow-git
    commits "round N — before round N" at the start of each round. The number
    of assistant turns in the captured history == how many rounds the agent has
    taken, so the tree the gate just checked is "before round = #assistant".
    Timestamps are unreliable (volume mtimes / container-vs-host clock), so we
    count turns. We try the exact round, then ±1 to absorb off-by-one, and
    return whichever commit exists.
    """
    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    by_round = {rnd: commit for commit, _ct, rnd in rounds if rnd is not None}
    for cand in (n_assistant, n_assistant - 1, n_assistant + 1):
        if cand in by_round:
            return by_round[cand], cand
    # fall back to the newest commit
    return rounds[0][0], rounds[0][2]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    base = sys.argv[1].rstrip("/")
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/replay-fixture"
    dumps_dir = os.path.join(base, "llm_dumps")
    sg = os.path.join(base, "miniswe_state", "shadow-git")

    if not os.path.isdir(dumps_dir):
        sys.exit(f"no llm_dumps in {base}")
    if not os.path.isdir(sg):
        sys.exit(f"no shadow-git in {base}")

    dump_path, dump = find_fixture_dump(dumps_dir)
    if not dump:
        sys.exit("no CLEAN (pre-reset) first gate rejection found — try another run")

    rounds = shadow_git_rounds(sg)
    if not rounds:
        sys.exit("shadow-git has no round commits")
    msgs = dump.get("messages", [])
    commit, rnd = map_messages_to_commit(msgs, rounds)

    os.makedirs(out, exist_ok=True)
    tree_dir = os.path.join(out, "tree")
    subprocess.run(["rm", "-rf", tree_dir])
    os.makedirs(tree_dir, exist_ok=True)
    # export the tree at that commit
    # Disable the LFS smudge filter — the shadow repo inherits .gitattributes
    # that route some paths through git-lfs, which isn't available/needed here.
    proc = subprocess.run(
        [
            "git", f"--git-dir={sg}",
            "-c", "filter.lfs.smudge=cat",
            "-c", "filter.lfs.process=",
            "-c", "filter.lfs.required=false",
            "archive", "--format=tar", commit,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"git archive failed for {commit}: {proc.stderr.decode()[:200]}")
    subprocess.run(["tar", "-x", "-C", tree_dir], input=proc.stdout)

    # context: save the messages (the replay's initial conversation_history) and
    # the tools/model params so the replay can reproduce the request shape.
    msgs = dump.get("messages", [])
    ctx = {
        "messages": msgs,
        "tools": dump.get("tools"),
        "model": dump.get("model"),
        "temperature": dump.get("temperature"),
    }
    json.dump(ctx, open(os.path.join(out, "context.json"), "w"), indent=2)

    manifest = {
        "source_run": base,
        "dump": os.path.basename(dump_path),
        "shadow_commit": commit,
        "round": rnd,
        "n_messages": len(msgs),
        "task_preview": (msgs[1].get("content") or "")[:160] if len(msgs) > 1 else "",
        "gate_msg_preview": (msgs[-1].get("content") or "")[:200],
    }
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)

    print("=== fixture extracted ===")
    for k, v in manifest.items():
        print(f"  {k}: {v}")
    print(f"  tree files: {sum(len(fs) for _,_,fs in os.walk(tree_dir))}")
    print(f"  written to: {out}/  (context.json, tree/, manifest.json)")


if __name__ == "__main__":
    main()
