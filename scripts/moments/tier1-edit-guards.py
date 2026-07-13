#!/usr/bin/env python3
"""tier1-edit-guards — probe two mechanical replace_range guards at the exact
moment the recurring dropped-header bug was born.

Moment: attempt 1 of lazy/run1 (compaction_20260713_152948), dump
req-1783949391-00055-000055.json. msg 66 = the model's replace_range
L297-338 (42-line range) that rewrote the provider loop from memory,
dropping the `if !block.header.is_empty() { push_str(block.header) }` block
and inventing a nonexistent `block.token_estimate` field. The real tool
result (msg 67) reported the LSP error (which the model then fixed) but was
structurally silent about the dropped header (it compiles fine) — the model
never noticed, and the run failed test:FAIL with 10/20 e2e_context failures.

Pre-edit file state reconstructed as pristine (bench SHA cc34d2626) +
add_param(assemble, system_prompt_override) [the only prior mod.rs edit,
msg 64] — verified against the msg-63 in-dump read with 0 mismatches.

Arms (each replaces msg 67's tool result with a synthetic one):
  control    — the REAL result verbatim. Baseline: history says the model
               fixes the LSP error and never notices the header loss.
  diff_echo  — real result + the honest applied diff (removed/added lines)
               + "review: revert if this isn't exactly what you intended".
  cap_reject — the edit is REJECTED: range exceeds a 30-line max; nothing
               applied. Message tells the model to split into smaller
               ranges targeting only lines that change.

Scoring (what the model's next action does about the header):
  REVERT              — calls revert (diff_echo success mode)
  EDIT-RESTORES-HEADER— any edit whose content re-adds block.header
  EDIT-LSP-FIX        — edits only the token_estimate error (control's
                        known-bad path: header stays lost)
  EDIT-OTHER          — other edit (for cap_reject: range size recorded;
                        <=30 = compliant split)
  READ / OTHER        — inspection or non-edit response

Usage: tier1-edit-guards.py [--k 12] [--arms control,diff_echo,cap_reject]
"""

import argparse
import json
import os
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
SCRATCH = "/tmp/claude-1000/-home-michal-dev-miniswe/91153bbc-3489-42aa-88d1-4ad66657da3b/scratchpad"

DUMP = os.path.join(
    REPO,
    "benchmark_results/compaction_20260713_152948__home_michal_models_gemma-4-26B-A4B-it-G/"
    "lazy/run1/llm_dumps/req-1783949391-00055-000055.json",
)
EDIT_MSG = 66   # assistant replace_range call
RESULT_MSG = 67  # its real tool result

CAP = 30

CAP_REJECT_TEXT = (
    "replace_range: range L297-338 spans 42 lines — exceeds the {cap}-line "
    "maximum. Wide ranges force you to retype unchanged code from memory, "
    "which silently drops lines. Nothing was applied. Split this into "
    "several smaller replace_range calls, each covering ONLY the lines that "
    "actually change, editing from the bottom of the file upward so earlier "
    "line numbers stay valid."
).format(cap=CAP)


def build_pre_edit_lines():
    """Pristine + add_param on assemble's signature (verified reconstruction)."""
    pristine = open(os.path.join(SCRATCH, "pristine_mod.rs")).read().splitlines()
    sig_start = next(i for i, l in enumerate(pristine) if l.startswith("pub fn assemble("))
    j = sig_start
    while ")" not in pristine[j]:
        j += 1
    return pristine[:j] + ["    system_prompt_override: Option<String>,"] + pristine[j:]


def build_applied_diff(dump):
    post = build_pre_edit_lines()
    old_lines = post[296:338]  # L297-338 (1-based, inclusive)
    tc = dump["messages"][EDIT_MSG]["tool_calls"][0]["function"]
    new_lines = json.loads(tc["arguments"])["content"].splitlines()
    out = ["Applied diff (lines removed from the file are prefixed '-', lines added '+'):"]
    for l in old_lines:
        out.append(f"-{l}")
    for l in new_lines:
        out.append(f"+{l}")
    out.append(
        "Review the diff above: every '-' line is GONE from the file. If this "
        "is not exactly the change you intended — e.g. a '-' line was removed "
        "unintentionally — call revert (back to rev_1) and re-apply a narrower edit."
    )
    return "\n".join(out)


def load_arm_messages(arm):
    dump = json.load(open(DUMP))
    msgs = dump["messages"][: RESULT_MSG + 1]
    real = msgs[RESULT_MSG]
    content = real["content"]
    if arm == "diff_echo":
        # inject the diff right after the applied/feedback block, before the
        # PLAN STATUS boilerplate, so it reads as part of the tool feedback
        marker = "PLAN STATUS:"
        diff = build_applied_diff(dump)
        if marker in content:
            head, tail = content.split(marker, 1)
            content = head + diff + "\n\n" + marker + tail
        else:
            content = content + "\n\n" + diff
    elif arm == "cap_reject":
        content = CAP_REJECT_TEXT
    msgs[RESULT_MSG] = {**real, "content": content}
    return msgs, dump["tools"], dump["model"], dump.get("temperature", 0.2), dump.get("max_tokens", 8000)


def call_llm(payload, timeout=180):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def classify(resp):
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return {"cat": "NO-TOOL-CALL", "detail": (msg.get("content") or "")[:150].replace("\n", " ")}
    tc = tcs[0]["function"]
    name = tc.get("name", "")
    try:
        args = json.loads(tc.get("arguments", "{}"))
    except Exception:
        return {"cat": "MALFORMED-ARGS", "detail": str(tc)[:150]}
    if name == "revert":
        return {"cat": "REVERT", "detail": json.dumps(args)[:100]}
    if name in ("replace_range", "insert_at", "write_file"):
        content = args.get("content", "") or ""
        s, e = args.get("start"), args.get("end")
        size = (e - s + 1) if isinstance(s, int) and isinstance(e, int) else None
        rng = f"L{s}-{e} (size={size})" if size else f"after={args.get('after_line')}"
        if "block.header" in content:
            return {"cat": "EDIT-RESTORES-HEADER", "detail": f"{name} {rng}"}
        if "token_estimate" in content or "estimate_tokens" in content:
            return {"cat": "EDIT-LSP-FIX", "detail": f"{name} {rng}"}
        return {"cat": "EDIT-OTHER", "detail": f"{name} {rng} content_lines={len(content.splitlines())}"}
    if name in ("file", "code", "show_rev", "check"):
        return {"cat": "READ", "detail": f"{name} {json.dumps(args)[:100]}"}
    return {"cat": f"OTHER:{name}", "detail": json.dumps(args)[:100]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--arms", default="control,diff_echo,cap_reject")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/edit-guards-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for arm in args.arms.split(","):
        messages, tools, model, temperature, max_tokens = load_arm_messages(arm)
        samples = []
        for i in range(args.k):
            payload = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            try:
                resp = call_llm(payload)
                c = classify(resp)
            except Exception as e:
                c = {"cat": "ERROR", "detail": str(e)[:150]}
            samples.append(c)
            print(f"[{arm}] {i + 1}/{args.k}: {c['cat']}  {c.get('detail', '')}", flush=True)
        results[arm] = samples
        json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY ===")
    from collections import Counter

    for arm, samples in results.items():
        cnt = Counter(s["cat"] for s in samples)
        print(f"{arm}: {dict(cnt)}")


if __name__ == "__main__":
    main()
