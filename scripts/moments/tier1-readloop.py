#!/usr/bin/env python3
"""tier1-readloop — K-sample next-action probe on the uds e2e read/no-op loops.

Moments (real decision points, captured 2026-07-15 from the app-with-deps
todo-skills e2e retry turn, fixture benchmark_results/_fixtures/uds-readloop-121347):
  noop     : last msg is the replace_range no-op guard rejection ("already
             match the content you provided") — the model kept resubmitting
             a byte-identical edit 4x in the source run.
  readloop : last msg is the Nth consecutive identical file-read result —
             the model re-read the same small file 20+ times in the source
             e2e runs (38-51 reads/run).

Variants (pure text transforms of the request JSON — no product code):
  control  : verbatim.
  inband   : the in-band tool-result fix. readloop: the final duplicate read
             result is REPLACED by a refusal (prohibition + prescribed next
             action). noop: every no-op guard rejection is reworded to drop
             "re-read the file" and add an explicit do-NOT-resubmit/re-read
             prohibition.
  compact  : forced context compaction, mirroring compressor.rs
             mask_old_observations with budget 0: every tool observation
             except the newest KEEP_RAW_OBS=3 becomes OBS_PLACEHOLDER;
             guard observations (<=4000 chars containing a GUARD_MARKER)
             are exempt. Simulates "loop detected -> compact now".
  both     : compact, then inband applied on top.

Scoring: first tool call of each sample. Loop-continuation signature is
LOOP-READ (readloop: re-read of the same path with an overlapping range) or
NOOP-REPLAY (noop: a replace_range that would change zero bytes, verified by
applying it to the moment-time reference file). Everything else is an escape
of some kind; EDIT-NEW (byte-changing edit) is the productive one.

NOTE on the KV-cache hypothesis: replay sends each arm as an independent
request, so this probe measures the context-CONTENT effect of compaction
only. A warm vs cold KV cache does not change logits (same tokens in,
numerics aside), so content is the mechanism replay can and does test.
"""

import argparse
import copy
import json
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIX = os.path.join(REPO, "benchmark_results/_fixtures/uds-readloop-121347")

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

# ── faithful mirror of compressor.rs constants ─────────────────────────
OBS_PLACEHOLDER = "[earlier tool output elided to save context]"
KEEP_RAW_OBS = 3
GUARD_MARKERS = [
    "[auto-revert]",
    "[hint]",
    "You are in a loop",
    "You are in an edit↔revert loop",
    "same read/inspection call",
]
GUARD_MAX_CHARS = 4000

NOOP_GUARD_RE = re.compile(
    r"already match the content you provided — \s*nothing changed\..*?"
    r"(?=\n\n\[CURRENT STATE\]|\Z)",
    re.DOTALL,
)
NOOP_HDR_RE = re.compile(r"replace_range: lines L(\d+)-(\d+) of (\S+) already match")

READ_REFUSAL = (
    "read refused: you have made this exact read 3 times and the file has NOT "
    "changed — the answer is not in this file's raw content. Do NOT read this "
    "file again. Take a DIFFERENT action now: make an edit, run a check or "
    "validator, or consult the docs."
)

NOOP_GUARD_V2_TAIL = (
    "already match the content you provided — nothing changed. The file ALREADY "
    "contains exactly this text. Do NOT resubmit this edit and do NOT re-read "
    "this file. If the problem persists, its cause is elsewhere: pick a DIFFERENT "
    "file, line range, or approach."
)


def is_guard_observation(content):
    return len(content) <= GUARD_MAX_CHARS and any(m in content for m in GUARD_MARKERS)


def transform_compact(messages):
    """mask_old_observations with raw_budget=0 (mask everything maskable)."""
    out = copy.deepcopy(messages)
    tool_idxs = [i for i, m in enumerate(out) if m.get("role") == "tool"]
    if len(tool_idxs) <= KEEP_RAW_OBS:
        return out
    for i in tool_idxs[: len(tool_idxs) - KEEP_RAW_OBS]:
        c = out[i].get("content")
        if not isinstance(c, str) or c == OBS_PLACEHOLDER or is_guard_observation(c):
            continue
        out[i]["content"] = OBS_PLACEHOLDER
    return out


def transform_inband(messages, kind):
    out = copy.deepcopy(messages)
    if kind == "readloop":
        # replace the final duplicate-read result with the refusal, keeping
        # the appended [CURRENT STATE] block the harness rides on results
        for m in reversed(out):
            if m.get("role") == "tool":
                c = m.get("content") or ""
                idx = c.find("\n\n[CURRENT STATE]")
                m["content"] = READ_REFUSAL + (c[idx:] if idx >= 0 else "")
                break
    else:  # noop: reword every guard rejection in context
        for m in out:
            c = m.get("content")
            if m.get("role") == "tool" and isinstance(c, str) \
                    and "already match the content you provided" in c:
                m["content"] = NOOP_GUARD_RE.sub(NOOP_GUARD_V2_TAIL, c)
    return out


def transform(messages, variant, kind):
    if variant == "control":
        return messages
    if variant == "inband":
        return transform_inband(messages, kind)
    if variant == "compact":
        return transform_compact(messages)
    if variant == "both":
        return transform_inband(transform_compact(messages), kind)
    raise ValueError(variant)


# ── moment introspection ───────────────────────────────────────────────

def last_tool_call_args(messages):
    """(name, args) of the tool call whose result is the LAST message."""
    last = messages[-1]
    tcid = last.get("tool_call_id")
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        for t in m.get("tool_calls") or []:
            if tcid is None or t.get("id") == tcid:
                return t["function"]["name"], json.loads(t["function"]["arguments"])
    return None, {}


def moment_signature(dump, kind):
    """Extract the loop signature the probe scores against."""
    msgs = dump["messages"]
    if kind == "readloop":
        name, args = last_tool_call_args(msgs)
        assert name == "file" and args.get("action") == "read", (name, args)
        return {"path": args["path"],
                "start": args.get("start"), "end": args.get("end")}
    m = NOOP_HDR_RE.search(msgs[-1]["content"])
    assert m, msgs[-1]["content"][:200]
    return {"path": m.group(3), "start": int(m.group(1)), "end": int(m.group(2))}


# ── scoring ────────────────────────────────────────────────────────────

def apply_edit(ref_lines, name, args):
    """Apply replace_range/insert_at to reference lines; None if malformed."""
    try:
        if name == "replace_range":
            s, e = int(args["start"]), int(args["end"])
            content = args.get("content", "")
            if not (1 <= s <= e <= len(ref_lines)):
                return None
            # mirror Rust str::lines(): a trailing newline does not produce
            # a trailing empty line (else a byte-identical edit with a final
            # "\n" misclassifies as EDIT-NEW instead of NOOP-REPLAY)
            if content.endswith("\n"):
                content = content[:-1]
            repl = content.split("\n") if content else []
            return ref_lines[: s - 1] + repl + ref_lines[e:]
        if name == "insert_at":
            a = int(args["after_line"])
            content = args.get("content", "")
            if not (0 <= a <= len(ref_lines)) or not content:
                return None
            return ref_lines[:a] + content.split("\n") + ref_lines[a:]
    except (KeyError, ValueError, TypeError):
        return None
    return None


def classify(resp, kind, sig, ref_dir):
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return {"cat": "PROSE", "detail": (msg.get("content") or "")[:100]}
    tc = tcs[0]["function"]
    name = tc["name"]
    try:
        args = json.loads(tc["arguments"])
    except Exception:
        return {"cat": "MALFORMED", "detail": name}

    detail = ""
    if name == "file":
        action = args.get("action", "")
        path = str(args.get("path", ""))
        detail = f"file({action} {path})"
        if action == "read" and path == sig["path"]:
            s, e = args.get("start"), args.get("end")
            if sig.get("start") is None or s is None \
                    or not (e is not None and (e < sig["start"] or s > sig["end"])):
                # whole-file read or overlapping range = the same loop
                cat = "LOOP-READ" if kind == "readloop" else "READ-TARGET"
                return {"cat": cat, "detail": detail}
            return {"cat": "READ-SAME", "detail": detail}
        if action in ("read", "search"):
            return {"cat": "READ-OTHER", "detail": detail}
        return {"cat": "FILE-OTHER", "detail": detail}

    if name in ("replace_range", "insert_at"):
        path = str(args.get("path", ""))
        detail = f"{name} {path} L{args.get('start', args.get('after_line'))}-{args.get('end', '')}"
        ref_path = os.path.join(ref_dir, path)
        if os.path.exists(ref_path):
            ref_lines = open(ref_path).read().split("\n")
            new = apply_edit(ref_lines, name, args)
            if new is not None and new == ref_lines:
                return {"cat": "NOOP-REPLAY", "detail": detail,
                        "content": args.get("content", "")[:2000]}
            if new is None:
                return {"cat": "EDIT-MALFORMED", "detail": detail}
        return {"cat": "EDIT-NEW", "detail": detail,
                "content": args.get("content", "")[:2000]}

    if name == "revert":
        return {"cat": "REVERT", "detail": json.dumps(args)[:80]}
    if name == "mcp_use":
        return {"cat": "MCP", "detail": json.dumps(args)[:100]}
    if name == "plan":
        return {"cat": "PLAN", "detail": str(args.get("action", ""))}
    if name == "shell":
        return {"cat": "SHELL", "detail": str(args.get("command", ""))[:80]}
    return {"cat": name.upper(), "detail": json.dumps(args)[:80]}


# ── 2-step rollout: synthesize tool results so a step-1 read can be served
#    and the DECISIVE step-2 action scored. Needed because under the compact
#    variants the file content is masked out of context, making a step-1
#    re-read rational recovery rather than loop evidence. ────────────────

ROLLOUT_CONTINUABLE = {"READ-TARGET", "READ-SAME", "READ-OTHER", "LOOP-READ", "MCP"}


def build_result_index(messages):
    """(tool_name, canonical_args) -> observed result content, from history."""
    calls = {}
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                calls[tc["id"]] = tc["function"]
    idx = {}
    for m in messages:
        if m.get("role") != "tool":
            continue
        fn = calls.get(m.get("tool_call_id"))
        c = m.get("content")
        if not fn or not isinstance(c, str) or c == OBS_PLACEHOLDER:
            continue
        try:
            key = (fn["name"], json.dumps(json.loads(fn["arguments"]), sort_keys=True))
        except Exception:
            continue
        idx[key] = c  # latest occurrence wins
    return idx


def synth_read_result(ref_dir, args):
    """Mimic miniswe's read format: '[path: N lines]' header + 'NNN│' lines."""
    path = str(args.get("path", ""))
    ref_path = os.path.join(ref_dir, path)
    if not os.path.exists(ref_path):
        return f"file not found: {path}"
    lines = open(ref_path).read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    s = int(args.get("start") or 1)
    e = int(args.get("end") or len(lines))
    s, e = max(1, s), min(len(lines), e)
    body = "\n".join(f"{i:>4}│{lines[i-1]}" for i in range(s, e + 1))
    return f"[{path}: {len(lines)} lines]\n{body}"


def synth_result(name, args, ref_dir, index):
    """Result content for a step-1 call, or None if not synthesizable."""
    try:
        key = (name, json.dumps(args, sort_keys=True))
    except Exception:
        return None
    if key in index:
        return index[key]
    if name == "file" and args.get("action") == "read":
        return synth_read_result(ref_dir, args)
    if name == "mcp_use":
        # serve the newest validation report seen in history, if this looks
        # like a validate call
        if "validate" in json.dumps(args):
            for (n, _), c in reversed(list(index.items())):
                if n == "mcp_use" and "Validation Report" in c:
                    return c
    return None


def rollout_step(payload, messages, resp):
    """Append the sampled assistant tool call + synthesized result; return
    (new_messages, name, args) or None if terminal."""
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None
    tc = tcs[0]
    try:
        args = json.loads(tc["function"]["arguments"])
    except Exception:
        return None
    return (
        messages
        + [{"role": "assistant", "content": msg.get("content"),
            "tool_calls": [{"id": tc["id"], "type": "function",
                            "function": {"name": tc["function"]["name"],
                                         "arguments": tc["function"]["arguments"]}}]}],
        tc["function"]["name"], args,
    )


def call_llm(payload, timeout=300):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1,
                    help=">1: after a continuable step-1 action (read/validate), "
                         "synthesize its result and score the NEXT action; the "
                         "final step's category is the sample's category")
    ap.add_argument("--moments", default="noop,readloop")
    ap.add_argument("--variants", default="control,inband,compact,both")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/readloop-v1"))
    args = ap.parse_args()

    moment_files = json.load(open(os.path.join(FIX, "moments.json")))
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for mname in args.moments.split(","):
        spec = moment_files[mname]
        dump = json.load(open(os.path.join(FIX, spec["dump"])))
        sig = moment_signature(dump, spec["kind"])
        ref_dir = os.path.join(FIX, spec["ref_dir"])
        # index real observed results from the UNtransformed history, so
        # rollout can serve authentic content even under compact variants
        result_index = build_result_index(dump["messages"])
        print(f"[{mname}] sig={sig}  n_msgs={len(dump['messages'])}")
        for variant in args.variants.split(","):
            key = f"{mname}/{variant}"
            if key in results and len(results[key]) >= args.k:
                print(f"[{key}] cached, skipping")
                continue
            msgs = transform(dump["messages"], variant, spec["kind"])
            payload = {
                "model": dump["model"],
                "messages": msgs,
                "tools": dump["tools"],
                "temperature": dump.get("temperature", 0.15),
                "max_tokens": dump.get("max_tokens", 16384),
                "stream": False,
            }
            if "chat_template_kwargs" in dump:
                payload["chat_template_kwargs"] = dump["chat_template_kwargs"]
            samples = results.get(key, [])
            for i in range(len(samples), args.k):
                steps = []
                cur_msgs = msgs
                try:
                    for step in range(args.steps):
                        p = dict(payload, messages=cur_msgs)
                        resp = call_llm(p)
                        c = classify(resp, spec["kind"], sig, ref_dir)
                        steps.append(c)
                        if step + 1 >= args.steps or c["cat"] not in ROLLOUT_CONTINUABLE:
                            break
                        nxt = rollout_step(p, cur_msgs, resp)
                        if nxt is None:
                            break
                        cur_msgs, tname, targs = nxt
                        obs = synth_result(tname, targs, ref_dir, result_index)
                        if obs is None:
                            break
                        tcid = cur_msgs[-1]["tool_calls"][0]["id"]
                        cur_msgs = cur_msgs + [{"role": "tool",
                                                "tool_call_id": tcid,
                                                "content": obs}]
                except Exception as e:
                    steps.append({"cat": "ERROR", "detail": str(e)[:100]})
                c = dict(steps[-1])
                if len(steps) > 1:
                    c["steps"] = steps
                samples.append(c)
                trail = " → ".join(s["cat"] for s in steps)
                print(f"[{key}] {i+1}/{args.k}: {trail}  {c.get('detail','')}", flush=True)
            results[key] = samples
            json.dump(results, open(results_path, "w"), indent=1)

    print("\n=== SUMMARY ===")
    cats = ["LOOP-READ", "READ-TARGET", "NOOP-REPLAY", "READ-SAME", "READ-OTHER",
            "EDIT-NEW", "REVERT", "MCP", "PLAN", "SHELL", "PROSE", "OTHER"]
    known = set(cats[:-1])
    hdr = f"{'moment/variant':<22}" + "".join(f"{c:>12}" for c in cats)
    print(hdr)
    for key, samples in results.items():
        row = f"{key:<22}"
        n = len(samples)
        for c in cats:
            if c == "OTHER":
                k = sum(1 for s in samples if s["cat"] not in known)
            else:
                k = sum(1 for s in samples if s["cat"] == c)
            row += f"{(f'{k}/{n}' if k else '-'):>12}"
        print(row)


if __name__ == "__main__":
    main()
