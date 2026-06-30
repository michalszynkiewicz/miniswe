import json, glob, sys
run_dir, out = sys.argv[1], sys.argv[2]
SKIP=("[Your earlier work","[Summary of earlier conversation]","[Session summary","[Older conversation turns dropped")
ELIDED="[earlier tool output elided to save context]"
seen=set(); traj=[]
for f in sorted(glob.glob(run_dir+"/llm_dumps/*.json")):
    try: d=json.load(open(f))
    except Exception: continue
    for m in d.get("messages",[]):
        role=m.get("role"); content=m.get("content") or ""; tcs=m.get("tool_calls") or []
        if role=="system": continue
        if any(content.startswith(p) for p in SKIP) or content==ELIDED: continue
        key=("tool",m.get("tool_call_id")) if role=="tool" else (("a",tcs[0].get("id")) if tcs else (role,content[:60]))
        if key in seen: continue
        seen.add(key)
        # keep ONLY the Message-schema fields so Rust's Vec<Message> deserializes
        mm={"role":role}
        if m.get("content") is not None: mm["content"]=m["content"]
        if tcs: mm["tool_calls"]=tcs
        if m.get("tool_call_id"): mm["tool_call_id"]=m["tool_call_id"]
        if m.get("name"): mm["name"]=m["name"]
        traj.append(mm)

# Unique, non-colliding needles planted in EARLY tool results (old band → gets
# compacted). They can't appear in real trajectory content and gemma can't guess
# the tokens. 2 verbatim-critical (exact tokens) + 2 semantic (paraphrasable but
# unique facts). All in TOOL results so no strategy preserves them "for free".
NEEDLES=[
 ("sentinel","// INVARIANT: the integration test pins the literal token GORP_5521 — do not rename it."),
 ("variable","// NOTE: assemble() carries the override in a local named ovr_slot_88; threading it there is what makes smoke pass."),
 ("scratchpad","// DESIGN RULE: whenever the override is active, the agent must DELETE the .miniswe/scratchpad.md file before replying."),
 ("loglevel","// OBSERVABILITY: the chosen override text must be logged at WARN level on startup."),
]
tool_idxs=[i for i,t in enumerate(traj) if t["role"]=="tool"]
# Place needles at NEEDLE_FRAC through the tool messages (default 0.6): old
# enough to be COMPACTED, recent enough to be compacted only ~once near the end
# (not aged out over dozens of rounds) — so strategies can actually differ.
import os
_frac=float(os.environ.get("NEEDLE_FRAC","0.6"))
_base=min(int(len(tool_idxs)*_frac), max(0,len(tool_idxs)-len(NEEDLES)))
spots=tool_idxs[_base:_base+len(NEEDLES)]
for (lbl,text),idx in zip(NEEDLES,spots):
    traj[idx]["content"]=(traj[idx].get("content") or "")+"\n"+text+"\n"
print(f"injected needles at message indices {spots} (of {len(traj)}); first tool at {tool_idxs[0]}, last at {tool_idxs[-1]}")

json.dump(traj, open(out,"w"))
print(f"wrote {len(traj)} messages -> {out}")
