# Running the bench on Vast.ai (cheap, scriptable, auto-teardown)

This runs the miniswe bench on a rented RTX 3090 instead of your local box — no
Docker, fully scripted: provision → fetch models → run → collect results →
destroy. Picked Vast.ai because it's the only option that lets us *guarantee* a
verified host with enough system RAM for qwen's CPU-expert offload (≥64 GB),
at the lowest price (~$0.20–0.30/hr for a 3090).

## What you get
- A faithful clone of your local 3090 (same 24 GB Ampere), but **uncapped** (no
  200 W limit) and not heating your room.
- gemma runs entirely in VRAM; qwen uses the same experts-on-CPU offload as
  locally (hence the ≥64 GB RAM requirement) — but uncapped, so faster.

---

## One-time account setup

1. **Create an account** at <https://cloud.vast.ai/> and verify your email.

2. **Add credit (PREPAID = your hard spending cap).** Billing → Add Credit. Pay
   by card or crypto. **This is the single most important cost control: you can
   never be charged more than your loaded balance.** For a full gemma+qwen cycle,
   ~$10–15 is plenty; load that and the worst case is bounded there.

3. **Create an API key.** Account → Keys (or <https://cloud.vast.ai/manage-keys/>).
   Copy it. This key drives everything and is also used by the on-VM dead-man's
   switch.

4. **Register an SSH key** so instances accept your connection. Account → SSH
   Keys → paste your `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`). Generate one with
   `ssh-keygen -t ed25519` if you don't have it.

5. **Install the CLI locally and authenticate:**
   ```bash
   pip install --upgrade vastai
   vastai set api-key <YOUR_KEY>
   export VAST_API_KEY=<YOUR_KEY>      # the run script + dead-man's switch read this
   ```

6. *(Optional)* **HuggingFace token** — the gemma/qwen GGUFs (unsloth) are public,
   but a token avoids download rate limits on a fresh box:
   ```bash
   export HF_TOKEN=hf_xxx
   ```

---

## Run it

From the repo root:
```bash
export VAST_API_KEY=...        # required
export HF_TOKEN=hf_...         # optional
scripts/vast/run-on-vast.sh both      # or: gemma | qwen
```

That single command:
1. Finds the cheapest **verified** 3090 with **≥64 GB RAM** and ≥180 GB disk under
   `$MAX_DPH`/hr.
2. Creates it (CUDA devel image), waits for SSH.
3. Rsyncs your **current working tree** (so the Phase-1 / wording changes under
   test go up — including uncommitted edits).
4. Arms the dead-man's switch, then runs `bootstrap-vast.sh` detached on the VM:
   installs rust + rust-analyzer + a CUDA `llama-server`, downloads the model(s),
   starts the server, runs `run-benchmark-native.sh` (Docker-free, 4 runs, 6-check
   best-of-3) for gemma and/or qwen.
5. Streams the VM log; **Ctrl-C is safe** — the instance still tears down.
6. Rsyncs results to `benchmark_results_vast/`, then destroys the instance.

Results land in `benchmark_results_vast/native_*/runN/` (same layout as the
docker bench: `run.log`, `config.toml`, `diff.patch`, `miniswe_state/`).

### Tunables (env vars)
| var | default | meaning |
|---|---|---|
| `RUNS` | 4 | runs per model |
| `MAX_DPH` | 0.50 | don't auto-rent above this $/hr |
| `MIN_RAM_GB` | 64 | system RAM floor (qwen offload) |
| `DISK_GB` | 180 | instance disk |
| `MAX_HOURS` | 12 | on-VM dead-man's-switch timer |
| `GPU_NAME` | RTX_3090 | e.g. `RTX_4090`, or `RTX_A6000` (48 GB → qwen needs no offload; drop `MIN_RAM_GB`) |
| `IMAGE` | nvidia/cuda:12.4.1-devel-ubuntu22.04 | base image (must have nvcc + apt + sshd via Vast) |

---

## Cost control — three independent layers
1. **Prepaid balance** — absolute ceiling; load only what a run costs.
2. **`trap … EXIT`** in `run-on-vast.sh` — destroys on success, error, or Ctrl-C.
3. **On-VM dead-man's switch** — destroys via the Vast API after `MAX_HOURS`
   even if your laptop dies/sleeps mid-run.

> We deliberately do **not** rely on Vast's idle detection — it's threshold-based
> and would misfire on this bursty workload (cargo builds look "idle"). The
> teardown here is deterministic (event + timer + balance), not heuristic.

**Verify teardown after your first run:** `vastai show instances` should list
nothing. If you ever see a stray instance: `vastai destroy instance <id>`.

---

## First-run shakedown notes (these scripts are new, untested against live Vast)
- The CLI offer-query DSL (`gpu_name=`, `cpu_ram>=`, `verified=`) occasionally
  changes — if the search returns nothing, run `vastai search offers --help` and
  adjust `QUERY` in `run-on-vast.sh`.
- The on-VM self-destruct uses `DELETE /api/v0/instances/<id>/` — confirm against
  current Vast API docs; the `trap` + prepaid still protect you if it's off.
- If the chosen `IMAGE` lacks `nvcc`/apt or doesn't expose SSH under Vast, pick a
  Vast-recommended CUDA *devel* template and set `IMAGE=`.
- First run downloads ~61 GB of models and builds llama.cpp (~5–10 min) + the
  driver — budget ~30–45 min of setup before the first bench result.
