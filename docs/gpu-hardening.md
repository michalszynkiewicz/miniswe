# GPU hardening for bench reliability

Written after the **Xid 79 "GPU has fallen off the bus"** event on 2026-08-27 18:54:09,
which killed the `app-with-deps` e2e run mid-prefill and took the displays with it.

This document is about making the *next* event diagnosable and less likely — not about a
confirmed root cause. There isn't one, but the 2026-08-27 follow-up investigation narrowed
it considerably: VRAM exhaustion, a llama.cpp memory bug, and PSU transient delivery are all
**ruled out or unlikely**, and the surviving hypothesis is an aging used card with the
prefill burst as trigger rather than cause. See "What we actually know" below.

Two verdicts in the first draft have since been **withdrawn** — "idle heat soak: ruled out"
and the appendix's clock-pinning tripwire — and one instruction was wrong
(`/etc/systemd/system-sleep/` is not a search path on this systemd). All three are corrected
in place below.

---

## 1. What we actually know

Hardware: **Palit** RTX 3090 (GA102, subsystem `1569:f278`, VBIOS 94.02.42.00.9D), default
power limit **370 W**, max 420 W, min 100 W. **Bought used in 2025** — its manufacture date
and prior duty history are unknown, and the card exposes neither a serial number nor a board
part number, so they cannot be recovered. PSU is a **be quiet! Straight Power 12 1000 W**,
new in 2025.
Ryzen 9 9950X3D, 123 GB RAM. Fedora 44, NVIDIA **610.57.04** (akmod, built 2026-08-03).
Wayland + KDE, three monitors on the same card. The crash boot ran kernel **7.1.8**; the
machine is on 7.1.9 since the reboot.

### Ruled out, with evidence

| Hypothesis | Verdict | Evidence |
|---|---|---|
| PCIe link / seating / riser | **Ruled out** | AER confirmed active (`_OSC: OS now controls [... AER ...]`); zero AER events in 5 days; link renegotiated clean at Gen4 x16 |
| CPU / RAM instability | **Ruled out** | Zero machine-check exceptions in the crashed boot |
| Core thermal | **Ruled out** | No throttle event; shutdown 98 °C / slowdown 95 °C never approached |
| Idle heat soak | **NOT ruled out** — earlier verdict withdrawn | The 13 W / P8 / 405 MHz baseline in the appendix was measured minutes after a reboot with nothing loaded. **That is not this card's resting state.** With the desktop merely restored it sits at **P0, 9751 MHz, ~106 W, fan 30 %** (item 4a). So the 9-hour "idle" window before the crash was ~106 W with GDDR6X pinned at maximum clock *and* a 68 GB model resident — sustained memory-side thermal load, on a card whose `Memory Current Temp` reads `N/A` |
| Power cap not applied | **Ruled out** | `-pl 200` set 2026-08-25 09:38:50; no suspend or driver reload before the crash |
| Second GPU consumer (ollama) | **Ruled out** | 16.2 s CPU over 4d 17h, no models loaded |
| Driver regression mid-boot | **Unlikely** | Module unchanged since 2026-08-03, survived hundreds of identical runs |
| PSU transient delivery | **Unlikely** | be quiet! Straight Power 12 1000 W, new in 2025. ATX 3.0 platforms are specified to tolerate large sub-millisecond GPU excursions — that spec exists for exactly this failure mode — and a 200 W-capped GPU plus a 9950X3D loads it to roughly a third of capacity, with no electrolytic aging at one year. Residual risk is cabling only: two independent 8-pin runs, not one pigtail |
| VRAM exhaustion / "spill" | **Ruled out** | llama.cpp allocates VRAM statically at load; the server had held that allocation 8h 23m. Linux NVIDIA has no WDDM-style spill-to-RAM — over-allocation fails, it does not degrade. No `cudaMalloc` or `ggml_backend_cuda_buffer_type_alloc_buffer` failure. **Positive control:** this machine's real VRAM-exhaustion signature is in the log on 2026-08-24 13:09:08 (8× `nvidia_drm: Failed to allocate NVKMS memory for GEM object` + kwin `Failed to allocate an egl gbm swapchain graphics buffer`) — entirely absent on 08-27 |
| llama.cpp memory bug | **Ruled out as cause** | llama-server *did* take SIGSEGV (GPF in libc via `libggml-base`), but the `Xid 79` is logged first and names pid 913382 as the victim. A userspace fault cannot electrically remove a PCIe device; once the GPU is off the bus every CUDA handle is garbage and ggml faulting on it is the expected consequence. The unrelated 08-20 llama-server SIGSEGV was a different model, different flags, different crash site (`libllama-server-impl.so`) and its boot logged **zero** Xids |
| Host RAM pressure | **Ruled out** | 68.35 GB of offloaded experts into 123 GB; no OOM killer, no swap storm |
| GPU compute overload | **Not ruled out** | `--n-cpu-moe 40` keeps 40/48 layers on CPU, but that makes the GPU duty cycle a rapid alternation (GPU burst → CPU burst → GPU burst, dozens of times per token). The card died 12 s into a ~96 KB-prompt prefill at `--batch-size 2048` — the heaviest sustained GPU burst in the cycle |

### Load at the time

Only **~80 minutes**, after ~9 hours fully idle (0 LLM requests 08:31 → 17:33):

| run | requests | window | duration |
|---|---|---|---|
| e2e run 1 | 69 | 17:33:59 → 18:30:03 | 56m 04s |
| e2e run 2 | 57 | 18:32:34 → 18:53:57 | 21m 23s |

The card had also completed the 12-run bench queue (00:39–04:34) and two Laguna-S docker
benches (07:59–08:28) that same day without incident.

**Server uptime is the number that matters, not request count.** The llama-server that died
had been up since **10:30:40** — 8 h 23 m — after roughly 13 restarts through the night. Its
exact command line, recovered from the coredump:

```
/app/llama-server --jinja --chat-template-kwargs {"enable_thinking":false} \
  --model .../Laguna-S-2.1-GGUF/UD-Q4_K_XL/Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf \
  --ctx-size 60000 --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-gpu-layers 999 --n-cpu-moe 40 --flash-attn on \
  --threads 16 --threads-batch 32 --batch-size 2048 -np 1 --port 8464 --metrics
```

Note this is **Laguna-S 2.1 (118B-A8B, 68.35 GB)**. The `devstral-small-2` string that
appears in the request dumps is only the model name miniswe puts in the API payload —
llama.cpp ignores it. Do not use the dumps to identify the loaded model; use the coredump
or the server's own log (item 2a).

### Conclusion

**No software-visible cause, and the two obvious software suspects are eliminated.**
PCIe AER was enabled and completely silent all day right up to the instant; `Xid 79` is the
first GPU event of 2026-08-27. The card went from healthy to gone with no warning.

What survived the reboot is worth knowing for next time: `/var/log/journal` **is**
persistent, so the kernel log, the coredump metadata and the full command line were all
recoverable. What was *not* recorded is GPU telemetry — power, clocks, temperature — and
that is the single gap that keeps the transient hypothesis from being testable.

Best-supported reading: **an aging card, with the prefill burst as trigger rather than
cause.** The distinction matters, because the load was not exceptional — the same card had
already completed the 12-run bench queue and two docker benches that same day, and run 1
alone did 69 requests with late prompts comparable to the ~96 KB one that coincided with the
crash. When a load a card routinely survives kills it once, the variable that changed is the
**margin**, not the stimulus. That is a probabilistic failure of a marginal component, which
is the signature of a part on its way out.

Three facts point the same way, and they were established after the first draft of this
document:

- The PSU is close to a best case (item 1 table), so the transient-delivery half of the
  original hypothesis is weak. What remains is a card that can no longer absorb a transient
  it used to.
- The card is a **used Palit** of unknown history — a budget board with weaker VRM and
  memory cooling than the premium 3090s, and close to the archetype of an ex-mining card.
- It has been holding GDDR6X at 9751 MHz around the clock for a year (item 4a) at 30 % fan,
  with the one sensor that would show the consequence reading `N/A` — and item 4a shows that
  is avoidable, not inherent.

`Xid 79` is an *electrical* disappearance — a power-delivery or interconnect failure, not a
memory failure, which would present as artifacts or Xid 13/31/63. So heat is the plausible
cause of the **aging**, not the proximate trigger. It remains a hypothesis at n=1: aging, a
one-off glitch, and a driver edge case are indistinguishable from a single event. Recurrence
is what separates them — aging shortens the interval and responds to a lower cap.

Items 2 and 3 below are the fixes for the measurement gap, item 4 is the targeted
mitigation, and item 4a is the one that addresses the mechanism.

---

## 2. Record GPU telemetry during every run — highest value

**Implemented 2026-08-28** — see "How it is wired now" at the end of this section. The
rationale below is why, and is worth keeping: it is the argument for not switching it off.

Right now every hypothesis is unfalsifiable. With a per-second sample log, the next event
answers itself: you see whether power spiked, whether the core was throttling, and what
the last state before the drop was.

Run this alongside any bench:

```bash
nvidia-smi \
  --query-gpu=timestamp,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,fan.speed,utilization.gpu,utilization.memory,clocks_throttle_reasons.active \
  --format=csv -l 1 \
  > ~/gpu-telemetry-$(date +%F_%H%M).csv &
echo $! > /tmp/gpu-telemetry.pid
```

Stop it with `kill $(cat /tmp/gpu-telemetry.pid)`.

To wire it into the bench harness, wrap `run_one()` in `scripts/queue-*.sh` so the logger
starts with the server and dies with it. **A 24 h log at 1 s is ~20 MB — keep them.**

Sample at **1 s, not 5 s**: a transient event will not appear at 5 s resolution. Include
`power.limit` — that turns "was the cap actually on?" into a column lookup instead of an
argument reconstructed from shell history after the fact.

After an incident, the last ~20 rows before the gap are the whole story.

### How it is wired now

`scripts/bench-gpu.sh` is a sourceable helper with three functions:

| Function | Called from | Does |
|---|---|---|
| `gpu_bench_start "$RESULTS_DIR"` | after the results dir is created | one-shot `nvidia-smi -q` → `gpu-info.txt`, starts the sampler → `gpu-telemetry.csv`, records server provenance → `llama-server.txt` |
| `gpu_telemetry_stop` | the driver's `cleanup` trap | kills the sampler; idempotent |
| `gpu_bench_finish "$RESULTS_DIR"` | the summary block | stops sampling, prints peak power/temp/clock and any throttle reasons, scans `journalctl -k` for Xid and "fallen off the bus" → `gpu-xid.txt` |

Sourced by all five live drivers: `run-benchmark-docker.sh`, `run-benchmark-docker-fast.sh`,
`run-replay-docker.sh`, `run-replay-matrix.sh`, `run-compaction-bench.sh`. The trace lands
**next to the run it belongs to**, not in `~`, so it is still there months later when you go
back to a result and ask what the card was doing.

Two deviations from the sketch above, both deliberate:

* **10 s, not 1 s.** A 6 h run at 1 s is ~20 MB *per run*, and these drivers do 5–6 runs per
  invocation. At 10 s a full matrix costs a few hundred KB. The events actually being hunted
  — a thermal ramp, a power ceiling, a card that stops responding — all last minutes. Set
  `GPU_SAMPLE_INTERVAL=1` when chasing something transient, `0` to disable.
* **`clocks_event_reasons.active`, not `clocks_throttle_reasons.active`.** Same values; the
  throttle spelling is the deprecated alias. The summary decodes the bits that explain a slow
  or dead run: `0x4` SW power cap, `0x8` HW slowdown, `0x40` HW thermal, `0x80` HW power
  brake. `0x1` is just GpuIdle and is not reported.

`power.limit` is not a sampled column — it is static for the run and already in
`gpu-info.txt`, so sampling it would spend a column per row to record one number.

### 2a. Keep the llama.cpp log — currently thrown away

`scripts/run-llama-cuda.sh` launches with `docker run --rm`, and docker's default log
driver is `json-file`, which stores the container's stdout under the container directory.
`--rm` deletes that directory on exit. **The dying server's own log — the last thing it
printed before the GPU vanished — was destroyed by the container teardown.** What the crash
was reconstructed from instead was systemd-coredump metadata, which happens to record the
full command line but nothing the process wrote.

One flag fixes it, and the journal is already persistent here. **Applied 2026-08-27:**

```diff
 exec docker run --rm \
+    --log-driver=journald \
     --gpus all --network=host --ulimit memlock=-1 --cap-add IPC_LOCK \
```

Then `journalctl CONTAINER_NAME=llama-server-<pid>` retrieves it after the fact, surviving
both `--rm` and a reboot. Verified against the real image — a probe container's stdout *and*
stderr were both readable from the journal after `--rm` had deleted the container.

---

## 3. Make the tuning survive reboot and suspend — confirmed footgun

`nvidia-smi -pl` does **not** persist across suspend/resume or reboot. Neither does
`nvidia-smi -lgc` (item 4) — **there is no driver-level flag that makes either one
permanent.** The only mechanism is to re-apply them at boot and after every resume. Your
shell history shows you doing exactly that, by hand, for the power limit:

```
Aug 22 19:04 suspend → 21:41 resume  →  Aug 22 21:55:40  sudo nvidia-smi -pl 200
Aug 24 22:08 suspend → Aug 25 07:55 resume  →  Aug 25 09:38:50  sudo nvidia-smi -pl 200
```

It happened to be in force on 2026-08-27 — but only because you remembered. One forgotten
resume and the next overnight bench runs at **370 W** instead of 200 W.

Two units are needed, because the two events are separate triggers: one oneshot for boot,
one resume-triggered oneshot for wake. Neither covers the other.

The `-lgc` line is shown commented out in both. Uncomment it only once item 4's measurement
has given you a real clock value — the `1500` below is a placeholder, not a recommendation.

### Do **not** use `/etc/systemd/system-sleep/` on this machine

An earlier draft of this document told you to drop a hook there. **That would silently do
nothing.** systemd 259 on Fedora 44 scans exactly one sleep-hook directory, and `/etc` is
not it:

```console
$ strings /usr/lib/systemd/systemd-sleep | grep system-sleep
/usr/lib/systemd/system-sleep
$ strings /usr/lib/systemd/systemd-sleep | grep -c '^/etc/systemd'
0
```

`/usr/lib/systemd/system-sleep/` *is* scanned and already holds the driver's own `nvidia`
hook plus `sysstat.sleep` — but it is vendor-owned, and a hook there needs `chmod +x` or
systemd skips it with no log line and no error. Use a resume-triggered unit instead: it
lives in `/etc`, has no exec-bit trap, orders explicitly, logs to the journal, and
`systemctl status` will tell you whether it actually ran.

### File to create: `/etc/systemd/system/nvidia-resume.service` (after wake)

```ini
[Unit]
Description=Re-apply NVIDIA tuning after resume
After=suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 200
# ExecStart=/usr/bin/nvidia-smi -lgc 210,1500

[Install]
WantedBy=suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
```

### File to create: `/etc/systemd/system/nvidia-tuning.service` (boot-time)

```ini
[Unit]
Description=Apply NVIDIA power limit and clock lock
After=multi-user.target
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 200
# ExecStart=/usr/bin/nvidia-smi -lgc 210,1500

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-tuning.service
sudo systemctl enable nvidia-resume.service          # no --now; it fires on wake
sudo systemctl enable --now nvidia-persistenced      # currently 'disabled'
```

After the next suspend/resume cycle, `systemctl status nvidia-resume.service` shows whether
it fired — which is the whole reason for preferring a unit over a sleep hook.

`-lgc` takes `min,max` in MHz. This card's supported graphics clocks run **210–2100 MHz**
in 523 steps, so `210,1500` means "free to idle all the way down, never boost above 1500".
Locking only the ceiling is the point; do **not** raise the floor, or the card stops
idling at 13 W.

### The persistence-mode wrinkle

`nvidia-persistenced.service` is currently **disabled** on this machine and persistence mode
reads **Disabled**. Without it, the driver tears down GPU state when the last client
detaches — and `-pl` / `-lgc` go with it. That is a second way to silently lose the
settings, independent of suspend and reboot.

It has probably never bitten here, because this 3090 drives three monitors and the
compositor is a permanently attached client. But that is inference, not measurement, and it
would stop being true the moment the card is used headless. `-pm 1` is first in both files
above for this reason; enabling `nvidia-persistenced` makes it stick.

### Verifying

At idle, checking the power limit is enough:

```bash
nvidia-smi --query-gpu=power.limit,persistence_mode --format=csv
# want: 200.00 W, Enabled
```

**The clock lock cannot be verified this way.** At idle the card sits at ~210 MHz, which is
below any ceiling you would set, so `clocks.current.sm` looks identical locked or unlocked.
The check has to happen *under load*: start a bench and confirm `clocks.current.sm` in the
telemetry CSV never exceeds the locked maximum. The `clocks_event_reasons.sw_power_cap`
column also stops being the thing pulling the clock down — `sw_thermal_slowdown` and the
applied-clocks reason take over.

### Undoing it

The two settings undo separately — there is no single reset:

```bash
sudo nvidia-smi -rgc          # release the clock lock (back to full boost range)
sudo nvidia-smi -pl 370       # back to the card's default power limit
```

and comment the `-lgc` lines back out (or `systemctl disable nvidia-tuning.service
nvidia-resume.service`), otherwise the next boot or wake re-applies whatever the units say.

---

## 4. Can we cap the VRAM clock? Would it help?

**Yes, it is supported. Under load it is the wrong lever — but see item 4a: at *idle* there
is now a real problem it might solve.**

Note the two query forms disagree, and the second one is the one that matters:

```
$ nvidia-smi -q -d SUPPORTED_CLOCKS          # compute clocks: one entry
    Memory : 9751 MHz

$ nvidia-smi --query-supported-clocks=memory # lockable clocks: five
    9751 MHz   9501 MHz   5001 MHz   810 MHz   405 MHz

$ nvidia-smi -lmci
    Memory Clock Switching Type: Runtime
```

So `sudo nvidia-smi -lmc <min>,<max>` is available. Under load, do not use it:

- **It costs throughput directly.** LLM decode is memory-bandwidth-bound. 9751 -> 9501 buys
  ~2.6 % less bandwidth for a negligible thermal change; 9751 -> 5001 roughly halves your
  tokens/sec. For a 73 GB MoE already running at 8-9 tok/s that is a serious regression.
- **You cannot verify any benefit.** `temperature.memory` reports `N/A` on this card and
  driver, and there is no ECC or row-remapper data. You would be trading measurable
  throughput for an unmeasurable thermal effect.

**At idle it was worth testing, and it failed.** Item 4a establishes that there *is* an
idle problem — the 4K head pins memory to 9751 MHz and ~70 W around the clock — so the
obvious question was whether `-lmc` could simply force it down. It cannot:

```
$ sudo nvidia-smi -lmc 405,405     # accepted silently, no NVRM message in the log
$ nvidia-smi --query-gpu=pstate,clocks.current.memory,power.draw --format=csv,noheader
P0, 9751 MHz, 103.91 W             # unchanged
```

`-lmc` is a *user* constraint; the display engine's bandwidth requirement is a *floor* the
driver will not violate, because undershooting it corrupts the screen. Item 4a's 4K@30 test
already showed the driver will sit at 405 MHz with all three monitors attached — it just
will not do so while one head needs ~594 MHz of pixel clock. The floor is a function of the
display configuration, and the only way to lower it is to change that configuration.

(`Clocks Event Reasons -> Display Clock Setting: Not Active` is a red herring here. That
flag reports an explicit override, not the implicit display-bandwidth floor.)

> **If you ever run `-lmc`, reset it.** The lock is registered even when it has no visible
> effect, and it outranks nothing only while the display floor is higher. As soon as that
> floor drops — the 4K goes to 30 Hz, a screen blanks, a compute context starts — the clock
> becomes free to obey the lock and fall to 405 MHz, roughly quartering decode throughput
> mid-run. `sudo nvidia-smi -rmc` before any bench.

An earlier version of this section claimed "there is no idle problem to fix — the card
already drops to 405 MHz / 13 W on its own"; that was the same bare-idle-after-reboot misread
that item 1's heat-soak row was corrected for, and it is withdrawn.

### The clock lever that *is* worth considering: core clock, not memory

If the failure was a power transient, the mechanism is the card boosting until it hits the
200 W ceiling, getting pulled back, and boosting again. The power limit is **reactive** —
it corrects after the fact, and that oscillation is exactly where transients live. Locking
the core clock is **proactive**: pick a clock the card can hold inside 200 W and the
oscillation disappears.

Don't guess the value — measure it. Run a normal bench with the telemetry logger from
item 2, then:

```bash
# what core clock did it actually sustain inside the 200 W cap?
awk -F', ' 'NR>1 {print $6}' ~/gpu-telemetry-*.csv   # $6 = clocks.current.sm | sort -n | awk '
  {a[NR]=$1} END {print "median:", a[int(NR/2)], " p90:", a[int(NR*0.9)]}'
```

Then lock slightly below the median:

```bash
sudo nvidia-smi -lgc 0,<median_minus_100>
# undo with:
sudo nvidia-smi -rgc
```

Once you have a value, make it permanent by uncommenting the `-lgc` lines in **both** files
in item 3 — the service for boot and the sleep hook for resume. `-lgc` is exactly as
volatile as `-pl`; a value applied by hand is gone at the next suspend.

Do that only **after** telemetry confirms it does not cost throughput. Treat this as an
experiment, not a default.

---

## 4a. The ~105 W idle floor — cause found and fixed: the 4K head

Until 2026-08-28 this card never dropped its memory clock during normal desktop use:
`P0 / 9751 MHz / ~105 W / fan 30 %` with no llama-server running — ~70 W over what the same
desktop costs without the 4K monitor, continuously, around the clock for the year the card
has been owned. That is the thermal history behind item 1's withdrawn "idle heat soak"
verdict, and it is now **fixed**; the diagnosis is kept because the reasoning was wrong twice
before it was right.

Two hypotheses were tested and **both are refuted** — recorded here so they are not retried:

- **Mixed refresh rates.** The displays ran 3840x2160@60, 1920x1200@59.95 and
  1920x1080@**180**. Dropping the 180 Hz panel to 60 Hz put all three heads at ~60 Hz and
  the card still read `P0 / 9751 MHz / 106.51 W`.
- **Chrome.** Process start times looked like a clean natural experiment — the desktop
  clients started 19:26 and the card read 13 W / P8 at 19:47; Chrome's GPU process started
  22:33 and the card read 106 W / P0 after. Killing Chrome changed nothing: `P0 / 9751 MHz /
  104.62 W` with only `kwin_wayland` and `1password` holding contexts — **both of which were
  already running at 19:47 during the 13 W reading**. Coincidence, not cause. The 19:47
  measurement was minutes after a reboot, before all three heads were being driven.

### What it actually is

The memory clock is pinned by the **pixel-clock bandwidth of the 4K head**, the classic
GA102 mclk-switch avoidance. Measured directly by changing DP-4's mode and watching the
card, restoring it each time:

| DP-4 mode | pstate | mem clock | power |
|---|---|---|---|
| 3840x2160@60 | P0 | 9751 MHz | **104.8 W** |
| 1920x1080@60 | P8 | 405 MHz | **35.4 W** |
| 3840x2160@**30** | P8 | 405 MHz | **36.6 W** |
| 3840x2160@60 (restored) | P0 | 9751 MHz | **104.4 W** |

The 4K@30 row is the one that identifies the mechanism: identical pixel count, half the
bandwidth, clock unpinned. It is the **pixel clock**, not the resolution and not the refresh
rate as such. That is also why the 180 Hz -> 60 Hz change did nothing — it moved the wrong
head. The transition is immediate, within one 3-second sample in both directions.

Reproduce with (each line reversible, ~15 s):

```bash
nvidia-smi --query-gpu=pstate,clocks.current.memory,power.draw --format=csv,noheader -l 3 &
kscreen-doctor output.DP-4.mode.2    # 3840x2160@30
kscreen-doctor output.DP-4.mode.1    # 3840x2160@60  (restore)
```

Note the 35 W floor is the *three heads at low bandwidth* floor, not the 13 W bare idle of
the appendix — those two numbers measure different things.

### There is no software lever — tested

`nvidia-smi -lgc` locks the **graphics** clock, which already idles at 210 MHz and is not
what draws the power. Memory needs `-lmc`, and the two query forms disagree about whether
that is even possible:

```
$ nvidia-smi -q -d SUPPORTED_CLOCKS          # compute clocks: 9751 MHz only
$ nvidia-smi --query-supported-clocks=memory # lockable: 9751 / 9501 / 5001 / 810 / 405
$ nvidia-smi -lmci                           # Memory Clock Switching Type: Runtime
```

The second form governs `-lmc`, so the lock is *available* — but `sudo nvidia-smi -lmc
405,405` leaves the card at `P0 / 9751 MHz / 103.91 W`, unchanged and without a driver
message. The display-bandwidth floor outranks it. See item 4 for the full result and the
mandatory `-rmc` reset.

### Resolved — measured 2026-08-28

**Fixed by moving displays to the iGPU.** The 4K panel went to USB4 port 1 over a
full-featured USB-C cable, and one side panel to the rear HDMI:

```
             before                       after
pstate       P0                           P8
mem clock    9751 MHz                     405 MHz
power        105 W                        26.96 W      <- 78 W recovered
temp         48 C                         37 C
fan          30 %                         0 %
```

The 4K negotiated `3840x2160@60` over DP Alt Mode with no loss of refresh, so a single
USB-C cable is sufficient for it.

Corrected connector map — an earlier draft had the two HDMI connectors backwards:

```
card1  0000:79:00.0  AMD Granite Ridge   DP-1      = USB4 Type-C port 1   (4K, in use)
                                         DP-2      = USB4 Type-C port 2   (free)
                                         HDMI-A-2  = rear HDMI            (in use)
                                         HDMI-A-1  = not wired
card2  0000:01:00.0  NVIDIA RTX 3090     DP-3, DP-4, DP-5, HDMI-A-3
```

The USB4 path is fed internally from the iGPU via the ASMedia ASM4242 host router
(`78:00.0`), not from a DP-IN loopback — which is why those ports appear as *amdgpu*
connectors at all.

**Cable caveat, since this is the usual failure.** DP Alt Mode repurposes the USB-C
SuperSpeed differential pairs; a charge-only or USB 2.0 cable physically lacks them and can
never carry video. Use a USB 3.2 Gen 1/Gen 2 or USB4/Thunderbolt cable. If the panel has an
OSD toggle between "USB 3.0" and "high resolution", choose the latter — USB 3.0 priority
leaves DP only 2 lanes (8.64 Gbps on HBR2), short of the ~12.5 Gbps that 4K60 8-bit needs.

### Remaining: one panel still on the 3090

`DP-3` (1920x1200@59.95) is the last NVIDIA-attached head, and `DP-2` (USB4 port 2) is free.
A **USB-C to DisplayPort cable** moves it — the panel does not need a USB-C input, only the
DP input it already has. That would:

- drop the card from ~27 W to the ~13 W bare idle, since it would drive nothing;
- eliminate the multi-GPU split entirely — single render device, no cross-PCIe buffer
  copies, no primary-selection question, one less thing to glitch on resume;
- **make `nvidia-persistenced` required rather than optional.** With no display client
  holding a context the driver deinitialises and discards `-pl` and `-lgc`. See item 3.

Against it: the iGPU currently composites 10.4 Mpixel across two heads and copes; the third
takes it to 12.7 Mpixel.

### Options, for the record

The two that were considered and are no longer needed:

1. **Drop the 4K to 30 Hz while idle or benching** — `kscreen-doctor output.DP-1.mode.2`
   before a run, `.mode.1` after. Saved the same ~68 W with no hardware change, at the cost
   of an unpleasant desktop.
2. **Accept it** — ~70 W continuous is roughly 610 kWh/year, but see below.

### Why this belongs in a reliability document, not an electricity bill

The card held GDDR6X at 9751 MHz essentially around the clock for the year it has been
owned, at 30 % fan, with `Memory Current Temp` reading `N/A`. On a 3090 — where the GDDR6X
sits on the *back* of the PCB and is the known weak point — that is the most plausible
mechanism for whatever aging produced the Xid 79 in item 1, and it is why the "idle heat
soak" row was flipped from *ruled out* to *not ruled out*.

That history cannot be undone, so item 1's aging hypothesis stands and item 8's recurrence
watch still applies. What changes is that the card is no longer *accumulating* it: 405 MHz
at 37 C with the fan stopped, instead of 9751 MHz at 48 C. If the Xid was heat-driven aging,
this removes the mechanism going forward.

Knock-on already in effect: with the memory clock free to drop, and — should the last panel
move too — no client holding the NVIDIA node between runs, the driver will deinitialise and
discard `-pl` and `-lgc`. Item 3's `-pm 1` and `nvidia-persistenced` stop being
belt-and-braces at that point and become required.

---

## 5. Restart the llama-server between runs — your own rule, violated

**Guarded 2026-08-28** — see "The guard" below. Enforcing it fully is still not possible
from the bench drivers, for the reason given at the end.

`CLAUDE.md` already says this. It was **not** followed on 2026-08-27: the server that died
had been up since 10:30:40 and served both e2e runs back to back. That makes server uptime
an uncontrolled variable in exactly the comparison you were trying to make, and it is the
one procedural thing that was different about the fatal run versus the clean docker benches
earlier the same day.

Fold the restart into the e2e driver the way `queue-loopfix.sh` already does it
(`kill_server` → `start_server` per run), rather than relying on remembering.

### The guard

The bench drivers cannot restart the server — `run-benchmark-docker.sh` says so in its own
header comment, and it is true of all of them: the server is a host-side process they only
point a URL at. So the rule is enforced from both ends instead.

**At start time**, `scripts/run-llama-cuda.sh` refuses to launch on top of a running
`llama-server-*` container and prints the `docker rm -f` line to fix it. All 17 `start-*.sh`
scripts route through this wrapper, so every one inherits the guard. `LLAMA_ALLOW_EXISTING=1`
overrides it. This also replaces a bad failure mode: with `--network=host` a second instance
died on the port bind, but only after the CUDA context and model load were already under way,
buried in llama-server's output.

**At run time**, `gpu_bench_start` writes the serving container's name, start time and uptime
to `llama-server.txt` in the results dir, then greps every previous run's `llama-server.txt`
for that same container name. If the same instance already served an earlier run it prints a
loud warning naming those runs. It does not abort — sometimes reusing a server is the
deliberate choice — but the fact is now recorded in the results directory rather than left to
be reconstructed from shell history afterwards, which is exactly what had to be done on
2026-08-27.

What this still does not cover: a server restarted *between* the arms of a single driver
invocation, and a server that predates the results directory it is being compared against.
Both remain the operator's responsibility.

---

## 6. Fix the e2e's failure reporting

`run_agent()` in `tests/e2e/run-todo-skills-miniswe.sh` does `|| rc=$?` and then
`return 0` unconditionally. A killed or crashed miniswe is therefore **indistinguishable
from a clean finish** in the log — which is what made the first run look like it had
"stopped on its own" when it had been killed.

Propagate the exit status, or at minimum write a sentinel file on non-zero so post-mortems
can tell a crash from a completion.

---

## 7. Not available on this system

- **Fan curve control.** Requires X11 with `Coolbits`; this is a Wayland/KDE session, so
  `nvidia-settings` cannot attach. Not worth chasing — the card idles at fan 0 % / 30 °C
  and the VBIOS curve handles load fine.
- **VRAM junction temperature.** `temperature.memory` is `N/A` on the 3090 with driver
  610.57.04. This is a permanent blind spot on this hardware; it is the main reason a
  memory-side fault can be neither confirmed nor excluded.

---

## 8. If it happens again

A second Xid 79 makes this a hardware verdict rather than a one-off. In order:

1. Pull the last 20 rows of the telemetry CSV before the gap — that is the evidence that
   does not exist for the 08-27 event.
2. Pull the server's own log with `journalctl CONTAINER_NAME=llama-server-<pid>` (needs
   item 2a) and the coredump record with `coredumpctl info llama-server`. Use
   `journalctl -b -1 -k` for the kernel side — after the reboot, plain `journalctl -k`
   reads the *current* boot and returns nothing, which looks like "no logs survived".
3. **Repad and repaste the card.** Highest-value physical action, and the only one that
   addresses a mechanism rather than a symptom. It is a used Palit 3090 of unknown history
   — the backside GDDR6X pads are the known failure point on this board and are very likely
   degraded. While it is open, look for prior-life tells: dust pattern, pad condition,
   whether the shroud has been off before.
4. Check PCIe power cable topology: two **independent** runs, not one daisy-chained pigtail.
   Cheap to verify, but do not expect much — the PSU is a one-year-old 1000 W ATX 3.0 unit
   at roughly a third load, which is close to a best case (see item 1).
5. Reseat the card.
6. Drop the cap to 175 W and see whether the interval between events lengthens.

Recovery from Xid 79 requires a **cold** power cycle — the driver sets
`recovery action = OS Reboot`, and warm reboots frequently fail to re-enumerate the device.

---

## Appendix: idle states, and which one is the baseline

The card has several idle states and they differ by ~90 W. Confusing them is what produced
two wrong diagnoses in item 4a, so the distinction is worth keeping straight.

**Bare idle** — 2026-08-27 19:47, minutes after reboot, nothing loaded, heads not yet driven:

```
pstate  mem_clk   sm_clk   power     temp     fan  util
P8      405 MHz   210 MHz  13-19 W   30-31 C  0 %  0 %
```

**Pinned idle** — 2026-08-27 22:30, desktop restored, *no* llama-server. Three heads on the
3090, one of them 4K60:

```
pstate  mem_clk   sm_clk   power     temp     fan   util
P0      9751 MHz  210 MHz  105.9 W   48 C     30 %  5 %
```

**Current resting idle** — 2026-08-28, after item 4a's fix. One 1920x1200 head left on the
3090; the 4K and one side panel moved to the iGPU:

```
pstate  mem_clk   sm_clk   power     temp     fan  util
P8      405 MHz   210 MHz  26.96 W   37 C     0 %  0 %
```

**Compare new measurements against the third one.** Two superseded readings of this appendix
are worth recording as traps:

- It once called `P0 / 9751 MHz / ~105 W` a clock-pinning *regression* to watch for. At the
  time that was simply the machine's normal state, so the tripwire would have fired
  constantly.
- It then treated bare idle as the baseline, which is what let item 1's "idle heat soak"
  row be marked *ruled out* on a reading taken before the displays were driven.

One further state is reachable: with the last panel moved off the 3090 (item 4a), the card
drives nothing and should return to bare idle at ~13 W. Re-measure here if that happens.
An intermediate `P8 / 405 MHz / ~35 W` was also measured with all three heads still on the
3090 but the 4K dropped to 1080p60 — the control that identified the cause.
