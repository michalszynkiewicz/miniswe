# Threading And UI Runtime Test Design

## Goal

Define the tests needed to make the current runtime/threading model robust.

This document focuses on:

- REPL responsiveness while work is in progress
- worker-to-UI event routing
- permission prompts
- shell timeout and cancellation flow
- LLM/job lifecycle consistency
- stale UI / stuck-job regressions

It is intentionally about behavior, not implementation details.

## Scope

The tests below cover:

- REPL mode
- the dedicated LLM worker
- the supervised shell lane
- the generic tool worker pool
- permission prompts routed through the UI thread

They do not try to validate:

- exact colors or visual styling
- terminal-emulator-specific quirks
- pixel-perfect layout

## Test Layers

We should use three layers of tests.

### 1. Deterministic Runtime Tests

These are the highest-value tests for correctness.

They should avoid real terminals and avoid real network calls.

They validate:

- event ordering
- cancellation behavior
- permission request/resume logic
- shell timeout control flow
- unified job state transitions

These tests should be the backbone of the suite.

### 2. REPL State-Machine Tests

These run one layer above runtime tests.

They should exercise:

- `App`
- `AppEvent`
- modal prompt flow
- active-job state
- background key handling while jobs are active

These are still deterministic, but closer to user-visible behavior.

### 3. PTY / End-To-End REPL Tests

These are expensive and somewhat flaky, so we should keep only a small number.

They validate:

- the real REPL process
- terminal input/output behavior
- permission modal appearance and dismissal
- interrupts during running jobs

These should cover only the most important interaction paths.

## Current Risk Areas

The test plan should explicitly cover these known failure classes:

1. Worker blocks waiting for raw stdin permission prompt
2. `edit_file` or another tool hangs and UI only shows `(interrupted)`
3. Shell timeout prompt appears but UI becomes stale or unresponsive
4. Scroll/input behavior changes when a job is running
5. The main UI says "working" after the underlying job is already dead
6. LLM/tool cancellation semantics differ by lane
7. Permission prompts are split across different code paths
8. Message history becomes invalid after interrupted tool phases

## Test Matrix

Each important lane should be covered by the same core checks:

| Behavior | LLM | shell | generic tool | web permission | MCP permission |
|---|---|---|---|---|---|
| start event/state | yes | yes | yes | yes | yes |
| completion event/state | yes | yes | yes | yes | yes |
| interrupt/cancel | yes | yes | yes | n/a | n/a |
| waiting-user state | no | yes | possible | yes | yes |
| UI remains responsive | yes | yes | yes | yes | yes |

## Tests To Add

## A. Permission Flow Tests

### A1. Worker-side web permission uses UI callback path

Type:
- deterministic runtime/repl test

What it proves:
- a worker can request permission through the UI thread
- no raw stdin prompt is used
- after approval, the worker resumes and completes

Setup:
- create `PermissionManager`
- register REPL `AppEvent` sender
- trigger a `web` action from a worker context

Assertions:
- `AppEvent::PermissionRequest` is emitted
- prompt text is correct
- sending `y` resumes the worker
- no blocking `stdin` path is taken

Status:
- should pass now

### A2. Worker-side MCP permission uses UI callback path

Type:
- deterministic runtime/repl test

What it proves:
- MCP follows the same callback path as web

Assertions:
- emits `PermissionRequest`
- `a` stores session approval
- second call does not prompt again

Status:
- likely passes or is close, but should be made explicit

### A3. Shell permission uses the same unified event path

Type:
- deterministic runtime/repl test

What it proves:
- shell permission is not a separate ad hoc preflight path

Assertions:
- shell permission comes through the same worker-to-UI request mechanism
- the same modal handling path is used as web/MCP

Status:
- should fail now

Reason:
- shell is still handled through a separate preflight path in REPL

### A4. Denied permission returns a clean tool result

Type:
- deterministic runtime/repl test

Assertions:
- denial produces a tool result, not a stuck worker
- active-job state is cleared
- no stale modal remains

Status:
- should pass for web after the recent fix
- should be tested explicitly

## B. Shell Supervision Tests

### B1. Shell timeout enters waiting-user state

Type:
- deterministic runtime test

What it proves:
- timeout is evented
- the worker does not block the main loop

Assertions:
- shell emits `TimedOut`
- UI/job state changes to waiting-user

Status:
- should pass now

### B2. Continue after shell timeout resumes the same job

Type:
- deterministic runtime test

Assertions:
- send `Continue`
- job keeps running
- second timeout or eventual completion is emitted correctly

Status:
- should pass now

### B3. Kill after shell timeout terminates the process group

Type:
- deterministic runtime test

Assertions:
- send `Kill`
- shell returns a completed result promptly
- no hanging descendants keep the job alive

Status:
- should pass now

### B4. Background shell command does not wedge the REPL

Type:
- PTY / integration test

Command example:
- a command that backgrounds a child process

Assertions:
- REPL remains responsive
- timeout prompt appears if appropriate
- interrupt/kill works

Status:
- should pass now if shell supervision is correct

This is one of the highest-value PTY tests.

## C. Interrupt And Cancellation Tests

### C1. Interrupting active LLM generation cancels cleanly

Type:
- deterministic runtime/repl test

Assertions:
- cancel flag is observed
- active-job state clears
- completion is marked interrupted/cancelled consistently

Status:
- should probably pass now

### C2. Interrupting active shell job cancels cleanly

Type:
- deterministic runtime/repl test

Assertions:
- worker receives cancellation
- process group is terminated
- UI state clears

Status:
- should pass now

### C3. Interrupting active generic tool cancels cleanly

Type:
- deterministic runtime/repl test

Use case:
- a mock long-running tool job

Assertions:
- job receives cancellation or is marked cancelled consistently
- UI does not remain in "working"

Status:
- may fail now depending on tool path

### C4. Interrupting `edit_file` turns into a real cancellation or timeout result

Type:
- integration test with mocked/stalled LLM backend

Assertions:
- `edit_file` does not hang forever
- user interrupt yields an actual result
- stage logging indicates where it was interrupted

Status:
- should be close after the recent timeout/cancel patch
- needs explicit test coverage

## D. Unified Job State Tests

### D1. Every job lane sets and clears active-job state

Type:
- REPL state-machine test

Lanes:
- LLM
- shell
- generic tool
- plan
- edit_file

Assertions:
- active job is set at start
- cleared on success
- cleared on failure
- cleared on interrupt

Status:
- should partially pass now

### D2. Jobs emit consistent lifecycle events

Type:
- runtime test

Desired event model:
- started
- waiting-user
- completed/failed/cancelled

Assertions:
- every lane follows the same shape

Status:
- should fail now

Reason:
- eventing is still fragmented across `LlmWorkerEvent`, `ShellWorkerEvent`, `oneshot` tool results, and ad hoc REPL logic

This is the right test to drive a unified runtime event model.

### D3. Stale "working" UI does not remain after worker death

Type:
- integration/repl test

Assertions:
- if worker channel closes unexpectedly, UI clears active state
- user sees an error instead of indefinite "working"

Status:
- should be tested explicitly

## E. REPL Responsiveness Tests

### E1. Scroll keys work while LLM is generating

Type:
- REPL state-machine test

Assertions:
- `PageUp/PageDown` and `Up/Down` with empty input still scroll during active LLM wait

Status:
- should pass now

### E2. Scroll keys work while a tool job is running

Type:
- REPL state-machine test

Assertions:
- same as above during generic tool wait

Status:
- should pass now

### E3. Scroll keys work while shell timeout modal is active

Type:
- REPL state-machine test

Assertions:
- background state remains stable
- modal input still wins over other input

Status:
- uncertain, worth testing

### E4. Permission modal closes after response and normal output resumes

Type:
- PTY or REPL state-machine test

Assertions:
- modal appears
- `y`, `n`, and `a` dismiss it
- status line is added to normal output
- modal does not remain visible

Status:
- should pass now, but this needs a stronger test than render-only coverage

## F. Message History / Recovery Tests

### F1. Interrupted tool-call phase is sanitized before next user message

Type:
- context integration test

Assertions:
- dangling assistant tool-call message is dropped
- next user turn produces valid message ordering

Status:
- should pass now

### F2. Interrupt during active job followed by new user guidance recovers cleanly

Type:
- PTY / integration test

Assertions:
- user can interrupt
- type a corrective instruction
- next LLM request is valid
- no template sequencing error

Status:
- should pass after the sanitize fix, but should be proven end-to-end

## G. Logging And Diagnostics Tests

### G1. `edit_file` stage logging appears during internal workflow

Type:
- integration test

Assertions:
- session log contains stage markers such as:
  - `preplan:start`
  - `patch:...`
  - `validate:lsp`

Status:
- should pass now after the recent logging patch

### G2. Timed-out internal non-streaming LLM call returns a clear error

Type:
- LLM integration test with a deliberately stalling backend

Assertions:
- request times out after configured timeout
- tool returns a readable timeout error

Status:
- should pass after the recent timeout patch, but needs explicit test coverage

## Recommended Implementation Order

We should add tests in this order:

1. Permission flow tests
2. Shell timeout and shell interrupt tests
3. Active-job lifecycle tests
4. `edit_file` interrupt/timeout tests
5. Minimal PTY tests for:
   - permission modal
   - shell timeout modal
   - interrupt + recovery
6. Unified runtime event tests that intentionally fail today

This order gives the best ratio of confidence to effort.

## Tests That Should Intentionally Fail Today

These are the most useful TDD targets for the next architectural cleanup:

1. Shell permission uses the same evented permission path as web/MCP
2. All job lanes expose one unified lifecycle event model
3. Cancellation produces one consistent final state/result across all lanes
4. Long-running tools emit runtime/UI progress events, not just logs

These failures would be useful, not noisy. They correspond to real remaining design gaps.

## Minimal PTY Suite

Keep the PTY suite very small.

Recommended PTY tests:

1. Web permission modal appears, accepts `y`, disappears, and job resumes
2. Shell timeout modal appears, accepts `k`, disappears, and job stops
3. Interrupt a running job, type a new user message, and verify the session recovers

Anything beyond that should be justified carefully, because PTY tests are expensive and more brittle.

## Out Of Scope For Automated Tests

Do not spend time trying to automate:

- color fidelity
- exact visual theme
- terminal-emulator-specific mouse behavior
- exact animation timing

Those should be verified manually when needed.

## Exit Criteria

The threading/UI model is in good shape when:

- no worker path can prompt via raw stdin in REPL mode
- no long-running job can wedge the UI event loop
- interrupts behave consistently across LLM, shell, and tool jobs
- active-job state is always cleared correctly
- permission and timeout modals are proven by tests
- at least one PTY recovery test passes reliably
