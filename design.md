# minime — A Context-Frugal CLI Coding Agent

*Designed for 64K context windows on consumer hardware. Optimized for Devstral Small 2 (24B) on RTX 3090 + 128GB RAM.*

*Reference hardware: RTX 3090 (24GB VRAM), 128GB system RAM. Q4_K_M weights (~14GB) + Q8_0 KV cache → 64K–90K usable context at 30+ tokens/second generation.*

---

## 1. Design Philosophy

Even at 64K context, you have roughly **1/3rd** the budget of Claude Code (~200K). Every token still matters — but you have enough room to be *comfortable* rather than desperate. The architecture is built around one principle:

> **Assemble exactly the right context for each step — never dump everything in and hope.**

Existing tools (Claude Code, OpenCode, gptme, Aider) were designed for 128K–200K+ windows. They load system prompts, memory files, tool schemas, conversation history, file contents, and command output all at once, relying on compaction when things overflow. At 64K with a 24B model, we can't afford compaction cycles (they waste an inference call) or bloated system prompts — but we have enough room for real code context if we're disciplined.

### Lessons Extracted from Existing Tools

| Tool | Key Innovation | What We Steal | What We Fix |
|------|---------------|---------------|-------------|
| **Claude Code** | CLAUDE.md, sub-agents, auto-compaction, ripgrep over embeddings | Layered memory files, grep-first search, task tool pattern | System prompt is ~8K tokens alone; tool schemas add ~2K more. Too heavy. |
| **OpenCode** | LSP diagnostics, multi-provider, Bubble Tea TUI | LSP integration for precise diagnostics, provider flexibility | No special context optimization; assumes large windows. |
| **gptme** | Lessons system, RAG via gptme-rag, context compression, context_cmd | Lessons as contextual guidance, dynamic context generation | RAG adds latency + memory overhead (embedding model); we use lighter approach. |
| **Aider** | Tree-sitter repo map with PageRank, token-budgeted maps | Graph-ranked repo map — the single best context technique for code | Map defaults to 1K tokens; we make it central and scale it to 4–8K. |
| **Manus** | todo.md self-prompting to maintain focus across long tool chains | Scratchpad rewriting at context tail for goal alignment | N/A — we adopt this directly. |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       minime CLI                              │
│                                                              │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────────────┐  │
│  │  Planner  │  │  Executor │  │   Knowledge Engine        │  │
│  │  (Plan    │  │  (Act     │  │   ┌────────────────────┐  │  │
│  │   Mode)   │──│   Mode)   │──│   │ Repo Map (AST)     │  │  │
│  └───────────┘  └───────────┘  │   │ Symbol Index       │  │  │
│       │              │         │   │ Snippet Store      │  │  │
│       ▼              ▼         │   │ Project Profile    │  │  │
│  ┌───────────────────────┐     │   │ Lessons Store      │  │  │
│  │  Context Assembler    │◄────┘   └────────────────────┘  │  │
│  │  (Per-Turn, Budgeted) │                                  │  │
│  └───────────┬───────────┘                                  │  │
│              ▼                                               │  │
│  ┌───────────────────────┐                                  │  │
│  │  LLM Interface        │  llama.cpp / Ollama / vLLM       │  │
│  │  (64K context budget) │                                  │  │
│  └───────────────────────┘                                  │  │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

**A. Context Assembler** — The brain of minime. Every turn, it builds a fresh context from:
1. System prompt (slim, ~2K tokens)
2. Project profile (auto-generated, ~500–800 tokens)
3. User guide (optional, ~500 tokens)
4. Relevant repo map slice (dynamic, 2K–8K tokens)
5. Scratchpad / task state (~500–2K tokens)
6. Retrieved file snippets (as needed, budget-controlled, up to ~16K tokens)
7. Conversation history (summary + last 3–5 raw turns, ~4K–8K tokens)
8. Active lessons (contextual tips, ~500 tokens)
9. Current user message

**B. Knowledge Engine** — Offline indexing that runs once and updates incrementally:
- Tree-sitter AST parse → symbol definitions + references across 40+ languages
- PageRank graph over cross-file dependencies (personalized per task)
- Pre-computed file summaries (one-line per file)
- Project profile auto-generation
- Lessons store (accumulated tips from past sessions)

**C. Planner / Executor** — Two-mode operation:
- **Plan mode**: Read-only. Explores the codebase, asks clarifying questions, produces a task plan written to `.minime/plan.md`. Gets the full 64K budget for exploration since there's no edit overhead.
- **Act mode**: Executes against the plan. Each step gets its own focused context assembled from the plan, scratchpad, and relevant code snippets.

---

## 3. The Knowledge Base System

This is where we invest the most. With a 24B model, **quality of context matters even more than quantity** — the model has less capacity for reasoning over noisy input than frontier models.

### 3.1 Project Profile (`.minime/profile.md`)

Auto-generated on `minime init`. A compressed, high-signal overview that gives the model a mental map of the project before it reads any code:

```markdown
# Project Profile (auto-generated — edit to refine)

## Identity
- Name: my-saas-app
- Language: TypeScript (98%), SQL (2%)
- Framework: Next.js 14 (App Router), Prisma, tRPC
- Package manager: pnpm
- Test runner: vitest
- Build: `pnpm build` | Test: `pnpm test` | Lint: `pnpm lint`

## Architecture (3 sentences max)
Monorepo with apps/web (Next.js frontend), packages/api (tRPC router),
and packages/db (Prisma schema + migrations). Auth via NextAuth with
JWT sessions. State management through React Query + tRPC.

## Key Conventions
- All API routes in packages/api/src/routers/
- DB schema in packages/db/prisma/schema.prisma
- Feature components in apps/web/src/features/{feature}/
- Barrel exports from each package index.ts

## Entry Points
- apps/web/src/app/layout.tsx (root layout)
- packages/api/src/root.ts (API router root)
- packages/db/src/index.ts (DB client export)

## Common Pitfalls
- Always use `ctx.db` for database access, never import prisma directly
- tRPC routers must be registered in root.ts to be accessible
- Run `pnpm db:generate` after schema changes before testing
```

**How it's generated:**
1. Scan `package.json` / `Cargo.toml` / `go.mod` / `pyproject.toml` for deps + scripts
2. Tree-sitter parse to detect frameworks, patterns, directory structure
3. Read existing README.md, docs/, and any existing CLAUDE.md / AGENTS.md
4. Use the LLM itself (one-shot, offline) to compress findings into the template
5. User reviews and edits — this is the one file worth spending 5 minutes on

**Budget: ~500–800 tokens.** Always loaded. Replaces the bloated CLAUDE.md pattern — research shows small models follow fewer instructions reliably, so we keep this focused on *what the model needs to navigate*, not behavioral rules.

### 3.2 Repo Map (Aider-Style, Enhanced)

The single most important context technique for code agents. Research shows Aider's AST + PageRank approach achieves the lowest context utilization (4.3–6.5%) among all tested agents while preserving architectural awareness through dependency graphs.

**Our implementation:**

```
Phase 1: Parse (on init + file watch)
  tree-sitter → extract definitions (functions, classes, types, interfaces)
                 + references (calls, imports, type usage)
  Supports: Python, JS/TS, Go, Rust, Java, C/C++, Ruby, PHP, and 30+ more

Phase 2: Graph (incremental)
  Build directed graph: file nodes, symbol-reference edges
  Run PageRank with task-aware personalization vector

Phase 3: Budget-Aware Rendering
  Given token budget T and current task context:
    1. Boost rank of files mentioned in task / recent edits / scratchpad
    2. Binary search for max symbols fitting in T tokens
    3. Render as condensed signatures with file paths
    4. Three tiers: full signatures → name-only → omitted
```

**Example output (at 4K token budget):**

```
src/auth/session.ts:
│ export async function createSession(user: User): Promise<Session>
│ export async function validateSession(token: string): Promise<User | null>
│ export function refreshToken(session: Session): string
│ interface SessionConfig { expiryMs: number; refreshWindow: number }

src/api/routers/user.ts:
│ export const userRouter = router({
│   getProfile: protectedProcedure.query(...)
│   updateProfile: protectedProcedure.input(z.object({...})).mutation(...)
│   changePassword: protectedProcedure.input(z.object({...})).mutation(...)

src/db/schema.prisma:
│ model User { id, email, name, passwordHash, sessions Session[] }
│ model Session { id, token, userId, expiresAt, createdAt }

src/middleware/auth.ts:
│ export function withAuth(handler: NextApiHandler): NextApiHandler
│ export function requireRole(role: Role): MiddlewareFn

src/api/routers/index.ts: (name only)
│ appRouter, userRouter, postRouter, adminRouter

src/utils/crypto.ts: (name only)
│ hashPassword, verifyPassword, generateToken
```

**Key enhancements over standard Aider:**
- **Task-aware personalization**: The PageRank personalization vector shifts based on the current task. Working on auth? Auth-related symbols rank higher. This is recomputed per turn.
- **Tiered detail**: Top-ranked files get full signatures with parameter types; mid-ranked get names only; low-ranked are omitted entirely. The tiers adjust to fit the budget.
- **Incremental updates**: File watcher triggers re-parse only for changed files. Graph update is O(changed files), not O(repo). Cached ASTs are mtime-invalidated.
- **Scalable budget**: At 64K context, we can afford 4–8K for the repo map (vs Aider's default 1K), giving the model much richer architectural awareness.

### 3.3 Snippet Store

Pre-indexed, retrievable code snippets for on-demand loading. This is the mechanism for getting actual code into context without reading entire files:

```
.minime/
  index/
    symbols.json      # { symbol: { file, line, kind, signature, deps } }
    graph.json         # adjacency list + PageRank scores
    summaries.json     # { file: "one-line description" }
    snippets/          # chunked at function/class boundaries
      auth__session__createSession.txt
      auth__session__validateSession.txt
      api__routers__user__changePassword.txt
      ...
```

**Chunking strategy**: Split at function/class boundaries using tree-sitter (not arbitrary character counts). Each chunk includes:
- The function/class body with full source code
- Its import context (just the relevant imports, not all of them)
- Type definitions it depends on (inlined if < 200 tokens, referenced if larger)
- Brief comment noting the file path and line range

**Retrieval is graph-based, not embedding-based.** No embeddings means no second model eating your RAM:
1. Extract keywords from user message + scratchpad → e.g. ["session", "validation", "password"]
2. Match against symbol names in the index → `validateSession`, `changePassword`
3. Walk the dependency graph 1 hop out → also pull `Session` type, `hashPassword` utility
4. Assemble snippets within budget → typically 2K–8K tokens for a focused task

**Why no embeddings:** Aider proved that ripgrep + AST beats embeddings in practice — Anthropic made the same discovery and switched Claude Code from Voyage embeddings to grep-based search. Running an embedding model alongside a 24B model on consumer hardware is impractical, and graph-based retrieval is deterministic and instant.

### 3.4 Task Scratchpad (`.minime/scratchpad.md`)

Inspired by Manus's todo.md self-prompting pattern. This file is **rewritten by the model at the end of every turn** and injected at the **tail of context** (exploiting recency bias — the strongest attention region):

```markdown
## Current Task
Fix session expiry bug: tokens not being invalidated after password change

## Plan (step 3 of 5)
[x] 1. Read session.ts — understand token lifecycle
[x] 2. Read changePassword in user.ts — found it calls db.user.update only
[ ] 3. Add invalidateAllSessions() after password update  ← CURRENT
[ ] 4. Write test for session invalidation on password change
[ ] 5. Run tests, verify fix, commit

## Working Memory
- Sessions stored in DB: model Session { id, token, userId, expiresAt }
- Password change: userRouter.changePassword at line 42 of user.ts
- Need: prisma.session.deleteMany({ where: { userId } }) after password update
- Existing test file: tests/auth/session.test.ts (12 tests, all passing)
- No existing invalidation function — need to add one to session.ts

## Files Modified This Session
- (none yet — about to edit src/api/routers/user.ts)

## Errors Encountered
- (none)
```

**Why this works so well for smaller models**: Instead of keeping full conversation history (which would be noisy and expensive to process), we compress all decisions and discoveries into structured state. The model reads the scratchpad and knows exactly where it is, what it learned, and what to do next — even after aggressive context trimming. The structured format also helps smaller models parse the state reliably.

### 3.5 Lessons Store (`.minime/lessons.md`)

Inspired by gptme's lessons system. Accumulated tips from past sessions, loaded on-demand when relevant:

```markdown
## Testing
- Always run `pnpm db:generate` before running tests after schema changes
- Mock the auth context with `createMockContext()` from test-utils
- Integration tests need `DATABASE_URL` env var pointing to test DB

## Common Gotchas
- tRPC mutations need `.input()` before `.mutation()` — order matters
- Prisma's `deleteMany` returns count, not deleted records
- Next.js App Router: server components can't use hooks
```

**Loading strategy**: Not always loaded. Before each turn, keyword-match the user's message + scratchpad against lesson headings. Only inject relevant sections (typically 200–500 tokens). This is the gptme pattern adapted for smaller context.

---

## 4. Context Budget Management

### Token Budget Allocation (64K window)

| Component | Tokens | % | Notes |
|-----------|--------|---|-------|
| System prompt | 2,000 | 3.1% | Role, tool descriptions, operating rules |
| Project profile | 800 | 1.3% | Always loaded |
| User guide | 500 | 0.8% | Optional custom instructions |
| Repo map slice | 5,000 | 7.8% | Task-personalized, tiered detail |
| Active lessons | 500 | 0.8% | Keyword-matched from lessons store |
| Scratchpad | 1,500 | 2.3% | Current task state + working memory |
| Retrieved snippets | 12,000 | 18.8% | On-demand, graph-based retrieval |
| Conversation history | 6,000 | 9.4% | Summary + last 3–5 raw turns |
| Current message | 1,500 | 2.3% | User's request + any pasted content |
| **Available for output** | **34,200** | **53.4%** | Model's reasoning + tool calls |

**Comparison with 32K budget** (original design):

| Component | 32K budget | 64K budget | Improvement |
|-----------|-----------|-----------|-------------|
| Retrieved snippets | 6,000 | 12,000 | 2× more code visible per turn |
| Conversation history | 3,000 | 6,000 | 3–5 turns instead of 2–3 |
| Repo map | 2,000 | 5,000 | Much richer architectural view |
| Output budget | 17,000 | 34,200 | Room for longer edits + reasoning |

The 64K budget is a qualitative leap — you can see enough code to understand multi-file relationships, keep enough conversation to maintain coherence, and give the model enough output room for complex edits with chain-of-thought reasoning.

### Conversation History Strategy

We use a **hybrid approach** — not pure compaction, not pure sliding window:

1. **Last 3–5 turns**: Kept in full (user message + model response + tool results)
2. **Older turns**: Extracted into scratchpad (decisions, discoveries, file paths), then discarded from raw history
3. **Tool results from older turns**: Replaced with one-line summaries:
   `[read src/auth/session.ts: 89 lines, exports createSession, validateSession, refreshToken]`
4. **Never run LLM-based summarization**: This wastes an inference call and produces lossy output with small models. Structured extraction is cheaper and more reliable.

**The scratchpad IS the memory.** If it's not in the scratchpad, it's not remembered. This forces the model to be explicit about what matters.

### Observation Masking

Tool outputs are the biggest context consumers. We apply aggressive post-processing:

1. **File reads**: If the model used `read_symbol`, show only that symbol. If it used `read_file` for a whole file, cap at 4K tokens with a note about truncation. For subsequent turns, replace with a summary line.
2. **Shell output**: Capped at 3K tokens. **Tail-priority** — errors are usually at the end. Show last N lines, with a count of omitted lines.
3. **Search results**: Cap at 20 matches. Show file:line:content for each. Group by file for readability.
4. **Post-use masking**: Once the model has processed a tool result and moved on, the full result is replaced in subsequent turns with a one-line summary.

---

## 5. Tool Design

Minimal tool surface. Claude Code has 8 tools; we have 10. Each is designed to return **pre-filtered, token-efficient** output. Devstral Small 2 is explicitly trained for tool use, so native function calling works well.

### 5.1 Core Tools (Code)

```
read_symbol(name: str, follow_deps?: bool)
  → Returns the definition + immediate context of a specific symbol
  → Surgical precision via tree-sitter — NOT "read whole file"
  → If follow_deps=true, also includes type definitions it depends on
  → Example: read_symbol("validateSession") → that function + its imports + Session type

read_file(path: str, start_line?: int, end_line?: int)
  → Read a file or line range. Falls back for when you need broader context
  → Auto-truncates to 4K tokens with line numbers preserved
  → Git snapshot before every edit session (not per-read)

search(query: str, scope?: str, max_results?: int)
  → ripgrep underneath. Returns matching lines with file:line context
  → scope: "project" (default), a directory path, or "symbols" (searches index)
  → max_results defaults to 20, adjustable

edit(path: str, old_content: str, new_content: str)
  → Search-and-replace edit format (proven most reliable in Aider benchmarks)
  → Validates edit via tree-sitter parse — catches syntax errors before writing
  → Git snapshot before first edit in session, git diff shown after edit
  → Returns: confirmation + the edited region with 3 lines of surrounding context

shell(command: str, timeout?: int)
  → Execute shell command with output capture
  → Auto-truncation: last 3K tokens (tail-priority for error visibility)
  → Blocklist for dangerous commands; confirmation prompt for destructive ones
  → Default timeout: 60s (prevents runaway builds)

task_update(scratchpad_content: str)
  → Model rewrites the scratchpad after each substantive step
  → Ensures structured state persists across context resets
  → Validated: must contain ## Current Task and ## Plan sections

diagnostics(path?: str)
  → LSP diagnostics for a file or project (if LSP configured)
  → Returns: errors + warnings with file:line:message
  → Falls back to linter output if no LSP available
```

### 5.2 Web Tools

```
web_search(query: str, max_results?: int)
  → DuckDuckGo search, returns title+URL+snippet for top 5 results
  → ~200 tokens total — model reads snippets to decide if full fetch needed
  → Backend configurable: DuckDuckGo (default) or self-hosted SearXNG

web_fetch(url: str, selector?: str)
  → Fetch URL and extract main content as clean markdown
  → Primary backend: Jina Reader API (r.jina.ai/{url}) — free, no key needed
  → Fallback: local trafilatura/readability extraction (fully offline)
  → Auto-truncated to 4K tokens; CSS selector narrows extraction
  → Cached per session to avoid re-fetching

docs_lookup(library: str, topic?: str)
  → Search local llms.txt cache for library documentation
  → Returns relevant section only, keyword-matched (~500–2K tokens)
  → Populated by `minimedocs add` command — instant, no network
```

### 5.2 Tool Schema Compression

Standard JSON Schema tool definitions consume ~200–300 tokens each. At 7 tools, that's ~1.5K–2K tokens just for schemas. We compress to natural language:

```
Tools available:
- read_symbol(name, follow_deps?) → source code of a specific function/class/type
- read_file(path, start_line?, end_line?) → file contents (or line range)
- search(query, scope?, max_results?) → grep matches with file:line context
- edit(path, old, new) → replace old text with new in file (validated)
- shell(cmd, timeout?) → execute command, return stdout/stderr (tail-truncated)
- task_update(content) → rewrite task scratchpad with current state
- diagnostics(path?) → LSP errors/warnings for file or project
- web_search(query, max_results?) → DuckDuckGo snippets (title+url+snippet)
- web_fetch(url, selector?) → fetch URL as clean markdown (truncated to 4K)
- docs_lookup(library, topic?) → search local docs cache for library info
```

**~350 tokens total** for all tool descriptions. Devstral Small 2 is trained on tool-use data, so natural-language descriptions work well. If the model struggles with a specific tool, we can add a one-shot example in the system prompt for that tool (~100 tokens).

---

## 6. The System Prompt (~2,000 tokens)

```markdown
You are minime, a coding agent operating in a terminal. You work on the
project described in the Profile below.

## How You Work
You operate in a loop: read context → reason → act → update state.
Each turn, you receive a fresh context with the project profile, a
relevant slice of the codebase map, your task scratchpad, and any
code snippets retrieved for the current step.

## Rules
1. Read before writing. Use read_symbol or search to understand code
   before making edits. Never guess at APIs or function signatures.
2. Small, focused edits. One logical change per edit call. The edit
   tool validates syntax — if it rejects your edit, fix the syntax.
3. Update state. After meaningful progress, call task_update to save
   what you learned and what's next. This is your memory between turns.
4. Verify changes. After edits, run tests or type-check. Don't assume
   an edit worked — confirm it.
5. Work step by step. Follow the plan in your scratchpad. Complete one
   step fully before moving to the next.
6. If uncertain, explore. Use search and read_symbol to understand the
   codebase. The repo map shows you the structure — use it to find
   the right files.

## Tools
- read_symbol(name, follow_deps?) → source code of a function/class/type
- read_file(path, start_line?, end_line?) → file contents or range
- search(query, scope?, max_results?) → grep matches with context
- edit(path, old, new) → validated search-and-replace in file
- shell(cmd, timeout?) → execute command, return output
- task_update(content) → rewrite your task scratchpad
- diagnostics(path?) → LSP errors/warnings

## Context You Receive Each Turn
- Project Profile: what this project is, its stack, conventions
- Repo Map: ranked overview of the most relevant code structure
- Scratchpad: your current task state and working memory
- Code Snippets: specific source code relevant to the current step
- Recent Conversation: the last few exchanges for continuity
- Lessons: tips from past sessions (when relevant)

- web_search(query, max_results?) → DuckDuckGo snippets (title+url+snippet)
- web_fetch(url, selector?) → fetch URL as clean markdown (truncated to 4K)
- docs_lookup(library, topic?) → search local docs cache for library info

## Web Access Rules
1. Check local docs first: use docs_lookup before web_search for known libraries
2. Search snippets first: web_search returns snippets — often enough to answer
3. Fetch selectively: only web_fetch if snippets are insufficient
4. Prefer official docs: fetch the official documentation URL over blog posts
5. Cache: fetched pages are cached for the session — don't re-fetch

## Response Format
Think through your approach, then act using tools. After each
meaningful step, call task_update with your updated state.
When the task is complete, summarize what you changed.
```

---

## 7. Workflow: A Complete Session

```
$ minime"fix the bug where sessions aren't invalidated after password change"

┌─ minime ────────────────────────────────────────────────────────┐
│                                                                │
│ Phase 1: PLAN                                                  │
│ Context: system(2K) + profile(0.8K) + repo_map(5K) + msg(0.3K)│
│ = 8.1K input → 56K available for exploration + output          │
│                                                                │
│ Model: "I need to understand session and password change code."│
│                                                                │
│   → search("changePassword") → found in user.ts:42            │
│   → read_symbol("changePassword") → 340 tokens                │
│   → read_symbol("createSession", follow_deps=true) → 520 tok  │
│   → read_symbol("validateSession") → 280 tokens               │
│   → search("invalidate session") → no matches                 │
│                                                                │
│ Model writes plan to .minime/plan.md:                           │
│   1. ✓ Explore session + password code                         │
│   2. Add invalidateUserSessions() to session.ts                │
│   3. Call it from changePassword in user.ts                    │
│   4. Write test in tests/auth/session.test.ts                  │
│   5. Run tests + commit                                        │
│                                                                │
│ Phase 2: ACT — Step 2                                          │
│ Context: system(2K) + profile(0.8K) + map_slice[auth](3K) +   │
│   scratchpad(1K) + snippet[session.ts](1.2K) = 8K input       │
│                                                                │
│   → edit("src/auth/session.ts",                                │
│       "export function refreshToken",                          │
│       "export async function invalidateUserSessions(\n"        │
│       "  userId: string\n"                                     │
│       "): Promise<number> {\n"                                 │
│       "  const { count } = await db.session.deleteMany({\n"    │
│       "    where: { userId }\n"                                │
│       "  });\n"                                                │
│       "  return count;\n"                                      │
│       "}\n\n"                                                  │
│       "export function refreshToken")                          │
│   ✓ Syntax valid. 1 file changed.                              │
│   → task_update(step 2 done, step 3 next)                     │
│                                                                │
│ Phase 2: ACT — Step 3                                          │
│ Context rebuilt with snippet[user.ts changePassword] loaded    │
│                                                                │
│   → edit("src/api/routers/user.ts",                            │
│       "await ctx.db.user.update({",                            │
│       "await invalidateUserSessions(input.userId);\n"          │
│       "    await ctx.db.user.update({")                        │
│   ✓ Syntax valid. 1 file changed.                              │
│   → diagnostics("src/api/routers/user.ts")                     │
│   → Missing import. Auto-fix:                                  │
│   → edit (add import { invalidateUserSessions } from session)  │
│   → task_update(step 3 done)                                   │
│                                                                │
│ Phase 2: ACT — Step 4 (test writing)                           │
│ Context rebuilt with test file + changed files loaded           │
│   → read_file("tests/auth/session.test.ts", 1, 30) (see setup)│
│   → edit (add 2 new test cases)                                │
│   → task_update(step 4 done)                                   │
│                                                                │
│ Phase 2: ACT — Step 5 (verify)                                 │
│   → shell("pnpm test tests/auth/session.test.ts")             │
│   All 14 tests pass (12 existing + 2 new) ✓                   │
│   → shell("git add -A && git diff --cached --stat")           │
│   3 files changed, 28 insertions(+)                            │
│                                                                │
│ ✓ Complete. 7 LLM calls, ~52K total tokens used.               │
│   Added invalidateUserSessions() to session.ts                 │
│   Called it from changePassword mutation                        │
│   Added 2 tests confirming sessions are cleared on pwd change  │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. Knowledge Base Preparation Commands

```bash
# Initialize project knowledge base
minime init
  → Detects language/framework from config files
  → Runs tree-sitter parse, builds symbol index + dependency graph
  → Generates .minime/profile.md (review and edit this!)
  → Imports existing CLAUDE.md / AGENTS.md into .minime/guide.md
  → Creates .minime/ directory structure
  → Reports: "Indexed 347 files, 2,891 symbols, 4,312 cross-references"

# Rebuild index (after major refactors or branch switches)
minimereindex [--full]
  → Incremental by default (only changed files since last index)
  → --full forces complete re-parse and graph rebuild

# Show what minime knows about your project
minimeinfo
  → Displays: profile summary, top-20 symbols by PageRank,
    index stats, context budget breakdown, model config

# Edit project-specific guidance
minimeguide
  → Opens .minime/guide.md in $EDITOR
  → On save: lints for token count, warns if over 500 tokens
  → Tip: "Keep this under 500 tokens. If it's longer, move
    details into .minime/lessons.md for on-demand loading."

# Preview what context a task would assemble
minimecontext "fix the auth bug"
  → Shows exactly what would be sent to the LLM for this prompt
  → Token breakdown by component (profile, map, snippets, etc.)
  → Lists which symbols would be retrieved and why
  → Invaluable for debugging poor model behavior

# Add a lesson from the current session
minimelearn "always run db:generate after schema changes"
  → Appends to .minime/lessons.md under auto-detected category
  → Will be loaded in future sessions when keywords match

# Show current model + hardware config
minimeconfig
  → Model: Devstral-Small-2-24B (Q4_K_M)
  → Context: 65536 tokens (KV cache: Q8_0)
  → VRAM: 14.2GB weights + 8.1GB KV = 22.3GB / 24GB
  → Generation speed: ~33 t/s

# Run in plan-only mode (exploration, no edits)
minimeplan "how should I refactor the auth module?"
  → Explores codebase, produces .minime/plan.md
  → No file modifications allowed
  → Useful for understanding before committing to changes

# Continue from last session
minime--continue
  → Loads last session's scratchpad and plan
  → Resumes from where you left off
```

---

## 9. `.minime/` Directory Structure

```
.minime/
├── profile.md            # Auto-generated project profile (~500-800 tokens)
├── guide.md              # User's custom instructions (keep < 500 tokens)
├── plan.md               # Current active plan (written by planner)
├── scratchpad.md          # Task state (rewritten each turn by model)
├── lessons.md             # Accumulated tips, keyword-searchable
│
├── index/
│   ├── symbols.json       # { name: { file, line, kind, signature, deps[] } }
│   ├── graph.json         # Adjacency list + PageRank scores
│   ├── summaries.json     # { file_path: "one-line description" }
│   ├── file_tree.txt      # Directory listing with brief annotations
│   └── cache/             # Parsed AST cache (mtime-invalidated)
│       ├── src__auth__session.ts.ast
│       └── ...
│
├── snippets/              # Pre-chunked code at function/class boundaries
│   ├── src__auth__session__createSession.txt
│   ├── src__auth__session__validateSession.txt
│   ├── src__api__routers__user__changePassword.txt
│   └── ...
│
└── sessions/              # Session logs for --continue
    ├── 2026-03-18T14-30-fix-sessions.jsonl
    └── 2026-03-17T09-15-add-search.jsonl
```

**What gets committed to git:** `profile.md`, `guide.md`, `lessons.md` — these are project knowledge that benefits the whole team. Everything else (index, snippets, sessions, plan, scratchpad) goes in `.gitignore`.

---

## 10. Key Design Decisions and Rationale

### Why No Embeddings / Vector Search?

1. **Proven inferior in practice.** Anthropic switched Claude Code from Voyage embeddings to ripgrep + agentic search after benchmarks showed better results. Aider made the same discovery.
2. **Hardware budget.** Running an embedding model (even a small one like nomic-embed-text) alongside a 24B model on a single RTX 3090 would compete for VRAM and slow inference.
3. **Determinism.** Symbol-graph search with keyword matching gives identical results every time. Embedding similarity is fuzzy — you might miss the exact function you need.
4. **Speed.** Graph lookup + ripgrep is instant. Embedding search adds ~100ms per query, which compounds over many tool calls.

### Why Per-Turn Context Assembly (Not Sliding Window)?

Sliding windows lose information at boundaries and carry stale context forward. Per-turn assembly lets us:
- Include exactly the symbols relevant to *this* step (not last step's files)
- Shift the repo map personalization as the task progresses
- Drop stale tool outputs entirely instead of compressing them
- Keep a consistent token budget every turn (no gradual bloat)

### Why Structured Scratchpad Over Conversation History?

Full conversation history for 10 turns with tool outputs can consume 30K+ tokens easily. The scratchpad captures the same information — decisions, discoveries, file locations, errors — in ~1.5K tokens. The model doesn't need to see "I searched for X and found Y at line 42" — it needs to see "Y exists at file.ts:42 and does Z." The structured format also helps smaller models parse state reliably.

### Why Two Modes (Plan/Act)?

Small models drift more on long-horizon tasks. The Manus team demonstrated that explicit task tracking (todo.md) prevents goal misalignment. Our two-mode approach extends this:
- **Plan mode** uses the full 64K budget for exploration — read many files, understand the architecture, then write a concrete step-by-step plan
- **Act mode** executes one step at a time, with focused context assembled for each step
- The `plan.md` file acts as external memory between steps — the model doesn't need to "remember" the plan, it reads it fresh each turn

### Why Not Sub-Agents?

Claude Code uses sub-agents (Task tool) to isolate context-heavy operations. This works at 200K with frontier models but is problematic for us:
- Spawning a sub-agent means another model load or concurrent inference — RAM pressure on consumer hardware
- Sub-agents lose main context and get only a task description — at 24B, this often leads to missed context
- Our per-turn assembly achieves similar isolation benefits without the overhead: each step gets fresh, focused context anyway

Instead, we use the **plan + scratchpad** pattern: the plan provides the "sub-agent's task description," and the scratchpad provides the "sub-agent's return value." Same benefits, zero overhead.

### Why Compressed Tool Schemas?

Research shows small models attend to fewer instructions reliably — the instruction-following decay is exponential rather than linear for models under ~70B. Every token of JSON Schema is a token not spent on actual code context. Natural-language tool descriptions work well for models explicitly trained on tool-use data (Devstral Small 2 is).

---

## 11. Implementation Plan

### Language: Rust (recommended) or Go

| Factor | Rust | Go |
|--------|------|-----|
| Startup time | <50ms | <100ms |
| tree-sitter bindings | Excellent (native) | Good (cgo) |
| Single binary | Yes | Yes |
| Memory efficiency | Superior (matters for 128GB system) | Good |
| TUI framework | ratatui (mature) | Bubble Tea (mature) |
| Build complexity | Higher | Lower |
| Community precedent | Aider's repo-map logic is in Python | OpenCode is in Go |

**Recommendation:** Rust if you want maximum performance and memory efficiency (matters when the LLM is already using 22GB+ VRAM). Go if you want faster development iteration. Both produce excellent CLI tools.

### Key Dependencies

- **tree-sitter** + language grammars: AST parsing for 40+ languages
- **ripgrep** (`rg`): Fast search, shelled out for simplicity
- **petgraph** (Rust) or equivalent: Graph data structure + PageRank implementation (~50 lines for basic PageRank)
- **reqwest/ureq** (Rust) or **net/http** (Go): HTTP client for OpenAI-compatible API
- **ratatui** (Rust) or **Bubble Tea** (Go): Terminal UI
- **serde** (Rust) or **encoding/json** (Go): JSON serialization for index files
- **notify** (Rust) or **fsnotify** (Go): File system watcher for incremental re-indexing

### LLM Interface

```toml
# .minime/config.toml
[model]
provider = "llama-cpp"          # or "ollama", "vllm", "openai-compatible"
endpoint = "http://localhost:8080"
model = "devstral-small-2"
context_window = 65536
temperature = 0.15              # Low for code tasks
max_output_tokens = 16384

[hardware]
vram_gb = 24
ram_budget_gb = 80              # For KV cache overflow if needed

[context]
repo_map_budget = 5000          # Tokens for repo map
snippet_budget = 12000          # Max tokens for retrieved code
history_turns = 5               # Raw turns to keep
```

### Recommended llama.cpp Server Config

```bash
# Start the server (run this before minime)
llama-server \
  --model Devstral-Small-2-24B-UD-Q4_K_XL.gguf \
  --ctx-size 65536 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --n-gpu-layers 99 \
  --flash-attn \
  --threads 8 \
  --port 8080 \
  --metrics
```

---

## 12. Comparison: Context Efficiency

| Metric | Claude Code | OpenCode | gptme | Aider | **minime** |
|--------|-------------|----------|-------|-------|-----------|
| Target context | 200K | 128K+ | Flexible | 8K–128K | **64K** |
| System prompt | ~8K | ~4K | ~3K | ~2K | **~2K** |
| Repo awareness | grep-based | LSP diag | RAG (optional) | AST+PageRank | **AST+PageRank** |
| Context strategy | Auto-compaction | Auto-compaction | Compression/RAG | Map + budget | **Per-turn assembly** |
| Memory across turns | Full history → compact | Full history | Compact/summarize | Git-based | **Scratchpad** |
| Tool count | 8 | 8+ | 10+ | 4 | **7** |
| Sub-agents | Yes (Task tool) | Yes (@general) | Yes (subagent mode) | No | **No (plan+scratchpad)** |
| Embedding dependency | None (removed) | None | Optional (chromadb) | None (removed) | **None** |
| Min viable context | ~30K | ~15K | ~8K | ~4K | **~12K** |
| Works well at 64K? | Degraded | Okay | Good | Good | **Optimized** |
| Local 24B model? | Not designed for it | Supports it | Supports it | Supports it | **Built for it** |

---

## 13. Context Compression Layer

Every piece of context passes through a compression pipeline before entering the LLM's context window. No second model required — all transformations are deterministic, instant, and lossless for code semantics.

### 13.1 Compression Pipeline

```
Raw Content → [Strip Format] → [Elide Imports] → [Dedup Types] → [Tier Select] → Compressed
                  -35%            -10%              -15%           variable
```

**Total effective multiplier: ~1.6×** — a 64K window carries ~100K worth of information.

### 13.2 Layer 1: Code Format Stripping

Remove indentation, blank lines, comments, and collapse whitespace before injecting code into context. Tree-sitter guarantees AST equivalence. Research shows SOTA models perform equally or *better* on unformatted code (GPT-4o improved from 66.4% to 71.5% on Python tasks).

```
// BEFORE (47 tokens):
export async function createSession(
  user: User
): Promise<Session> {
  // Create a new session for the authenticated user
  const token = generateToken();
  const session = await db.session.create({
    data: {
      token,
      userId: user.id,
      expiresAt: new Date(Date.now() + SESSION_TTL),
    },
  });
  return session;
}

// AFTER (31 tokens — 34% savings):
export async function createSession(user:User):Promise<Session>{const token=generateToken();const session=await db.session.create({data:{token,userId:user.id,expiresAt:new Date(Date.now()+SESSION_TTL)}});return session;}
```

**Critical rule**: Only the active edit target retains formatting (Tier 3), because the model must produce exact `old_content` matches for the edit tool. Everything else (repo map, retrieved context, history) is compressed.

### 13.3 Layer 2: Structured Context Format

Replace natural language prose with dense key-value notation for system context. Models trained on code parse structured data at least as well as prose — and it's 40–60% shorter.

```
// System prompt uses condensed format:
[PROJECT]
name=my-saas-app|lang=TypeScript(98%),SQL(2%)
fw=Next.js14/AppRouter,Prisma,tRPC|pkg=pnpm
cmd:build=pnpm build|test=pnpm test|lint=pnpm lint
[ARCH]
mono:apps/web→Next.js|packages/api→tRPC|packages/db→Prisma
auth=NextAuth/JWT|state=ReactQuery+tRPC
[CONV]
api=packages/api/src/routers/|schema=packages/db/prisma/schema.prisma
features=apps/web/src/features/{name}/|exports=barrel@index.ts
```

This replaces ~800 tokens of prose with ~320 tokens of structured data.

### 13.4 Layer 3: Tiered Code Representation

Four tiers, selected by the context assembler based on purpose:

| Tier | Format | Tokens/Symbol | Used For |
|------|--------|---------------|----------|
| 0 | Name only | 1–2 | Low-ranked repo map entries |
| 1 | Signature skeleton | 5–15 | Mid-ranked repo map entries |
| 2 | Compressed body (no format) | Original × 0.65 | Retrieved snippets for reading |
| 3 | Full formatted source | Original | Active edit target only |

### 13.5 Layer 4: Import Elision + Type Deduplication

- **Import elision**: Omit standard library imports (the model knows them). Keep only project-internal and third-party imports.
- **Type deduplication**: When multiple snippets reference the same type, define it once in a `[TYPES]` header and reference by name:

```
[TYPES]Session={id:string;token:string;userId:string;expiresAt:Date}
[TYPES]User={id:string;email:string;name:string;passwordHash:string}
// Snippets below just use "Session" and "User" — model knows the shapes
```

### 13.6 Layer 5: History as Diffs

Tool results from prior turns are replaced with structured summaries:

```
// Full result (in the turn it was produced): 89 lines of code
// In subsequent turns (compressed):
[t3]read:src/auth/session.ts→89L,exports:createSession,validateSession,refreshToken;uses:db.session,generateToken
[t4]search:"invalidate session"→0 matches
[t5]edit:src/api/routers/user.ts:42→added invalidateUserSessions() call
```

### 13.7 Net Impact on 64K Budget

| Component | Natural Language | Compressed | Effective Capacity |
|-----------|-----------------|------------|-------------------|
| System prompt | 2,000 tok | 1,200 tok | Same info, 40% less |
| Project profile | 800 tok | 350 tok | Same info, 56% less |
| Repo map (5K budget) | ~40 symbols | ~70 symbols | 75% more coverage |
| Snippets (12K budget) | ~4 functions | ~7 functions | 75% more code |
| Conversation history | 3–5 turns | 6–10 turns | 2× more memory |
| **Effective context** | **64K equivalent** | **~100–110K equivalent** | **1.6× multiplier** |

### 13.8 What We Skip

- **LLMLingua / token pruning**: Needs a second model (GPT-2 or LLaMA-7B) running, competing for RAM. The savings on top of format stripping don't justify the overhead, and smaller target models are sensitive to aggressive pruning.
- **Soft prompt / KV cache compression**: Requires model-specific fine-tuning. Not portable.
- **Semantic compression (LLM-based summarization)**: Wastes an inference call at 24B speeds. Lossy. Our structured extraction is cheaper and more reliable.

---

## 14. Web Search & Documentation Access

A coding agent needs web access for: looking up API docs, checking error messages, finding library usage examples, and reading documentation for unfamiliar dependencies. The key constraint is the same as everything else — **minimize tokens, maximize signal**.

### 14.1 Architecture: Tiered Web Access

```
┌────────────────────────────────────────────────────────┐
│                  Web Access Stack                       │
│                                                        │
│  Tier 1: Offline-First (0 latency, 0 tokens wasted)   │
│  ┌──────────────────────────────────────────────────┐  │
│  │ llms.txt cache  │  Local docs  │  Man pages      │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓ miss                         │
│  Tier 2: Lightweight Search (low latency, few tokens)  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ DuckDuckGo API │ Snippets only │ Top 5 results   │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓ need more                    │
│  Tier 3: Deep Fetch (higher latency, more tokens)      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Jina Reader API │ HTML→Markdown │ Section extract │  │
│  │ — or —                                           │  │
│  │ Local: trafilatura / readability │ No API needed  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 14.2 Tier 1: Offline Documentation Cache

Before hitting the network, check local sources. These are free, instant, and don't consume context tokens until the model actually needs them.

**llms.txt files**: The emerging standard for LLM-friendly documentation. Many libraries now ship a `llms.txt` or `llms-full.txt` at their doc root. minime downloads and caches these on `minime init`:

```bash
minimedocs add https://docs.astro.build/llms.txt
minimedocs add https://orm.drizzle.team/llms.txt
# Cached to .minime/docs/astro.llms.txt, .minime/docs/drizzle.llms.txt
```

**Package-derived docs**: On `minime init`, scan `package.json` / `Cargo.toml` / `go.mod` for dependencies. For known libraries, auto-fetch their `llms.txt` or condensed API references.

**Local man/help pages**: For CLI tools used in build scripts (`pnpm`, `docker`, `git`), the model can shell out to `--help` instead of searching the web.

**Format**: llms.txt files are already compressed markdown. We store them as-is and retrieve relevant sections by keyword match (not the whole file).

### 14.3 Tier 2: Web Search (DuckDuckGo)

When the model needs current information or can't find something locally, it performs a lightweight search. **No API key required.**

```
web_search(query: str, max_results?: int) → search results
  → Uses DuckDuckGo's HTML API via the `duckduckgo-search` library (Rust: ddg-search crate, or shell out)
  → Returns: title + URL + snippet for top 5 results
  → Snippet-only mode by default (no page fetching) — ~200 tokens total
  → Model decides if it needs to fetch full page content (Tier 3)
```

**Why DuckDuckGo**: No API key, no account, no rate limit concerns for low-volume agent use, privacy-preserving. For self-hosters, SearXNG is supported as an alternative backend (single config change).

**Context-efficient return format**:
```
[SEARCH:"next.js app router middleware"]
1. next.js docs: Middleware | Next.js
   nextjs.org/docs/app/building-your-application/routing/middleware
   "Middleware allows you to run code before a request is completed. Based on the incoming request, you can modify the response by rewriting, redirecting..."
2. stackoverflow: How to use middleware in Next.js App Router?
   stackoverflow.com/questions/76245632
   "In Next.js 13+ with App Router, create a middleware.ts file in your project root..."
3. (3 more results)
```

~150–250 tokens for 5 results. The model reads snippets and decides whether to fetch full content.

### 14.4 Tier 3: Page Fetching & Content Extraction

When the model needs the full content of a specific page (API docs, a Stack Overflow answer, a blog post):

```
web_fetch(url: str, selectors?: str) → cleaned page content
  → Fetches URL and extracts main content as markdown
  → Two backends (configurable):
    A) Jina Reader API: https://r.jina.ai/{url} (free tier, best quality)
    B) Local: trafilatura library (pure Python/Rust, no external dependency)
  → Auto-truncation: max 4K tokens of extracted content
  → CSS selector support: fetch only specific sections (e.g., "#api-reference")
```

**Jina Reader** is the recommended default: just prepend `https://r.jina.ai/` to any URL to convert it to clean, LLM-friendly markdown. Free tier, no API key needed for basic use, handles JavaScript-rendered pages. Apache 2.0 licensed.

**Local fallback** with `trafilatura` (Python) or a Rust equivalent — no external API dependency, works fully offline, but handles fewer edge cases (no JS rendering).

**Post-fetch compression**: The fetched content goes through the same compression pipeline as code — strip formatting, extract only the relevant section, truncate to budget.

### 14.5 Tool Definitions

Three new tools added to minime's tool set (total now: 10):

```
Tools:
- web_search(query, max_results?) → DuckDuckGo snippets (title+url+snippet)
- web_fetch(url, selector?) → page content as clean markdown (truncated to 4K)
- docs_lookup(library, topic?) → search local llms.txt cache for library docs
```

**Schema cost**: ~80 additional tokens in the system prompt for all three tools.

### 14.6 Smart Web Access Patterns

The model doesn't just blindly search — minime's system prompt guides efficient web use:

```
## Web Access Rules
1. Check local docs first: use docs_lookup before web_search for known libraries
2. Search snippets first: web_search returns snippets — often enough to answer
3. Fetch selectively: only web_fetch if snippets are insufficient
4. Prefer official docs: when multiple results, fetch the official documentation URL
5. Cache awareness: fetched pages are cached locally for the session
```

### 14.7 Documentation Preparation Command

```bash
# Add documentation sources
minimedocs add <url>              # Fetch and cache llms.txt or page
minimedocs add --package <name>   # Auto-find docs for a dependency
minimedocs list                   # Show cached documentation
minimedocs refresh                # Re-fetch all cached docs

# Example workflow:
minimedocs add https://trpc.io/llms.txt
minimedocs add https://www.prisma.io/docs/llms-full.txt
minimedocs add --package next     # Auto-detects Next.js, fetches docs

# Now the model can use docs_lookup("prisma", "deleteMany") 
# without any web request
```

### 14.8 SearXNG Self-Hosted Alternative

For fully offline/private setups, minime supports SearXNG as a search backend:

```toml
# .minime/config.toml
[web]
search_backend = "searxng"        # or "duckduckgo" (default)
searxng_url = "http://localhost:8080"
fetch_backend = "local"           # or "jina" (default)
```

SearXNG aggregates results from multiple engines (Google, Bing, DuckDuckGo, etc.) through a single local instance, providing better result quality than any single engine while maintaining full privacy.

---

## 15. Future Extensions

### Phase 2: Multi-Session Coordination
When working on large features, run 2–3 minime sessions in git worktrees:
```bash
git worktree add ../project-auth -b feature/auth-refactor
cd ../project-auth && minime"refactor the auth module"
```
Each session is fully isolated with its own scratchpad and plan.

### Phase 3: LSP Deep Integration
Beyond diagnostics, use LSP for:
- Go-to-definition as a tool (more precise than grep for finding implementations)
- Find-references (who calls this function?)
- Rename-symbol (safe refactoring with LSP validation)

### Phase 4: Learning Loop
After each session, offer to extract lessons:
```
minimelearn --from-session
  → "I noticed you always run db:generate after schema changes.
     Save this as a lesson? [y/n]"
```
Builds a project-specific knowledge base that improves over sessions.

### Phase 5: Optional Upgrade Path
When running with a larger model or API (Claude, GPT-4, etc.):
- Auto-detect context window size and adjust budgets
- Enable richer conversation history
- Enable sub-agent delegation for complex tasks
- Same tool interface, same knowledge base — just more room
