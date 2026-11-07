# Safe Headless Execution

How to run buzztr in headless mode (`-y`) safely, preventing unintended filesystem access.

## Built-in Safety

buzztr already enforces:
- **Path jailing** — all file operations are restricted to the project root. Absolute paths and `../` traversal are blocked.
- **Shell allowlist** — common dev commands (cargo, git, npm, etc.) auto-approved, everything else prompted in interactive mode or auto-approved with `-y`.
- **Web query visibility** — each search/fetch shows what's being sent (interactive mode only).

With `-y`, all permissions are auto-approved. The path jail still holds, but shell commands run without confirmation. Use one of the isolation methods below to limit blast radius.

## Option 1: Git Worktree (simplest)

Run in a disposable copy of your repo. If anything goes wrong, delete it.

```bash
# Create a throwaway workspace
git worktree add /tmp/buzztr-job -b buzztr-test
cd /tmp/buzztr-job

# Initialize and run
buzztr init
buzztr -y "fix the failing tests"

# Review what changed
git diff

# If happy, merge back
git checkout main
git merge buzztr-test

# Clean up
git worktree remove /tmp/buzztr-job
git branch -d buzztr-test
```

**What this gives you:** easy rollback, no risk to your working branch. The path jail restricts file access to `/tmp/buzztr-job`.

**What this does NOT give you:** shell command isolation. buzztr can still run arbitrary commands as your user (e.g., `curl`, `rm` outside the project via shell).

## Option 2: Locked-Down User (medium isolation)

Run as a dedicated user with limited filesystem permissions.

```bash
# One-time setup
sudo useradd -m -s /bin/bash buzztr-runner
# Give access to dev tools
sudo -u buzztr-runner bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

# Copy project to runner's home
sudo -u buzztr-runner cp -r /path/to/project /home/buzztr-runner/project

# Run
sudo -u buzztr-runner bash -c '
  cd /home/buzztr-runner/project
  export PATH="$HOME/.cargo/bin:$PATH"
  buzztr init
  buzztr -y "fix the failing tests"
'

# Review and copy results back
sudo -u buzztr-runner git -C /home/buzztr-runner/project diff
```

**What this gives you:** filesystem isolation via Unix permissions. The runner can't read/write files it doesn't own. Shell commands run as the restricted user.

**LLM access:** works out of the box — any user can make HTTP requests to `localhost:8464`.

## Option 3: Docker Container (strongest isolation)

Full sandboxing — filesystem, network, and process isolation.

```dockerfile
# Dockerfile.buzztr
FROM rust:slim

RUN apt-get update && apt-get install -y \
    ripgrep git nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY target/release/buzztr /usr/local/bin/buzztr

WORKDIR /workspace
```

```bash
# Build the container
docker build -t buzztr-runner -f Dockerfile.buzztr .

# Run with project mounted
docker run --rm \
  --network=host \
  -v $(pwd):/workspace \
  -w /workspace \
  buzztr-runner \
  sh -c 'buzztr init && buzztr -y "fix the failing tests"'
```

**What this gives you:** full isolation. Even if the LLM escapes the path jail or runs malicious shell commands, damage is contained to the disposable container.

**LLM access:** `--network=host` gives the container access to `localhost:8464` where llama-server runs. For tighter control, use `--add-host=host.docker.internal:host-gateway` and set `endpoint = "http://host.docker.internal:8464"` in config.

**MCP servers:** if you need MCP, mount the MCP server binaries into the container or install them in the Dockerfile.

## Option 4: CI/CD Pipeline

Run buzztr as a CI step (GitHub Actions, GitLab CI, etc.):

```yaml
# .github/workflows/buzztr.yml
name: buzztr fix
on:
  issues:
    types: [opened]

jobs:
  fix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start LLM server
        run: |
          # Start llama-server or use a cloud API endpoint
          # Set endpoint in .buzztr/config.toml

      - name: Run buzztr
        run: |
          buzztr init
          buzztr -y "fix: ${{ github.event.issue.title }}"

      - name: Create PR
        run: |
          git checkout -b buzztr/fix-${{ github.event.issue.number }}
          git add -A
          git commit -m "fix: ${{ github.event.issue.title }}"
          gh pr create --title "fix: ${{ github.event.issue.title }}" --body "Automated fix by buzztr"
```

## Recommendation

| Scenario | Use |
|---|---|
| Quick test on your own code | Git worktree |
| Running on untrusted tasks | Docker |
| Production/CI | Docker or locked-down user |
| Maximum security | Docker with no network (use cloud LLM API) |
