# Safe Headless Execution

How to run miniswe in headless mode (`-y`) safely, preventing unintended filesystem access.

## Built-in Safety

miniswe already enforces:
- **Path jailing** — all file operations are restricted to the project root. Absolute paths and `../` traversal are blocked.
- **Shell allowlist** — common dev commands (cargo, git, npm, etc.) auto-approved, everything else prompted in interactive mode or auto-approved with `-y`.
- **Web query visibility** — each search/fetch shows what's being sent (interactive mode only).

With `-y`, all permissions are auto-approved. The path jail still holds, but shell commands run without confirmation. Use one of the isolation methods below to limit blast radius.

## Option 1: Git Worktree (simplest)

Run in a disposable copy of your repo. If anything goes wrong, delete it.

```bash
# Create a throwaway workspace
git worktree add /tmp/miniswe-job -b miniswe-test
cd /tmp/miniswe-job

# Initialize and run
miniswe init
miniswe -y "fix the failing tests"

# Review what changed
git diff

# If happy, merge back
git checkout main
git merge miniswe-test

# Clean up
git worktree remove /tmp/miniswe-job
git branch -d miniswe-test
```

**What this gives you:** easy rollback, no risk to your working branch. The path jail restricts file access to `/tmp/miniswe-job`.

**What this does NOT give you:** shell command isolation. miniswe can still run arbitrary commands as your user (e.g., `curl`, `rm` outside the project via shell).

## Option 2: Locked-Down User (medium isolation)

Run as a dedicated user with limited filesystem permissions.

```bash
# One-time setup
sudo useradd -m -s /bin/bash miniswe-runner
# Give access to dev tools
sudo -u miniswe-runner bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

# Copy project to runner's home
sudo -u miniswe-runner cp -r /path/to/project /home/miniswe-runner/project

# Run
sudo -u miniswe-runner bash -c '
  cd /home/miniswe-runner/project
  export PATH="$HOME/.cargo/bin:$PATH"
  miniswe init
  miniswe -y "fix the failing tests"
'

# Review and copy results back
sudo -u miniswe-runner git -C /home/miniswe-runner/project diff
```

**What this gives you:** filesystem isolation via Unix permissions. The runner can't read/write files it doesn't own. Shell commands run as the restricted user.

**LLM access:** works out of the box — any user can make HTTP requests to `localhost:8464`.

## Option 3: Docker Container (strongest isolation)

Full sandboxing — filesystem, network, and process isolation.

```dockerfile
# Dockerfile.miniswe
FROM rust:slim

RUN apt-get update && apt-get install -y \
    ripgrep git nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY target/release/miniswe /usr/local/bin/miniswe

WORKDIR /workspace
```

```bash
# Build the container
docker build -t miniswe-runner -f Dockerfile.miniswe .

# Run with project mounted
docker run --rm \
  --network=host \
  -v $(pwd):/workspace \
  -w /workspace \
  miniswe-runner \
  sh -c 'miniswe init && miniswe -y "fix the failing tests"'
```

**What this gives you:** full isolation. Even if the LLM escapes the path jail or runs malicious shell commands, damage is contained to the disposable container.

**LLM access:** `--network=host` gives the container access to `localhost:8464` where llama-server runs. For tighter control, use `--add-host=host.docker.internal:host-gateway` and set `endpoint = "http://host.docker.internal:8464"` in config.

**MCP servers:** if you need MCP, mount the MCP server binaries into the container or install them in the Dockerfile.

## Option 4: CI/CD Pipeline

Run miniswe as a CI step (GitHub Actions, GitLab CI, etc.):

```yaml
# .github/workflows/miniswe.yml
name: miniswe fix
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
          # Set endpoint in .miniswe/config.toml

      - name: Run miniswe
        run: |
          miniswe init
          miniswe -y "fix: ${{ github.event.issue.title }}"

      - name: Create PR
        run: |
          git checkout -b miniswe/fix-${{ github.event.issue.number }}
          git add -A
          git commit -m "fix: ${{ github.event.issue.title }}"
          gh pr create --title "fix: ${{ github.event.issue.title }}" --body "Automated fix by miniswe"
```

## Recommendation

| Scenario | Use |
|---|---|
| Quick test on your own code | Git worktree |
| Running on untrusted tasks | Docker |
| Production/CI | Docker or locked-down user |
| Maximum security | Docker with no network (use cloud LLM API) |
