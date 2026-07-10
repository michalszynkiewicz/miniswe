//! File snapshots via a shadow git repo in `.miniswe/shadow-git/`.
//!
//! Creates a separate git repo that tracks the project working tree
//! without touching the real `.git`. Each round can be snapshotted
//! and reverted to. The shadow repo is never pushed, has no remotes,
//! and is cleaned up on `miniswe init`.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

/// Manages file snapshots via a shadow git repo.
pub struct SnapshotManager {
    git_dir: PathBuf,
    work_tree: PathBuf,
    current_round: usize,
}

impl SnapshotManager {
    /// Initialize the shadow git repo and take an initial snapshot (round 0).
    pub fn init(project_root: &Path) -> Result<Self> {
        let git_dir = project_root.join(".miniswe").join("shadow-git");

        // Create or reinitialize
        if git_dir.exists() {
            std::fs::remove_dir_all(&git_dir).ok();
        }

        let status = Command::new("git")
            .args(["init", "--bare"])
            .arg(&git_dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .context("failed to init shadow git")?;

        if !status.success() {
            anyhow::bail!("git init --bare failed");
        }

        let manager = Self {
            git_dir: git_dir.clone(),
            work_tree: project_root.to_path_buf(),
            current_round: 0,
        };

        // Set identity so commits work in any environment
        manager.git_config("user.email", "miniswe@local")?;
        manager.git_config("user.name", "miniswe")?;

        // The shadow git dir lives INSIDE its own work tree (project_root),
        // unlike a normal repo's `.git`, which git auto-excludes from `git
        // add` by name. A bare `--git-dir` pointing at an arbitrary nested
        // path gets no such special-casing, so `git add -A` would otherwise
        // track the shadow repo's own object files as blobs within itself —
        // self-referential tracking that `git reset --hard` (unlike the
        // gentler `checkout <commit> -- <pathspec>`) refuses to reconcile.
        // `info/exclude` is git's repo-local, untracked equivalent of
        // .gitignore — exclude the shadow-git dir from ever being added.
        std::fs::write(
            git_dir.join("info").join("exclude"),
            "/.miniswe/shadow-git/\n",
        )
        .context("failed to write shadow-git exclude file")?;

        // Initial snapshot
        manager.snapshot("session start")?;

        Ok(manager)
    }

    /// Take a snapshot of the current state.
    pub fn snapshot(&self, label: &str) -> Result<()> {
        // Stage all changes
        let status = self.git(&["add", "-A"])?;
        if !status.success() {
            anyhow::bail!("git add failed");
        }

        // Commit (allow empty for initial state)
        let msg = format!("round {} — {}", self.current_round, label);
        let status = self.git(&["commit", "--allow-empty", "-m", &msg])?;
        if !status.success() {
            // Nothing to commit is fine
        }

        Ok(())
    }

    /// Record a new round. Call before each agent round starts.
    pub fn begin_round(&mut self, round: usize) -> Result<()> {
        self.current_round = round;
        self.snapshot(&format!("before round {round}"))
    }

    /// Revert all files to the state at a specific round.
    pub fn revert_to_round(&self, target_round: usize) -> Result<String> {
        // Find the commit for that round
        let output = self.git_output(&[
            "log",
            "--oneline",
            "--all",
            "--grep",
            &format!("round {target_round}"),
        ])?;

        let commit = output
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().next())
            .context(format!("no snapshot found for round {target_round}"))?
            .to_string();

        // `reset --hard`, not `checkout <commit> -- .`: checkout with a
        // pathspec only restores content for files present in the target
        // commit — it leaves anything created in a LATER round untouched on
        // disk, so a file added after the target round survives a "revert"
        // to before it existed. `reset --hard` moves the branch, resets the
        // index, and syncs the working tree to match exactly, deleting
        // anything tracked in the current HEAD but absent from the target.
        let status = self.git(&["reset", "--hard", &commit])?;
        if !status.success() {
            anyhow::bail!("git reset --hard failed for round {target_round}");
        }

        Ok(format!(
            "Reverted to round {target_round} (commit {commit})"
        ))
    }

    /// Revert a single file to its state at a specific round.
    pub fn revert_file(&self, path: &str, target_round: usize) -> Result<String> {
        let output = self.git_output(&[
            "log",
            "--oneline",
            "--all",
            "--grep",
            &format!("round {target_round}"),
        ])?;

        let commit = output
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().next())
            .context(format!("no snapshot found for round {target_round}"))?
            .to_string();

        // A file that didn't exist yet at the target round has no content to
        // check out — `checkout <commit> -- path` fails with a pathspec
        // error for a path absent from the target tree and leaves the file
        // untouched, instead of correctly deleting it. "Reverted to a round
        // before this file existed" means the file shouldn't exist, so
        // remove it instead.
        let existed = self
            .git(&["cat-file", "-e", &format!("{commit}:{path}")])?
            .success();

        if existed {
            let status = self.git(&["checkout", &commit, "--", path])?;
            if !status.success() {
                anyhow::bail!("git checkout failed for {path} at round {target_round}");
            }
        } else {
            let status = self.git(&["rm", "-f", "--ignore-unmatch", "--", path])?;
            if !status.success() {
                anyhow::bail!("git rm failed for {path} at round {target_round}");
            }
        }

        Ok(format!("Reverted {path} to round {target_round}"))
    }

    /// Revert everything to session start.
    pub fn revert_all(&self) -> Result<String> {
        self.revert_to_round(0)
    }

    /// List available snapshots.
    pub fn list_snapshots(&self) -> Result<String> {
        self.git_output(&["log", "--oneline", "--all"])
    }

    /// Run a git command with the shadow git dir and work tree.
    fn git(&self, args: &[&str]) -> Result<std::process::ExitStatus> {
        Command::new("git")
            .arg("--git-dir")
            .arg(&self.git_dir)
            .arg("--work-tree")
            .arg(&self.work_tree)
            .args(args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .context("failed to run git")
    }

    /// Set a config value in the shadow git repo.
    fn git_config(&self, key: &str, value: &str) -> Result<()> {
        Command::new("git")
            .arg("--git-dir")
            .arg(&self.git_dir)
            .args(["config", key, value])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .context("failed to run git config")?;
        Ok(())
    }

    /// Run a git command and capture stdout.
    fn git_output(&self, args: &[&str]) -> Result<String> {
        let output = Command::new("git")
            .arg("--git-dir")
            .arg(&self.git_dir)
            .arg("--work-tree")
            .arg(&self.work_tree)
            .args(args)
            .output()
            .context("failed to run git")?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
