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
            git_dir,
            work_tree: project_root.to_path_buf(),
            current_round: 0,
        };

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
            "log", "--oneline", "--all", "--grep", &format!("round {target_round}")
        ])?;

        let commit = output.lines().next()
            .and_then(|l| l.split_whitespace().next())
            .context(format!("no snapshot found for round {target_round}"))?
            .to_string();

        // Checkout that commit's tree into the work tree
        let status = self.git(&["checkout", &commit, "--", "."])?;
        if !status.success() {
            anyhow::bail!("git checkout failed for round {target_round}");
        }

        Ok(format!("Reverted to round {target_round} (commit {commit})"))
    }

    /// Revert a single file to its state at a specific round.
    pub fn revert_file(&self, path: &str, target_round: usize) -> Result<String> {
        let output = self.git_output(&[
            "log", "--oneline", "--all", "--grep", &format!("round {target_round}")
        ])?;

        let commit = output.lines().next()
            .and_then(|l| l.split_whitespace().next())
            .context(format!("no snapshot found for round {target_round}"))?
            .to_string();

        let status = self.git(&["checkout", &commit, "--", path])?;
        if !status.success() {
            anyhow::bail!("git checkout failed for {path} at round {target_round}");
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
            .arg("--git-dir").arg(&self.git_dir)
            .arg("--work-tree").arg(&self.work_tree)
            .args(args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .context("failed to run git")
    }

    /// Run a git command and capture stdout.
    fn git_output(&self, args: &[&str]) -> Result<String> {
        let output = Command::new("git")
            .arg("--git-dir").arg(&self.git_dir)
            .arg("--work-tree").arg(&self.work_tree)
            .args(args)
            .output()
            .context("failed to run git")?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
