//! Keep fd 2 away from the TUI: while the alternate screen is up, anything
//! written to stderr (`eprintln!`, subprocess stderr, tracing) would scroll
//! the terminal and corrupt ratatui's diff renderer. [`StderrRedirect`]
//! points fd 2 at a log file for its lifetime and restores it on drop.

use std::io;
use std::path::Path;

/// The saved original fd 2 (-1 = not redirected). Global so the panic hook
/// can restore stderr without owning the guard.
#[cfg(unix)]
static SAVED_STDERR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Put the original stderr back. Idempotent; safe from a panic hook.
pub fn restore_stderr() {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::sync::atomic::Ordering;
        let saved = SAVED_STDERR.swap(-1, Ordering::SeqCst);
        if saved >= 0 {
            let _ = io::stderr().flush();
            // SAFETY: `saved` came from `dup` and is closed exactly once
            // (the swap above hands it to a single caller).
            unsafe {
                libc::dup2(saved, 2);
                libc::close(saved);
            }
        }
    }
}

/// RAII guard: fd 2 appends to a file until dropped. No-op off Unix.
pub struct StderrRedirect {
    _private: (),
}

#[cfg(unix)]
impl StderrRedirect {
    /// Redirect fd 2 to `path` (opened for append, created if missing).
    pub fn to_file(path: &Path) -> io::Result<Self> {
        use std::os::fd::AsRawFd;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        // SAFETY: plain fd duplication; `file` stays open across the dup2
        // and is closed on drop afterwards (fd 2 keeps its own reference).
        unsafe {
            let saved = libc::dup(2);
            if saved < 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::dup2(file.as_raw_fd(), 2) < 0 {
                let err = io::Error::last_os_error();
                libc::close(saved);
                return Err(err);
            }
            // A nested redirect would orphan the earlier saved fd; close it.
            let prev = SAVED_STDERR.swap(saved, std::sync::atomic::Ordering::SeqCst);
            if prev >= 0 {
                libc::close(prev);
            }
            Ok(Self { _private: () })
        }
    }
}

#[cfg(not(unix))]
impl StderrRedirect {
    pub fn to_file(_path: &Path) -> io::Result<Self> {
        Ok(Self { _private: () })
    }
}

impl Drop for StderrRedirect {
    fn drop(&mut self) {
        restore_stderr();
    }
}
