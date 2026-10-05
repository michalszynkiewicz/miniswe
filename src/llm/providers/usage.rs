//! Token usage accounting, accumulated across calls on an [`LlmClient`](super::super::client::LlmClient).

use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe running totals for a single client's usage. Cheap to clone
/// via `Arc`; every field is an independent atomic counter so concurrent
/// calls never lose an update.
#[derive(Debug, Default)]
pub struct UsageTotals {
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    cached_tokens: AtomicU64,
    cache_write_tokens: AtomicU64,
    calls: AtomicU64,
}

impl UsageTotals {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one call's usage into the running totals.
    pub fn record(
        &self,
        prompt_tokens: u64,
        completion_tokens: u64,
        cached_tokens: u64,
        cache_write_tokens: u64,
    ) {
        self.prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(completion_tokens, Ordering::Relaxed);
        self.cached_tokens
            .fetch_add(cached_tokens, Ordering::Relaxed);
        self.cache_write_tokens
            .fetch_add(cache_write_tokens, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot the current totals.
    pub fn snapshot(&self) -> UsageSnapshot {
        UsageSnapshot {
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            cached_tokens: self.cached_tokens.load(Ordering::Relaxed),
            cache_write_tokens: self.cache_write_tokens.load(Ordering::Relaxed),
            calls: self.calls.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time copy of [`UsageTotals`], cheap to pass around and log.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UsageSnapshot {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub cache_write_tokens: u64,
    pub calls: u64,
}

impl UsageSnapshot {
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

impl std::ops::Add for UsageSnapshot {
    type Output = UsageSnapshot;

    fn add(self, rhs: UsageSnapshot) -> UsageSnapshot {
        UsageSnapshot {
            prompt_tokens: self.prompt_tokens + rhs.prompt_tokens,
            completion_tokens: self.completion_tokens + rhs.completion_tokens,
            cached_tokens: self.cached_tokens + rhs.cached_tokens,
            cache_write_tokens: self.cache_write_tokens + rhs.cache_write_tokens,
            calls: self.calls + rhs.calls,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_snapshot_accumulate() {
        let totals = UsageTotals::new();
        totals.record(100, 20, 10, 6);
        totals.record(50, 5, 0, 4);
        let snap = totals.snapshot();
        assert_eq!(snap.prompt_tokens, 150);
        assert_eq!(snap.completion_tokens, 25);
        assert_eq!(snap.cached_tokens, 10);
        assert_eq!(snap.cache_write_tokens, 10);
        assert_eq!(snap.calls, 2);
        assert_eq!(snap.total_tokens(), 175);
    }

    #[test]
    fn snapshots_add() {
        let a = UsageSnapshot {
            prompt_tokens: 1,
            completion_tokens: 2,
            cached_tokens: 3,
            cache_write_tokens: 1,
            calls: 1,
        };
        let b = UsageSnapshot {
            prompt_tokens: 10,
            completion_tokens: 20,
            cached_tokens: 30,
            cache_write_tokens: 10,
            calls: 1,
        };
        let sum = a + b;
        assert_eq!(sum.prompt_tokens, 11);
        assert_eq!(sum.completion_tokens, 22);
        assert_eq!(sum.cached_tokens, 33);
        assert_eq!(sum.cache_write_tokens, 11);
        assert_eq!(sum.calls, 2);
    }
}
