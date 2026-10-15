//! Real-chart check for the Helm template checker (`src/tools/fast/helm.rs`),
//! gated on `HELM_CORPUS_DIR` pointing at a directory of cloned chart repos.
//! `scripts/run-helm-corpus-e2e.sh` downloads them to /tmp and runs this; CI
//! leaves the variable unset, so it skips there.
//!
//! The unit tests in helm.rs pin individual patterns; this proves the checker
//! accepts what real charts actually contain. Every chart template is valid
//! by construction (these charts are released and installed daily), so any
//! `Err` here is a false `[ast] broken` that would feed the auto-revert
//! cascade.

use std::path::{Path, PathBuf};

use miniswe::tools::fast::parse_check;

/// Templates that really are broken upstream, so a failure there is the
/// checker being right. Each renders invalid YAML (or a duplicate key) in
/// at least one values branch. Paths are relative to `HELM_CORPUS_DIR`; an
/// entry that starts passing (fixed upstream) is reported, not failed.
const KNOWN_UPSTREAM_BUGS: &[(&str, &str)] = &[
    (
        "bitnami/bitnami/kube-arangodb/templates/hpa.yaml",
        "`namespace:` set twice in metadata",
    ),
    (
        "bitnami/bitnami/envoy-gateway/templates/certgen/job.yaml",
        "`volumes:` indented into the containers list when extraVolumes is set",
    ),
    (
        "bitnami/bitnami/kuberay/templates/cluster/raycluster.yaml",
        "`livenessProbe:` & co. indented into the containers list when custom probes are set",
    ),
    (
        "bitnami/bitnami/concourse/templates/web/deployment.yaml",
        "conjur_ca_cert item indented under the previous item's `path:`",
    ),
    (
        "bitnami/bitnami/seaweedfs/templates/tls-secret.yaml",
        "`name: {{ … -}}` trims the newline, joining `namespace:` onto the `name:` line",
    ),
    (
        "bitnami/bitnami/seaweedfs/templates/webadv/tls-secret.yaml",
        "`name: {{ … -}}` trims the newline, joining `namespace:` onto the `name:` line",
    ),
];

#[test]
fn real_chart_templates_pass_the_fast_mode_check() {
    let Ok(dir) = std::env::var("HELM_CORPUS_DIR") else {
        eprintln!("HELM_CORPUS_DIR unset — skipping");
        return;
    };
    let root = PathBuf::from(&dir);
    let mut files = Vec::new();
    collect_templates(&root, false, &mut files);
    files.sort();
    assert!(!files.is_empty(), "no chart templates found under {dir}");

    let mut failures = Vec::new();
    let mut known = Vec::new();
    let mut fixed_upstream = Vec::new();
    let (mut yaml, mut templated, mut tpl) = (0, 0, 0);
    for path in &files {
        let Ok(content) = std::fs::read_to_string(path) else {
            continue; // not UTF-8; the fast-mode tools never see it either
        };
        let rel = path.strip_prefix(&root).unwrap_or(path).to_string_lossy();
        if rel.ends_with(".tpl") {
            tpl += 1;
        } else {
            yaml += 1;
            if content.contains("{{") {
                templated += 1;
            }
        }
        let bug = KNOWN_UPSTREAM_BUGS.iter().find(|(p, _)| *p == rel);
        match (parse_check(&rel, &content), bug) {
            (Err(e), Some((_, why))) => {
                known.push(format!("{rel}: {e}\n         upstream bug: {why}"))
            }
            (Err(e), None) => failures.push(format!("{rel}: {e}")),
            (Ok(()), Some(_)) => fixed_upstream.push(rel.to_string()),
            (Ok(()), None) => {}
        }
    }

    eprintln!(
        "{} files checked: {yaml} .yaml/.yml ({templated} with template actions), {tpl} .tpl; \
         {} failed, {} known upstream bugs caught",
        files.len(),
        failures.len(),
        known.len()
    );
    for k in &known {
        eprintln!("  KNOWN {k}");
    }
    for f in &fixed_upstream {
        eprintln!("  NOW PASSES (drop from KNOWN_UPSTREAM_BUGS?) {f}");
    }
    for f in &failures {
        eprintln!("  FAIL {f}");
    }
    assert!(
        failures.is_empty(),
        "{} of {} real chart templates reported broken (listed above)",
        failures.len(),
        files.len()
    );
}

/// `.yaml`/`.yml`/`.tpl` files below any `templates/` directory, skipping
/// `.git`.
fn collect_templates(dir: &Path, in_templates: bool, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            if name != ".git" {
                collect_templates(&path, in_templates || name == "templates", out);
            }
        } else if kind.is_file()
            && in_templates
            && matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("yaml" | "yml" | "tpl")
            )
        {
            out.push(path);
        }
    }
}
