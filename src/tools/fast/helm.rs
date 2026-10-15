//! Syntax check for Helm (Go `text/template`) YAML templates.
//!
//! Plain tree-sitter-yaml rejects every chart template: `{{- if … }}`,
//! `{{ include … | nindent 4 }}` and friends are not YAML. Reporting that
//! as `[ast] broken` on every edit made `auto_revert_ast_cascade` throw away
//! correct work after three edits. So a templated YAML file is checked in
//! two layers instead:
//!
//! 1. **Template layer**: lex the `{{ … }}` actions (strings, raw strings
//!    and `/* */` comments respected) and check that every action is closed,
//!    parentheses balance inside it, and `if`/`range`/`with`/`define`/`block`
//!    pair with `end`, with `else` only where it belongs.
//! 2. **YAML layer**: render the template structurally and parse the result
//!    as YAML. Actions that print nothing (control flow, comments, `$x :=`)
//!    are removed, honouring `{{-`/`-}}` whitespace trimming. An output
//!    action alone on its line, or ending a `key:` line (`labels: {{-
//!    include … | nindent 4 }}`), injects YAML of its own and is removed too.
//!    Any other output action (`name: {{ .Values.x }}`) becomes a unique
//!    placeholder scalar of the same width. `if`/`else` branches are
//!    mutually exclusive, so this runs twice: once taking the first branch of
//!    every conditional, once the last. A key set in both branches is then
//!    not a duplicate, and an `if` rendering a mapping with an `else`
//!    rendering a list is not a clash. `define` bodies are not rendered in
//!    place and are dropped.
//!
//! Error positions are mapped back to the original file.

/// One `{{ … }}` action: byte span (delimiters included), its keyword, and
/// its whitespace-trim markers.
#[derive(Debug, Clone)]
struct Action {
    start: usize,
    end: usize,
    kind: Kind,
    trim_left: bool,
    trim_right: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `if`, `range`, `with`: opens a block that may have `else` branches.
    Open,
    /// `define`, `block`: opens a block whose body is not rendered in place
    /// (`block` is, but it is rare in chart YAML and dropping it is safe).
    Define,
    Else,
    End,
    /// Prints nothing: comment, `$x := …`, `break`, `continue`.
    Silent,
    /// Prints something: `.Values.x`, `include`, `template`, `toYaml`, …
    Output,
}

impl Kind {
    fn prints(self) -> bool {
        self == Kind::Output
    }
}

/// A block: the opening action, every `{{ else }}`, and the `{{ end }}`, as
/// indices into the action list.
struct Block {
    open: usize,
    elses: Vec<usize>,
    end: usize,
    define: bool,
}

/// Signature of the plain YAML checker this module runs on each rendering:
/// content, plus `comparable(a, b)` telling it whether keys at 0-based
/// `(row, col)` positions `a` and `b` may be reported as duplicates of each
/// other.
pub(super) type YamlCheck<'a> =
    &'a dyn Fn(&str, &dyn Fn((usize, usize), (usize, usize)) -> bool) -> Result<(), String>;

/// The module's only entry point. Returns `None` when the file is not a
/// Go/Helm template (the caller runs its usual check), otherwise the result
/// of checking it as one:
/// - `.yaml`/`.yml` containing `{{`: template layer + YAML layer;
/// - `.tpl` (`_helpers.tpl` and friends, pure template text): template layer.
pub(super) fn check(ext: &str, content: &str, yaml_check: YamlCheck) -> Option<Result<(), String>> {
    match ext {
        "tpl" => Some(lex(content).and_then(|a| pair_blocks(content, &a).map(drop))),
        "yaml" | "yml" if content.contains("{{") => Some(check_yaml_template(content, yaml_check)),
        _ => None,
    }
}

fn check_yaml_template(content: &str, yaml_check: YamlCheck) -> Result<(), String> {
    let actions = lex(content)?;
    let blocks = pair_blocks(content, &actions)?;
    // Errors in the first-branch rendering win, so a report is stable.
    for keep_last in [false, true] {
        let r = render(content, &actions, &blocks, keep_last);
        let comparable = |a, b| {
            scope(&blocks, &actions, r.origin_of(a)) == scope(&blocks, &actions, r.origin_of(b))
        };
        yaml_check(&r.text, &comparable).map_err(|e| {
            format!(
                "{} (YAML with template actions masked)",
                r.remap_error(content, &e)
            )
        })?;
    }
    Ok(())
}

/// The blocks enclosing byte `pos`, outermost first. Two keys are only
/// reported as duplicates when their scopes are identical, i.e. they are
/// always rendered together. A key set inside `{{ if .a }}` and again inside
/// a sibling `{{ if .b }}` is usually a pair of alternatives, and a key
/// inside `{{ with .Values.sidecars }}{{ toYaml . }}` may belong to the
/// injected YAML rather than the mapping it appears to sit in.
fn scope(blocks: &[Block], actions: &[Action], pos: usize) -> Vec<usize> {
    let mut enclosing: Vec<(usize, usize)> = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| actions[b.open].end <= pos && pos < actions[b.end].start)
        .map(|(n, b)| (actions[b.open].start, n))
        .collect();
    enclosing.sort_unstable();
    enclosing.into_iter().map(|(_, n)| n).collect()
}

fn line_col(src: &str, byte: usize) -> (usize, usize) {
    let before = &src[..byte];
    let line = before.matches('\n').count() + 1;
    let col = before.rfind('\n').map_or(byte, |nl| byte - nl - 1) + 1;
    (line, col)
}

fn err_at(src: &str, byte: usize, msg: impl std::fmt::Display) -> String {
    let (l, c) = line_col(src, byte);
    format!("{l}:{c}: {msg}")
}

/// Find every action. Errors on an unclosed action, string or comment, and
/// on unbalanced parentheses.
fn lex(src: &str) -> Result<Vec<Action>, String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(off) = src[i..].find("{{") {
        let start = i + off;
        let mut j = start + 2;
        let trim_left =
            b.get(j) == Some(&b'-') && b.get(j + 1).is_some_and(|c| c.is_ascii_whitespace());
        if trim_left {
            j += 1;
        }
        let body_start = j;
        let mut depth: i32 = 0;
        let mut end = None;
        while j < b.len() {
            match b[j] {
                q @ (b'"' | b'\'') => {
                    j += 1;
                    while j < b.len() && b[j] != q && b[j] != b'\n' {
                        j += if b[j] == b'\\' { 2 } else { 1 };
                    }
                    if j >= b.len() || b[j] != q {
                        return Err(err_at(src, start, "unterminated string in template action"));
                    }
                    j += 1;
                }
                b'`' => {
                    let Some(close) = src[j + 1..].find('`') else {
                        return Err(err_at(
                            src,
                            start,
                            "unterminated raw string in template action",
                        ));
                    };
                    j += close + 2;
                }
                b'/' if b.get(j + 1) == Some(&b'*') => {
                    let Some(close) = src[j + 2..].find("*/") else {
                        return Err(err_at(
                            src,
                            start,
                            "unterminated comment in template action",
                        ));
                    };
                    j += 2 + close + 2;
                }
                b'(' => {
                    depth += 1;
                    j += 1;
                }
                b')' => {
                    depth -= 1;
                    if depth < 0 {
                        return Err(err_at(src, j, "unmatched ')' in template action"));
                    }
                    j += 1;
                }
                b'}' if b.get(j + 1) == Some(&b'}') => {
                    end = Some(j + 2);
                    break;
                }
                b'{' if b.get(j + 1) == Some(&b'{') => {
                    return Err(err_at(
                        src,
                        start,
                        "template action not closed with '}}' before the next '{{'",
                    ));
                }
                _ => j += 1,
            }
        }
        let Some(end) = end else {
            return Err(err_at(src, start, "template action not closed with '}}'"));
        };
        if depth != 0 {
            return Err(err_at(src, start, "unbalanced '(' in template action"));
        }
        let mut body = &src[body_start..end - 2];
        let mut trim_right = false;
        if let Some(stripped) = body.strip_suffix('-')
            && stripped.ends_with(|c: char| c.is_ascii_whitespace())
        {
            body = stripped;
            trim_right = true;
        }
        out.push(Action {
            start,
            end,
            kind: classify(body),
            trim_left,
            trim_right,
        });
        i = end;
    }
    Ok(out)
}

fn classify(body: &str) -> Kind {
    let body = body.trim_start();
    if body.starts_with("/*") {
        return Kind::Silent;
    }
    let mut words = body.split_whitespace();
    let first = words.next().unwrap_or("");
    if first.starts_with('$') && matches!(words.next(), Some(":=" | "=")) {
        return Kind::Silent;
    }
    let keyword = first
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .next()
        .unwrap_or("");
    match keyword {
        "if" | "range" | "with" => Kind::Open,
        "define" | "block" => Kind::Define,
        "else" => Kind::Else,
        "end" => Kind::End,
        "break" | "continue" => Kind::Silent,
        _ => Kind::Output,
    }
}

fn pair_blocks(src: &str, actions: &[Action]) -> Result<Vec<Block>, String> {
    let mut stack: Vec<Block> = Vec::new();
    let mut done = Vec::new();
    for (idx, a) in actions.iter().enumerate() {
        match a.kind {
            Kind::Open | Kind::Define => stack.push(Block {
                open: idx,
                elses: Vec::new(),
                end: 0,
                define: a.kind == Kind::Define,
            }),
            Kind::Else => match stack.last_mut() {
                Some(blk) if !blk.define => blk.elses.push(idx),
                _ => {
                    return Err(err_at(
                        src,
                        a.start,
                        "{{ else }} outside an if/range/with block",
                    ));
                }
            },
            Kind::End => {
                let Some(mut blk) = stack.pop() else {
                    return Err(err_at(
                        src,
                        a.start,
                        "{{ end }} with no open if/range/with/define",
                    ));
                };
                blk.end = idx;
                done.push(blk);
            }
            Kind::Silent | Kind::Output => {}
        }
    }
    if let Some(blk) = stack.last() {
        let a = &actions[blk.open];
        let what = src[a.start..a.end].trim();
        return Err(err_at(
            src,
            a.start,
            format!("{what} is never closed with {{{{ end }}}}"),
        ));
    }
    Ok(done)
}

/// One structural rendering of the template, with every output byte
/// traced back to its source byte.
struct Rendering {
    text: String,
    origin: Vec<usize>,
    line_starts: Vec<usize>,
}

impl Rendering {
    /// Source byte for a 0-based `(row, col)` in the rendering.
    fn origin_of(&self, (row, col): (usize, usize)) -> usize {
        let idx = self
            .line_starts
            .get(row)
            .map_or(self.text.len(), |s| s + col);
        self.origin
            .get(idx)
            .or(self.origin.last())
            .copied()
            .unwrap_or(0)
    }

    /// Rewrite a leading `row:col:` (1-based) to the source position.
    fn remap_error(&self, src: &str, err: &str) -> String {
        let mut parts = err.splitn(3, ':');
        let (Some(r), Some(c), Some(rest)) = (parts.next(), parts.next(), parts.next()) else {
            return err.to_string();
        };
        let (Ok(r), Ok(c)) = (r.parse::<usize>(), c.parse::<usize>()) else {
            return err.to_string();
        };
        let (l, c) = line_col(src, self.origin_of((r - 1, c - 1)).min(src.len()));
        let rest = remap_first_defined(self, src, rest);
        format!("{l}:{c}:{rest}")
    }
}

/// Duplicate-key errors carry a second position, `(first defined at r:c)`.
fn remap_first_defined(r: &Rendering, src: &str, rest: &str) -> String {
    const MARK: &str = "(first defined at ";
    let Some(at) = rest.find(MARK) else {
        return rest.to_string();
    };
    let tail = &rest[at + MARK.len()..];
    let Some(close) = tail.find(')') else {
        return rest.to_string();
    };
    let Some((row, col)) = tail[..close].split_once(':') else {
        return rest.to_string();
    };
    let (Ok(row), Ok(col)) = (row.parse::<usize>(), col.parse::<usize>()) else {
        return rest.to_string();
    };
    let (l, c) = line_col(src, r.origin_of((row - 1, col - 1)).min(src.len()));
    format!("{}{MARK}{l}:{c}{}", &rest[..at], &tail[close..])
}

/// Render `src` taking the first (or last) branch of every conditional.
fn render(src: &str, actions: &[Action], blocks: &[Block], keep_last: bool) -> Rendering {
    let n = src.len();
    // Whether each source byte reaches the rendering.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fate {
        Keep,
        Drop,
    }
    let mut fate = vec![Fate::Keep; n];
    let drop = |from: usize, to: usize, fate: &mut [Fate]| fate[from..to].fill(Fate::Drop);

    // Branches this rendering does not take, and define bodies.
    for blk in blocks {
        let open_end = actions[blk.open].end;
        let end_start = actions[blk.end].start;
        if blk.define {
            drop(open_end, end_start, &mut fate);
            continue;
        }
        let mut bounds = Vec::new();
        let mut from = open_end;
        for &e in &blk.elses {
            bounds.push((from, actions[e].start));
            from = actions[e].end;
        }
        bounds.push((from, end_start));
        let keep = if keep_last { bounds.len() - 1 } else { 0 };
        for (i, &(s, e)) in bounds.iter().enumerate() {
            if i != keep {
                drop(s, e, &mut fate);
            }
        }
    }

    // Which actions this rendering reaches, and placeholder text for the
    // output actions that stay inline.
    let live: Vec<bool> = actions
        .iter()
        .map(|a| fate[a.start] == Fate::Keep)
        .collect();
    let mut placeholder: Vec<Option<Vec<u8>>> = vec![None; actions.len()];
    for (i, a) in actions.iter().enumerate() {
        if !live[i] {
            continue;
        }
        let inline = a.kind.prints()
            && !standalone(src, a, actions)
            && !block_value(src, a, actions)
            && !src[a.start..a.end].contains('\n');
        if inline {
            // Unique (`t1xxx`, `t2xxx`, …) so two masked keys never read as
            // duplicates of each other.
            let tag = format!("t{i:x}");
            let width = a.end - a.start;
            let mut text = tag.into_bytes();
            text.resize(width.max(text.len()), b'x');
            placeholder[i] = Some(text);
        }
        drop(a.start, a.end, &mut fate);
    }

    // Emit, applying `{{-` / `-}}` trimming around removed actions (spaces
    // and tabs only, so rows never move).
    let bytes = src.as_bytes();
    let mut text: Vec<u8> = Vec::with_capacity(n);
    let mut origin: Vec<usize> = Vec::with_capacity(n);
    let mut next_action = 0;
    let mut trim_following = false;
    let mut join_next = false;
    let mut pos = 0;
    while pos < n {
        if next_action < actions.len() && actions[next_action].start == pos {
            let i = next_action;
            next_action += 1;
            let a = &actions[i];
            let taken = live[i];
            if let Some(ph) = &placeholder[i] {
                for (k, &c) in ph.iter().enumerate() {
                    text.push(c);
                    origin.push((a.start + k).min(a.end - 1));
                }
                trim_following = false;
                join_next = false;
            } else {
                if taken && a.trim_left {
                    trim_left(&mut text, &mut origin, a.start);
                }
                for (k, _) in src[a.start..a.end].match_indices('\n') {
                    text.push(b'\n');
                    origin.push(a.start + k);
                }
                trim_following = taken && a.trim_right;
                // `apiVersion: {{ if … -}}` with the value on the next line:
                // Go's `-}}` eats that newline too, joining the value onto
                // the key. Only done when the line already holds content, so
                // a standalone `{{- end -}}` keeps rows where they are.
                join_next = trim_following && !current_line(&text).trim_ascii().is_empty();
            }
            pos = a.end;
            continue;
        }
        let c = bytes[pos];
        if fate[pos] == Fate::Drop {
            // A branch not taken renders nothing, newlines included; error
            // positions come from `origin`, not from row counts. `-}}` only
            // trims the text right after it, which is this unrendered text.
            trim_following = false;
            join_next = false;
        } else if trim_following
            && (c == b' ' || c == b'\t' || (join_next && c.is_ascii_whitespace()))
        {
            // eaten by a preceding `-}}`
        } else {
            trim_following = false;
            join_next = false;
            text.push(c);
            origin.push(pos);
        }
        pos += 1;
    }

    // Lines left holding only whitespace become empty: inside a block scalar
    // (`script: |` whose first line was `{{- if … }}`) a long all-space line
    // before the content is a YAML error.
    let mut out_text = Vec::with_capacity(text.len());
    let mut out_origin = Vec::with_capacity(text.len());
    let mut line_starts = vec![0];
    for line in split_lines(&text) {
        let (s, e) = line;
        let blank = text[s..e].iter().all(|c| c.is_ascii_whitespace());
        if !blank {
            out_text.extend_from_slice(&text[s..e]);
            out_origin.extend_from_slice(&origin[s..e]);
        }
        if e < text.len() {
            out_text.push(b'\n');
            out_origin.push(origin[e]);
            line_starts.push(out_text.len());
        }
    }
    Rendering {
        // Only whole actions (ASCII-delimited) and ASCII whitespace were
        // removed or replaced, so this stays valid UTF-8.
        text: String::from_utf8(out_text).unwrap_or_else(|_| src.to_string()),
        origin: out_origin,
        line_starts,
    }
}

/// Apply a `{{-` before source byte `at`: drop the spaces and tabs that
/// were emitted straight from the source bytes before it. Go would also eat
/// a newline and the indentation before it; rows must not move, so when the
/// run reaches the start of the line nothing is trimmed (that keeps a
/// block-scalar line's indentation intact).
fn trim_left(text: &mut Vec<u8>, origin: &mut Vec<usize>, at: usize) {
    let mut n = 0;
    let mut expect = at;
    for (&c, &o) in text.iter().rev().zip(origin.iter().rev()) {
        if (c != b' ' && c != b'\t') || o + 1 != expect {
            break;
        }
        n += 1;
        expect = o;
    }
    let keep = text.len() - n;
    if keep == 0 || text[keep - 1] == b'\n' {
        return;
    }
    text.truncate(keep);
    origin.truncate(keep);
}

/// The text after the last newline of `text`.
fn current_line(text: &[u8]) -> &[u8] {
    let start = text.iter().rposition(|&c| c == b'\n').map_or(0, |p| p + 1);
    &text[start..]
}

/// `(start, end)` of each line in `text`, `end` excluding the newline.
fn split_lines(text: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = 0;
    for (i, &c) in text.iter().enumerate() {
        if c == b'\n' {
            out.push((s, i));
            s = i + 1;
        }
    }
    out.push((s, text.len()));
    out
}

/// An action is standalone when its line holds nothing but actions and
/// whitespace: it renders to nothing or to injected YAML, never to an inline
/// value.
fn standalone(src: &str, a: &Action, actions: &[Action]) -> bool {
    let line_start = src[..a.start].rfind('\n').map_or(0, |p| p + 1);
    let line_end = src[a.end..].find('\n').map_or(src.len(), |p| a.end + p);
    let mut pos = line_start;
    for other in actions
        .iter()
        .filter(|o| o.start >= line_start && o.end <= line_end)
    {
        if !src[pos..other.start].trim().is_empty() {
            return false;
        }
        pos = other.end;
    }
    src[pos..line_end].trim().is_empty()
}

/// An action that ends its line right after `key:` (`labels: {{- include
/// … | nindent 4 }}`) usually injects a nested block, and the template
/// often continues that block on the following, deeper-indented lines. A
/// placeholder scalar there would make those lines a YAML error, so the
/// value is left empty instead (`labels:` is valid YAML either way).
fn block_value(src: &str, a: &Action, actions: &[Action]) -> bool {
    let line_start = src[..a.start].rfind('\n').map_or(0, |p| p + 1);
    let line_end = src[a.end..].find('\n').map_or(src.len(), |p| a.end + p);
    // Nothing but actions and whitespace after this one…
    let mut pos = a.end;
    for other in actions
        .iter()
        .filter(|o| o.start >= a.end && o.end <= line_end)
    {
        if !src[pos..other.start].trim().is_empty() {
            return false;
        }
        pos = other.end;
    }
    if !src[pos..line_end].trim().is_empty() {
        return false;
    }
    // …and before it, skipping earlier trailing actions, `key:` then space.
    let mut prefix_end = a.start;
    for other in actions
        .iter()
        .rev()
        .filter(|o| o.start >= line_start && o.end <= a.start)
    {
        if !src[other.end..prefix_end].trim().is_empty() {
            break;
        }
        prefix_end = other.start;
    }
    let prefix = &src[line_start..prefix_end];
    let trimmed = prefix.trim_end();
    // `key: |-` / `key: >` (a block-scalar header) works the same way: the
    // action supplies the scalar's indented lines.
    let trimmed = trimmed
        .trim_end_matches(|c: char| c.is_ascii_digit() || c == '-' || c == '+')
        .strip_suffix(['|', '>'])
        .map_or(trimmed, str::trim_end);
    trimmed.ends_with(':') && trimmed.len() < prefix.len()
}

#[cfg(test)]
mod tests {
    use crate::tools::fast::parse_check;

    fn ok(src: &str) {
        let r = parse_check("chart/templates/t.yaml", src);
        assert!(r.is_ok(), "expected clean, got {r:?} for:\n{src}");
    }

    fn broken(src: &str) -> String {
        parse_check("chart/templates/t.yaml", src).expect_err(src)
    }

    #[test]
    fn common_helm_patterns_parse() {
        ok(
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {{ include \"app.fullname\" . }}\n  labels:\n    {{- include \"app.labels\" . | nindent 4 }}\nspec:\n  {{- if not .Values.autoscaling.enabled }}\n  replicas: {{ .Values.replicaCount }}\n  {{- end }}\n  template:\n    spec:\n      containers:\n        - name: {{ .Chart.Name }}\n          image: \"{{ .Values.image.repository }}:{{ .Values.image.tag }}\"\n",
        );
        ok("{{- if .Values.ingress.enabled -}}\napiVersion: v1\nkind: Ingress\n{{- end }}\n");
        ok(
            "env:\n{{- range $k, $v := .Values.env }}\n  - name: {{ $k }}\n    value: {{ $v | quote }}\n{{- end }}\n",
        );
        ok(
            "{{- with .Values.nodeSelector }}\nnodeSelector:\n  {{- toYaml . | nindent 8 }}\n{{- end }}\n",
        );
        ok(
            "{{/* comment with }} inside */}}\n{{- $name := include \"x\" . }}\nname: {{ $name }}\n",
        );
    }

    #[test]
    fn key_injected_on_the_key_line_then_continued_below() {
        ok(
            "metadata:\n  labels: {{- include \"common.labels\" . | nindent 4 }}\n    app.kubernetes.io/component: master\nspec: {}\n",
        );
    }

    #[test]
    fn same_key_in_if_and_else_is_not_a_duplicate() {
        ok("spec:\n  {{- if .Values.a }}\n  type: A\n  {{- else }}\n  type: B\n  {{- end }}\n");
    }

    #[test]
    fn same_key_in_sibling_ifs_is_not_a_duplicate() {
        ok(
            "annotations:\n  {{- if .a }}\n  checksum/config: {{ .a }}\n  {{- end }}\n  {{- if .b }}\n  checksum/config: {{ .b }}\n  {{- end }}\n",
        );
    }

    #[test]
    fn branches_rendering_different_shapes_parse() {
        // Mapping in one branch, list in the other.
        ok("items:\n{{- if .a }}\n  key: value\n{{- else }}\n  - one\n{{- end }}\n");
    }

    #[test]
    fn inline_if_else_choosing_a_key() {
        ok(
            "{{ if .Values.secret -}} stringData: {{- else -}} data: {{- end }}\n  config.yaml: |\n{{ toYaml .Values.config | indent 4 }}\n",
        );
        ok("kind: {{ if .Values.secret -}} Secret {{- else -}} ConfigMap {{- end }}\n");
    }

    #[test]
    fn control_flow_inside_a_block_scalar() {
        ok(
            "data:\n  run.sh: |\n    {{- if .Values.debug }}\n    set -x\n    {{- end }}\n    NAMES=\"{{- range $i, $e := .Values.n -}}\n    {{- if $i }} {{ end }}{{ $e }}\n    {{- end }}\"\n    exec app\n",
        );
    }

    #[test]
    fn plain_yaml_errors_still_caught_in_templates() {
        let err = broken("metadata:\n  name: {{ .Values.name }}\n  labels\n    app: x\n");
        assert!(err.contains("(YAML with template actions masked)"), "{err}");
    }

    #[test]
    fn block_scalar_header_followed_by_an_injected_body() {
        // grafana-sampling: `config.alloy: |- {{- include … | nindent 4 }}`
        ok("data:\n  config.alloy: |- {{- include \"cfg\" . | nindent 4 }}\nkind: ConfigMap\n");
        ok("data:\n  run.sh: > {{ .Values.script | nindent 4 }}\n");
    }

    #[test]
    fn trimmed_inline_if_joins_the_value_from_the_next_line() {
        // promtail: Go's `-}}` eats the newline, so the value lands on the key.
        ok(
            "apiVersion: {{ if .Values.v2 -}}\nautoscaling/v2\n{{- else -}}\nautoscaling/v2beta2\n{{- end }}\nkind: HorizontalPodAutoscaler\n",
        );
    }

    #[test]
    fn trimmed_output_action_that_swallows_the_next_key_is_broken() {
        // A real chart bug (bitnami seaweedfs): `-}}` joins `namespace:` onto
        // the `name:` line, which Helm renders as invalid YAML.
        broken("metadata:\n  name: {{ printf \"%s-crt\" .Release.Name -}}\n  namespace: ns\n");
    }

    #[test]
    fn error_only_in_the_else_branch_is_caught() {
        // Only the second (last-branch) rendering sees this one; the same
        // file with the else branch fixed is clean.
        let bad_else = "spec:\n  {{- if .Values.a }}\n  type: A\n  {{- else }}\n  type: B\n  labels\n    app: x\n  {{- end }}\n";
        let err = broken(bad_else);
        assert!(err.contains("(YAML with template actions masked)"), "{err}");
        ok(&bad_else.replace("  labels\n", "  labels:\n"));
    }

    #[test]
    fn duplicate_key_in_the_same_scope_caught_with_original_positions() {
        let err = broken(
            "{{- if .Values.enabled }}\nmetadata:\n  name: {{ .Values.name }}\n  name: other\n{{- end }}\n",
        );
        assert!(err.starts_with("4:3:"), "{err}");
        assert!(err.contains("duplicate key 'name'"), "{err}");
        assert!(err.contains("first defined at 3:3"), "{err}");
    }

    #[test]
    fn template_structure_errors_point_at_the_action() {
        let err = broken("a: 1\n{{- if .x }}\nb: 2\n");
        assert!(
            err.starts_with("2:1:") && err.contains("never closed"),
            "{err}"
        );
        assert!(broken("a: {{ .x \nb: 1\n").contains("not closed"));
        assert!(broken("a: {{ (include \"x\" . }}\n").contains("unbalanced"));
        assert!(broken("a: {{ \"unterminated }}\n").contains("unterminated string"));
        assert!(broken("a: 1\n{{ end }}\n").contains("no open"));
        assert!(broken("a: 1\n{{ else }}\n").contains("outside"));
        assert!(broken("{{ define \"x\" }}{{ else }}{{ end }}\na: 1\n").contains("outside"));
    }

    #[test]
    fn tpl_files_get_the_template_layer() {
        let helpers = "{{/* Expand the name */}}\n{{- define \"app.name\" -}}\n{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix \"-\" }}\n{{- end }}\n";
        assert!(parse_check("chart/templates/_helpers.tpl", helpers).is_ok());
        let err = parse_check(
            "chart/templates/_helpers.tpl",
            "{{- define \"app.name\" -}}\nx\n",
        )
        .unwrap_err();
        assert!(err.contains("never closed"), "{err}");
    }

    #[test]
    fn yaml_without_template_actions_is_left_to_the_plain_check() {
        assert!(super::check("yaml", "a: 1\n", &|_, _| Err("plain".into())).is_none());
        assert!(super::check("json", "{{ x }}", &|_, _| Ok(())).is_none());
    }
}
