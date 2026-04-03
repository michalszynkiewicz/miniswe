# TODO

## Docs subsystem

### No `docs read` CLI command for humans

The CLI has `add`, `list`, `refresh` but no way to read cached docs without
manually `cat`-ing files from `~/.miniswe/docs/`.

**Fix:** Add a `docs read <name>` subcommand that prints cached content to
stdout (with optional `--topic` flag to filter sections, reusing
`extract_relevant_sections`).

### Filename matching is fragile

`docs add` derives the filename from the URL's last path segment
(e.g. `https://docs.rs/tokio/latest/tokio/` -> `tokio`). The LLM's
`docs_lookup` then does a case-insensitive substring match on filenames.
This breaks when the slug doesn't match the library name (e.g.
`hooks.html` for React docs).

**Fix:** Store a sidecar metadata file (`docs/_index.json`) mapping each
cached file to its source URL, library name, and fetch timestamp. Use the
library field for `docs_lookup` matching instead of the filename. This also
enables `docs refresh`.

### Raw HTML is stored, wasting LLM context tokens

`docs add` saves the HTTP response body as-is. For most web pages this is
full HTML with tags, scripts, and nav chrome — all noise for the LLM.

**Fix:** Run fetched content through an HTML-to-markdown converter (e.g.
`htmd` or `html2text` crate) before saving. Fall back to raw storage if
conversion fails or if the content is already plain text / markdown.

### `docs refresh` is unimplemented

The original URLs aren't stored, so there's nothing to re-fetch.

**Fix:** Depends on the metadata index above. Once `_index.json` tracks
source URLs, `refresh` iterates the index and re-fetches + reconverts each
entry.
