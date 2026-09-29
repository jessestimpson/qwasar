# highlight.js 11.9.0 (vendored)

Syntax highlighting for fenced code blocks in the transcript (PLAN.md 5.6).
Runs in **JavaScriptCore**, a system framework, so there is nothing for a user
to install — the same bar the rest of this project holds.

## Provenance

Fetched from the official CDN distribution of the npm package
`@highlightjs/cdn-assets@11.9.0`:

    core.min.js        https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.9.0/highlight.min.js
    langs/<name>.min.js  .../@highlightjs/cdn-assets@11.9.0/languages/<name>.min.js
    LICENSE            .../@highlightjs/cdn-assets@11.9.0/LICENSE

`core.min.js` is 121,727 bytes and already registers the ~40 common languages;
`langs/` holds all 192 language modules (1.4 MB). Verified: 192 languages
register, and Swift, Elixir, Erlang, C, Objective-C, bash, Python, JSON, Rust
and Makefile all highlight — including Elixir sigils with interpolation, bash
heredocs and Rust raw strings, which are the cases a hand-rolled lexer gets
wrong.

## Layout

Kept as separate files and concatenated by `build.sh`, for the reason
`tools/bin2c` gives for the Metal kernels: the pieces stay reviewable and
diffable, and the single artefact is generated rather than committed.
`highlight.bundle.js` is therefore **not** in git — `make` builds it, offline,
from what is here.

    ./build.sh          # -> highlight.bundle.js (~1.16 MB)

## Upgrading

Refetch the three URLs above at the new version, update the version in this
file and in `build.sh`'s header, then rebuild and run the transcript fixtures.

## Licence

BSD-3-Clause. See `LICENSE`. The bundle carries the notice in its header.
