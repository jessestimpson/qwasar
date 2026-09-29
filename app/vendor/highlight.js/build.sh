#!/bin/sh
# Rebuilds highlight.bundle.js from the vendored core + language modules.
# Fetch with: see the URLs in README.md. Concatenation order is core first,
# then languages alphabetically, because a language module calls
# hljs.registerLanguage and needs hljs to exist.
set -eu
cd "$(dirname "$0")"
{ printf '/* highlight.js 11.9.0 -- https://highlightjs.org -- BSD-3-Clause (see LICENSE).\n * Vendored: core build (~40 languages) plus all 192 language modules, concatenated\n * by vendor/highlight.js/build.sh. Run at runtime in JavaScriptCore; see PLAN.md 5.6.\n * DO NOT EDIT -- regenerate with build.sh. */\n'
  cat core.min.js; echo
  for f in $(ls langs | sort); do cat "langs/$f"; echo; done
} > highlight.bundle.js
echo "highlight.bundle.js: $(wc -c < highlight.bundle.js) bytes, $(ls langs | wc -l | tr -d ' ') language modules"
