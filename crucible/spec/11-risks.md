## 11. Interaction risks worth naming now

**A generic `invoke` may cost tool-calling accuracy.** §2.2 forces the fixed
surface, but a model trained to call `grep_ast(pattern: ...)` directly may call
`invoke(name: "grep_ast", args: "{...}")` less reliably — nested JSON inside an
XML parameter is a format it has seen less of. **Measure this in Milestone 4**
against a small suite of agent-written tools. If accuracy is bad, the fallback
is to accept a re-prefill when the tool set changes, amortised by only allowing
tool-surface changes at user-turn boundaries and by the system-prefix
checkpoint. Do not assume; measure.

**The tree is shared, and shared means concurrent.** User and agent editing
the same file at the same moment is last-write-wins at the file level — and
the file is now in the user's own working copy (§7.4). The same hazard as
two people over one checkout, stated in the briefing, not machined around.
The sharper edge is untracked files: git cannot restore what it never
tracked, so an agent overwrite of an untracked file is a real loss. Watch
for either mattering in practice before building anything.

**The model must understand where it is.** The system turn has to say so,
concretely: `/work` IS the user's tree and every edit is immediately theirs,
version control belongs to the user and git is not the model's to operate,
there is no network, the sandbox may restart. A model that thinks its edits
are invisible will narrate instead of working; one that reaches for git
anyway is harmless — its `/work/.git` is a private shadow — but it is
wasting the turn.

**VM boot is on the critical path of the first tool call.** Two seconds is fine.
Ten is not, and it is easy to reach ten by putting OpenRC in the image. Boot
time is a tracked number.

---
