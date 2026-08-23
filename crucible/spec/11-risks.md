## 11. Interaction risks worth naming now

**A generic `invoke` may cost tool-calling accuracy.** §2.2 forces the fixed
surface, but a model trained to call `grep_ast(pattern: ...)` directly may call
`invoke(name: "grep_ast", args: "{...}")` less reliably — nested JSON inside an
XML parameter is a format it has seen less of. **Measure this in Milestone 4**
against a small suite of agent-written tools. If accuracy is bad, the fallback
is to accept a re-prefill when the tool set changes, amortised by only allowing
tool-surface changes at user-turn boundaries and by the system-prefix
checkpoint. Do not assume; measure.

**Sandboxed work can drift from a moving tree.** Long sessions against a tree
the user is also editing will conflict. §7.4 step 7 detects it; it does not
prevent it. Consider a periodic re-baseline offer for sessions older than an
hour.

**The model must understand that it is in a box.** The system turn has to say
so, concretely: `/work` is a copy, changes are not real until proposed and
approved, there is no network, the sandbox may restart. A model that thinks it
edited the user's file and then does not propose the patch has done nothing at
all.

**VM boot is on the critical path of the first tool call.** Two seconds is fine.
Ten is not, and it is easy to reach ten by putting OpenRC in the image. Boot
time is a tracked number.

---
