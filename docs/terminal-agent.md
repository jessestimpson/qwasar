# qwasar-agent, the terminal coding agent

```
qwasar-agent -C ~/src/project "fix the bug in stats.c and rebuild"
```

```
  read path=stats.c
  edit path=stats.c old=        if (v[i] < best) best = v[i];  new=        if (v[i] > b...
  bash command=cc -o stats stats.c && ./stats

Fixed. The comparison `v[i] < best` was tracking the minimum instead of the
maximum. Rebuilt and ran: mean=2.80 max=5.00
```

With no task it opens a REPL. Six tools (`read`, `write`, `edit`, `list`,
`grep`, `bash`); commands `/help`, `/new`, `/sessions`, `/effort`, `/think`,
`/yes`, `/ctx`, `/save`, `/quit`; `/image` and `/video` attach media
mid-conversation.

## Sessions

**The agent is a Session API client** ([API.md](../API.md)). Its system
prompt and tools become a session the server owns, and every turn sends only
what is new. If nothing is listening on the port it starts a server itself
(`-m` or `$QWASAR_MODEL` names the model), which stops when the agent does.
Conversations persist on the server: `--resume last` continues this
directory's most recent one, `/sessions` lists them, `/save` parks the
current one warm on disk. The agent links no engine: it is 170 KB and starts
instantly.

## Worth knowing

**Its tools run on your machine**, as you.

* **Reads run unattended; writes and commands ask first** (unless `--yes`).
  A declined action is reported back to the model so it can try something else.
* You can **type the next message while the model is still writing**; it runs
  when the turn finishes. Ctrl-C interrupts. No alternate screen — the
  transcript scrolls and copies like normal terminal output.
* `edit` is line-anchored search and replace: the quoted text must match a run
  of whole lines exactly once, or the edit is refused. No fuzzy matching.
* If `AGENT.md` exists in the working directory it is added to the system
  prompt as project guidance.
* A task stops after 24 rounds of tool calls (`--steps` changes it).
