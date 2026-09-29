# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Backup & Checkpoint Protocol

**Unpushed work does not durably exist.** This container can reset silently, mid-session, with no
reliable warning — two mild resets have rewound the local git pointer (recoverable via fetch); one
reset destroyed an entire isolated worktree with 7 unpushed, unverified commits, saved only by luck
(an agent's own chat transcript surviving independently). That's not a real safety net.

The backup rule (mandatory push-early protocol):
- Push a checkpoint to a `backup/<name>` branch as soon as there's any real progress — after the
  first commit, not the last. An unpushed commit, however correct, does not durably exist.
- Re-push periodically during long work, not just at the end. A reset can land mid-task.
- At the start of any session (or before any big task), verify local state actually matches
  origin — `git fetch origin <branch>`, compare `git log --oneline -1` locally vs. `origin/<branch>`.
  Never assume a clean local checkout reflects reality.
- Isolation (a worktree) protects against edit conflicts. It is NOT a substitute for pushing. An
  unpushed worktree is exactly as exposed to total loss as working directly on the main checkout.
- Only the final, reviewed result needs to land on the real working branch. Backup branches can be
  left in place or cleaned up later — their only job is to survive long enough to matter.

**Automated enforcement:** a `Stop` hook (`~/.claude/stop-hook-git-check.sh`, outside this repo —
machine config, not something `git push` carries forward) blocks ending a turn while the current
branch or any git worktree has uncommitted, untracked, or unpushed work. If a future session finds
this isn't being caught anymore, that hook may be missing and needs re-adding — see
`MACHINE LEARNING RD CONTEXT.md` for the exact script to restore it.

**The checkpoint rule (`/checkpoint`):** a hard usage cutoff isn't reliably self-detectable from
inside a session, but a warning sign (a usage indicator, degrading responses) might be caught
first. `/checkpoint` forces an immediate stop-and-save:
- Commits and pushes whatever exists — even incomplete, to a `checkpoint/*` branch if it's not
  safe for the main branch yet.
- Writes a complete handoff summary directly in the chat reply, not just to a file — because the
  next session's context comes from the conversation, not disk.
- Captures any durable lesson into `MACHINE LEARNING RD CONTEXT.md` if one emerged.
- Ends the turn without picking up further work.

Use it any time a session seems close to a hard cutoff.

## 6. No Emojis

**Never add emojis to notebooks, markdown docs, or code comments in this repo.**

- Applies to `.ipynb`, `.md` (including `MACHINE LEARNING RD CONTEXT.md` and `README.md`), and
  code comments in `.py` files.
- Use plain words instead of status emoji (e.g. "Fixed", "Open", "CRITICAL" — not checkmarks,
  warning signs, or clipboard icons). The word alone carries the meaning.
- Plain typographic characters that aren't emoji — arrows (`→`, `↔`), section markers (`§`),
  math/comparison symbols — are fine; they're notation, not decoration. Don't strip those.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
