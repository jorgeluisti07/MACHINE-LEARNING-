# Backlog: ETESA/CND's actual day-ahead submission process

**Type:** open question, needs a human/domain-knowledge answer — not resolvable from the code or
the data alone.

**Status:** genuinely unresolved. Not blocking any currently-shipped work (see
`decisions.md` D-05 — the current constant-horizon methodology is already settled regardless of the
answer here), but would inform whether a supplementary experiment is worth building later.

## The question

Does Panama's day-ahead electricity market/grid operations (ETESA, CND, and/or the regional
MER/SIEPAC market) actually work on:

(a) **a single daily submission deadline** — one forecast issued once per day, covering the next
day's 24 hours with varying horizons (h+1 for the first hour, up to h+24 for the last), similar to
how many day-ahead market-clearing processes operate internationally; or

(b) **a continuously-updated operational forecast** — closer to what this project's current
constant-horizon, hourly-reissue design evaluates?

Both patterns are real in different contexts internationally (single-deadline for market clearing,
continuous reissue for operational/balancing forecasting) — an independent ML-methodology review
during this project explicitly could not confirm which applies to ETESA/CND specifically, and
flagged that as a real gap in confidence rather than guessing.

## Why it's in the backlog and not the roadmap

This isn't an implementation task — it's a fact to learn (from ETESA documentation, a domain
contact, or published market rules), the same category as the original #11 hour-convention question
(`MACHINE LEARNING RD CONTEXT.md` §12.2 point 11), which also sat unresolved until it could be
confirmed empirically/structurally rather than assumed. Unlike that question, this one may not be
resolvable from the data alone — it likely needs an external source (ETESA's own operational
documentation, or asking someone at ETESA/CND directly).

## What answering it would unlock

If the answer turns out to be (a) — single daily deadline, varying horizon — it would justify
building a supplementary varying-horizon experiment (mirroring the pattern already used for
`experiments/realistic_weather_ablation.py`): a genuinely new experiment, not a replacement for the
current primary methodology, evaluated with the same holdout discipline as everything else
(`decisions.md` D-08). See `specs/roadmap.md`'s "Varying-horizon / single-daily-issue forecast
redesign" entry for how this would be scoped if it becomes active work.

If the answer is (b), or the distinction turns out not to matter operationally for ETESA's actual
planning use case, no further action is needed beyond the explicit documentation already added in
`MACHINE LEARNING RD CONTEXT.md` §2 (the "Timing definition, stated explicitly" paragraph).

## Source

External review discussion (2026-08), independent code-review and ML-methodology subagent
findings; `MACHINE LEARNING RD CONTEXT.md` §2 (v26.22 fix); `decisions.md` D-05.
