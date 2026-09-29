---
description: Spec-driven workflow for a change to the forecasting pipeline — interview, spec, staged implementation, verification, adversarial review, ship.
---

# /feature-spec

This formalizes the workflow this project has already been following informally for every
model-affecting or methodology-affecting change (the leakage fixes, the P5 sweep, the realistic-
weather ablation, the weather hour-convention fix). It is not new process — it is that process,
written down so it doesn't depend on being re-derived from memory each time.

Use this for anything that changes model behavior, features, evaluation methodology, or reported
numbers. Skip it for pure hygiene (comment cleanup, README wording, banner removal) — those don't
need a spec, just do them and commit.

**Read `decisions.md` and `mistakes.md` before step 1.** Do not re-propose something already
decided (check `decisions.md`) or re-introduce a bug already found and ruled out (check
`mistakes.md`). If a proposal directly contradicts a locked decision, say so explicitly and ask
before proceeding — don't silently override it.

## Step 0 — Orient

Read, in order: `CLAUDE.md`, `decisions.md`, `mistakes.md`, `specs/roadmap.md`, and the relevant
section(s) of `MACHINE LEARNING RD CONTEXT.md` for background this task touches. Confirm current
code behavior against what these docs claim — don't assume the docs are still accurate; verify with
a grep/read before relying on a claim (this is exactly the check that caught M-06).

## Step 1 — Push an empty backup branch

Before writing any code: `git push origin HEAD:backup/<short-name>` (or update an existing backup
branch). This is not optional — see `CLAUDE.md` §5. Work that only exists locally does not durably
exist in this environment.

## Step 2 — Interview

One round of targeted questions before writing any code, covering:
- **Scope.** What's explicitly in scope, what's explicitly out of scope.
- **Decisions/tradeoffs.** Where more than one reasonable approach exists, name them and ask which
  one, rather than picking silently (`CLAUDE.md` §1).
- **Context.** Anything about the data, the deployment context, or prior attempts (check
  `mistakes.md`/`decisions.md` first) that changes the right approach.

Do not proceed to Step 3 until these are answered. If the user says "just proceed" or the answer is
obvious from `decisions.md`/`mistakes.md`, say so explicitly and move on — don't skip the step
silently.

## Step 3 — Write the spec

Three files, in `specs/` (or a task-specific subfolder if the change is large enough to warrant
one):

- **`requirements.md`** — scope, explicit out-of-scope, the decisions/tradeoffs from Step 2 and
  which way they were resolved, and any context that shaped the approach.
- **`plan.md`** — staged steps. Each stage must end in a commit + push — never batch multiple
  stages into one commit at the end. If a stage touches leakage-sensitive code (any per-window
  rebuild, any lag/shift feature, any calibration/normalization step, any train/validation/test
  split), say so explicitly in the plan so the verification step below doesn't get skipped for it.
- **`validation.md`** — concrete pass/fail checks (see the checklist in Step 6; write the specific
  instantiation of it for this change here, not just "verify it works").

## Step 4 — Confirm with the user

Show the three files. Do not start implementing until the user confirms the spec, unless they've
explicitly pre-authorized proceeding without a confirmation round for this specific task.

## Step 5 — Implement, staged

Follow `plan.md`'s stages. Each stage: implement, verify against real data (Step 6), commit, push.
Never batch to the end — `CLAUDE.md` §5's backup rule exists because unpushed work does not durably
exist in this environment, and that risk compounds with every stage left uncommitted.

## Step 6 — Verify against the real data/pipeline, not assumptions

This is the checklist this project has actually applied for every real change so far (see
`MACHINE LEARNING RD CONTEXT.md` §12.6, §12.7, §12.8, §12.11 for worked examples) — instantiate the
relevant parts of it in `validation.md`, don't skip items just because they weren't listed there
generically:

- **Leakage re-verified per window, not asserted.** For any new or changed feature that depends on
  a full-dataset or full-window statistic (a ratio, a mean, a profile), confirm it's refit
  training-only per rolling window (see `mistakes.md` M-01, M-04) — ideally with a unit test that
  perturbs post-cutoff data and asserts the feature is invariant to it, not just a read-through.
- **Results re-derived from `results_summary.csv`/`forecasts.csv`, never trusted from a printed
  cell or a remembered number.** Run the full rolling pipeline, read the actual output files, and
  cite those numbers — a stale hardcoded figure is exactly M-07.
- **Compare against the oracle/realistic split where the change touches weather features**, or
  against the relevant existing baseline (Naive, Weekly-Naive, Linear) where it touches something
  else — an isolated new-model-only number without a baseline comparison isn't verification.
- **Reproducibility bounded by the documented ±0.05pp seed-noise band** (see `README.md` §
  Assumptions, "Run-to-run variability"). A result that differs from a prior run by less than that
  band is not evidence of a real change — don't over-interpret small deltas as findings.
- **Index integrity check still passes** (gap-free, duplicate-free hourly index) — if the change
  touches data loading or joining, re-confirm this explicitly (`mistakes.md` M-09).
- **`py_compile` clean on both `.py` files; per-cell `ast.parse` clean on the notebook.** Minimum
  bar before claiming "done," even for comment-only changes.
- **Holdout discipline respected** (`decisions.md` D-08): if this change is a genuine correctness
  fix, one re-run against the test period is legitimate. If it's a new modeling variant/roadmap
  item, verify against the **validation split only** — do not touch the test period, and say so
  explicitly in `validation.md` so the reviewer in Step 7 can confirm it.

## Step 7 — Reflect (adversarial review)

After the change is implemented and verified, an **independent pass on a higher-reasoning model
than whatever implemented it** reviews the result — skeptical of the implementer's own narrative,
not confirming it. This mirrors the adversarial-subagent reviews already used for the calibration
leak fix, the P5 sweep, and the weighted-ensemble weight search (`MACHINE LEARNING RD CONTEXT.md`
§8 rows 3, 17; §12.6).

The reviewer must, at minimum:
1. **Re-derive the reported numbers independently** — read the actual CSV output, don't trust the
   implementer's summary of it.
2. **Check for missed leakage instances** — specifically re-trace any new full-dataset statistic,
   any new lag/shift feature's timestamp-relative framing, and any new validation-tuning step's
   date range against the test period (the exact failure modes in M-01, M-05, M-08).
3. **Check verification calibration** — did `validation.md`'s checks actually get run, or just
   asserted as done? Distinguish "I ran the pipeline and read the CSV" from "this should work."
4. **Check whether a "finding" is new or a recurrence of an existing `mistakes.md` rule.** If the
   reviewer surfaces something that looks like a new incident, check it against every `M-` entry
   first — a recurrence of an already-documented mistake means the *rule* needs sharpening (why
   didn't it prevent this?), not a new `M-` entry for the same underlying issue. Only add a new
   `M-` entry for a genuinely distinct failure mode.
5. **Report plainly**, including "found nothing" as a valid, useful outcome — don't manufacture a
   finding to seem thorough, and don't rubber-stamp without actually re-deriving anything.

## Step 8 — Ship

- Write a narrative record in `specs/completed/<slug>.md` — what shipped, why, the evidence, the
  before/after numbers if applicable, verification performed, and revert instructions (see any file
  in `specs/completed/` for the expected shape).
- Add or update the corresponding entry in `specs/roadmap.md` — move it from "open" to "shipped,"
  or remove it if it was never a roadmap item.
- If the change produced a durable methodological decision (not just a bug fix), add it to
  `decisions.md` with the next `D-` id. If it was a genuine incident, add it to `mistakes.md` with
  the next `M-` id and an actionable rule, unless Step 7.4 determined it's a recurrence of an
  existing rule.
- Final commit + push, per `CLAUDE.md` §5 — the same backup discipline applies at the end as it did
  at the start.
