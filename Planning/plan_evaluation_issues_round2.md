# Plan Reevaluation - Remaining Issues

Most of the earlier issues were fixed. The updated plans are materially better.

Only the remaining issues are listed below.

## 1. The overall plan still contains one stale split reference

Evidence:
- `Planning/overall_plan.md` section 1 still says the core experiment uses a `20/10/20` GSM8k subset.
- The same file later switches to `20/15/20` in the algorithm, split rationale, and repository structure.
- `Planning/brandon_plan.md` also uses `20/15/20`.

Why this is an issue:
- This is now the main cross-file inconsistency left in the plans.
- If someone follows only the top-level scope summary, they can prepare the wrong validation split.

## 2. The per-question analysis still depends on an underspecified output contract

Evidence:
- Brandon's plan requires per-question analysis and concrete improved/regressed examples.
- The overall plan's repository structure includes `results/run_*/details.jsonl`.
- But the overall results-format section only specifies `metrics.json` and `prompts_history.json`.
- Roger's integration checklist also only requires `metrics.json` and `prompts_history.json`.

Why this is an issue:
- Brandon's analysis needs per-question records, but the save format for that artifact is still not clearly contracted.
- This can easily become an integration problem even if both people follow their own plans.

## 3. Repeated runs are now in the narrative, but not yet in the execution checklist

Evidence:
- Brandon's plan now says to run the core experiment at least twice with different seeds.
- The overall plan also recommends at least two seeded runs in the risk-mitigation section.
- But Brandon's deliverables checklist still only says `Core experiment run completed`.

Why this is an issue:
- The repeated-run requirement is easy to forget if it is not turned into a checklist item.
- That would weaken the reproducibility evidence even though the updated narrative now correctly asks for it.
