# Plan Evaluation Issues

Scope checked:
- `Planning/overall_plan.md`
- `Planning/brandon_plan.md`
- `Planning/roger_plan.md`
- `CS 679 - Project Proposal - RnB.md`
- `evaluation_template.md`
- `project_instructions.txt`

Only school-project-level issues are listed below. I am not flagging production-grade concerns.

## 1. The plans narrow the project scope, but they do not say that this was an intentional change from instructor feedback

Evidence:
- The proposal includes two claims to reproduce: prompt optimization on GSM8k and test-time optimization on 2 LeetCode Hard problems.
- All three plans only cover prompt optimization on GSM8k.
- `project_instructions.txt` says the professor's feedback suggests keeping only prompt optimization is sufficient.

Why this is an issue:
- The narrowed scope is probably acceptable, but the plans never record that it was a deliberate response to professor feedback.
- As written, the plans look inconsistent with the approved proposal instead of intentionally revised.

## 2. The dataset/split change is not justified, and the chosen split makes the optimization gate very noisy

Evidence:
- The proposal targets a `35/10/5` split.
- The plans switch to `20/10/20`.
- Prompt updates are accepted or rejected based on validation accuracy on only 10 examples.
- With batch size 3 and 5 iterations, the optimizer will sample at most 15 training items, and with replacement it may cover even fewer unique items.

Why this is an issue:
- One validation question changes accuracy by 10 percentage points, so prompt acceptance can flip on a single example.
- The smaller train split also weakens the optimization signal compared with the original proposal, but the plans do not explain why this tradeoff is worth it.

## 3. Reproducibility controls are incomplete for a stochastic experiment

Evidence:
- Brandon's dataset prep mentions a fixed seed, but the shared config does not include a run seed.
- The optimization loop uses sampling with replacement.
- The judge temperature is set to `0.7`.
- The plans describe a single run trajectory, not repeated runs or seed-controlled reruns.

Why this is an issue:
- Two runs can produce different prompts and different accuracies even if the code is unchanged.
- Since `project_instructions.txt` emphasizes reproducibility, the plans should control or at least record the randomness in the optimization loop itself, not just in dataset sampling.

## 4. The self-judge experiment is confounded, so it cannot cleanly answer its stated question

Evidence:
- Core experiment: `~3B` solver + `~8B` judge.
- `config_ext_self_judge.yaml`: `~8B` solver + `~8B` self-judge.
- The extension is described as testing "self-judge vs. external judge."

Why this is an issue:
- This extension changes two things at once: judge mode and solver strength.
- If results change, you will not know whether the cause was self-judging or simply using a stronger 8B solver.

## 5. Optional extensions are treated too much like required report evidence

Evidence:
- The overall plan marks weak-judge and self-judge experiments as "nice-to-have."
- Brandon's plan still assigns a dedicated extension comparison chart and a report section to those experiments.
- The overall report outline also reserves a full "Extensions & Analysis" section for them.

Why this is an issue:
- If time runs out, the report plan has a built-in gap.
- It also risks pulling effort away from the one result that actually matters for the course project: a clean core reproduction.

## 6. The "problem difficulty" analysis is under-specified

Evidence:
- Brandon's analysis plan says to categorize GSM8k questions by "problem difficulty (number of steps)."
- No rule is given for how "number of steps" will be measured or labeled.

Why this is an issue:
- This makes the analysis subjective and hard to reproduce.
- It is also extra work that is not clearly needed by the evaluation rubric.
