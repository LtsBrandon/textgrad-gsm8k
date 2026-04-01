# TextGrad GSM8k Reproduction

Reproducing the **Prompt Optimization** claim from:
> Yuksekgonul et al., "Optimizing generative AI by backpropagating language model feedback," *Nature* 639, 609-616 (2025)

We use local LLMs via LM Studio (~3B solver + ~8B judge) instead of the paper's OpenAI cloud models.

---

## Repo Structure

```
Project/
├── Brandon/                 # Brandon's deliverables
│   ├── config/              #   Experiment config YAML files
│   ├── data/                #   GSM8k dataset splits + prep script
│   └── prompts/             #   4 prompt templates for the TextGrad loop
├── Roger/                   # Roger's deliverables
│   ├── src/                 #   Core Python modules (engine, optimizer, pipeline, etc.)
│   └── tests/               #   Unit tests + mock integration tests
├── Planning/                # Implementation plans (reference)
│   ├── overall_plan.md      #   Master plan: algorithm, contracts, repo structure
│   ├── brandon_plan.md      #   Brandon's detailed deliverables
│   └── roger_plan.md        #   Roger's detailed deliverables + module signatures
├── analysis/                # Results analysis scripts + plots
├── submission/              # Final .ipynb notebooks for hand-in
├── logs/                    # (runtime) LLM call logs — gitignored
├── results/                 # (runtime) Experiment results — gitignored
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Project Progress

### Phase 1: Planning — DONE

| Item | Status |
|------|--------|
| Overall plan (algorithm, interface contracts, experiment matrix) | Done |
| Brandon's plan (prompts, data, config, analysis deliverables) | Done |
| Roger's plan (module signatures, testing strategy, build order) | Done |
| Plan evaluation round 1 (6 issues found and fixed) | Done |
| Plan evaluation round 2 (3 issues found and fixed) | Done |

Plans are in `Planning/`. The key reference for integration is **`Planning/overall_plan.md` section 4** (Shared Interface Contracts) — both sides must follow these formats exactly.

---

### Phase 2: Brandon's Foundation — DONE

| Item | Location | Status |
|------|----------|--------|
| Initial solver prompt | `Brandon/prompts/system_prompt_init.txt` | Done |
| Evaluation prompt (loss function) | `Brandon/prompts/evaluation_prompt.txt` | Done |
| Gradient prompt (backward pass) | `Brandon/prompts/gradient_prompt.txt` | Done |
| Optimizer prompt (TGD rewrite) | `Brandon/prompts/optimizer_prompt.txt` | Done |
| Dataset prep script | `Brandon/data/prepare_dataset.py` | Done |
| Training set (20 examples) | `Brandon/data/train.jsonl` | Done |
| Validation set (15 examples) | `Brandon/data/val.jsonl` | Done |
| Test set (20 examples) | `Brandon/data/test.jsonl` | Done |
| Answer extraction test cases for Roger | `Brandon/data/answer_extraction_cases.json` | Done |
| Core experiment config | `Brandon/config/config_core.yaml` | Done (model names TBD) |
| Weak-judge extension config | `Brandon/config/config_ext_weak_judge.yaml` | Done (model names TBD) |
| Self-judge extension config | `Brandon/config/config_ext_self_judge.yaml` | Done (model names TBD) |

Placeholder template variables in each prompt match the contract in `overall_plan.md` section 4.3:

| Prompt File | Placeholders |
|-------------|-------------|
| `system_prompt_init.txt` | *(none — static initial text)* |
| `evaluation_prompt.txt` | `{question}`, `{model_answer}`, `{ground_truth}` |
| `gradient_prompt.txt` | `{system_prompt}`, `{aggregated_losses}` |
| `optimizer_prompt.txt` | `{current_prompt}`, `{gradient_feedback}` |

---

### Phase 3: Roger's Core Engine — NEXT UP

Roger builds 7 modules in `Roger/src/`. Recommended build order:

| Order | Module | What It Does | Est. Effort |
|-------|--------|-------------|-------------|
| 1 | `variable.py` | Text variable with gradient accumulation | ~30 min |
| 2 | `utils.py` | Config/dataset/template loading, logging helpers | ~45 min |
| 3 | `evaluation.py` | Answer extraction (regex) + exact match scoring | ~1 hr |
| 4 | `llm_client.py` | OpenAI SDK wrapper pointed at LM Studio | ~1 hr |
| 5 | `engine.py` | Forward pass, loss evaluation, gradient computation | ~2-3 hrs |
| 6 | `optimizer.py` | Textual Gradient Descent (TGD) prompt rewriter | ~1 hr |
| 7 | `pipeline.py` | Main loop: ties everything together, saves results | ~2-3 hrs |

Tests go in `Roger/tests/`. Use `MockLLMClient` for testing without a GPU — see `roger_plan.md` section 4 for details.

**Key files Roger needs from Brandon's side** (all ready):
- `Brandon/data/answer_extraction_cases.json` — 8 test cases for `evaluation.py` regex
- `Brandon/prompts/*.txt` — templates your code loads via `utils.load_prompt_template()`
- `Brandon/config/config_core.yaml` — schema your `utils.load_config()` must parse
- `Brandon/data/*.jsonl` — schema your `utils.load_dataset()` must parse

Full module signatures and class APIs are in **`Planning/roger_plan.md` section 2**.

---

### Phase 4: Integration & Experiments — AFTER ROGER'S CODE IS READY

**Brandon's remaining work** (blocked until Phase 3 is done):

| Item | Status |
|------|--------|
| Download models in LM Studio (~3B solver, ~8B judge) | Not started |
| Fill actual model names into config YAML files | Not started |
| Test prompts manually in LM Studio chat | Not started |
| Run core experiment seed 42 | Blocked on Roger |
| Run core experiment seed 123 | Blocked on Roger |
| Run extension experiments (if time permits) | Blocked on Roger |
| Build `analysis/analyze_results.py` | Blocked on results |
| Write submission notebooks | Blocked on results |

**How to run** (once Roger's pipeline is ready):
```bash
pip install -r requirements.txt
# Start LM Studio server at http://127.0.0.1:1234
python Roger/src/pipeline.py --config Brandon/config/config_core.yaml
```

---

### Phase 5: Analysis & Submission — LAST

| Item | Owner | Status |
|------|-------|--------|
| Accuracy-over-iterations line plot | Brandon | Not started |
| Prompt evolution table | Brandon | Not started |
| Per-question analysis (2-3 examples) | Brandon | Not started |
| Extension comparison bar chart (if run) | Brandon | Not started |
| `submission/submission_experiment.ipynb` | Both | Not started |
| `submission/submission_analysis.ipynb` | Brandon | Not started |
| `submission/submission_appendix.ipynb` | Both (optional) | Not started |

Final submission is `.ipynb` notebooks only. Git is for collaboration, not grading.

---

## Quick Reference

- **Interface contracts** (the handshake between Brandon and Roger): `Planning/overall_plan.md` section 4
- **Roger's module APIs**: `Planning/roger_plan.md` section 2
- **Brandon's prompt design rationale**: `Planning/brandon_plan.md` sections 1-2
- **Experiment matrix**: `Planning/overall_plan.md` section 6
- **Config schema**: `Planning/overall_plan.md` section 4.1

## Dependencies

```
pip install -r requirements.txt
```

Requires: `openai`, `pyyaml`, `datasets`, `matplotlib` — see `requirements.txt` for details on why we use the `openai` library with a local server.
