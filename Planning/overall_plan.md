# TextGrad Reproduction — Overall Project Plan

## CS 679 Neural Networks — University of Waterloo, W2026

**Team**: Brandon Peng & Roger Zhu
**Paper**: Yuksekgonul et al., "Optimizing generative AI by backpropagating language model feedback," *Nature* 639, 609–616 (2025)

---

## 1. Project Overview

We are reproducing the **Prompt Optimization** claim from the TextGrad paper. The core finding: iterative "textual gradients" (natural language critique from a stronger LLM) can optimize a system prompt for a weaker LLM, improving its accuracy on GSM8k math problems.

**Original paper setup**: gpt-3.5-turbo (solver) + gpt-4o (judge/gradient engine)
**Our setup**: Small local LLM ~3B (solver) + Larger local LLM ~8B (judge), run via LM Studio

**Scope change from proposal**: The original proposal included two claims — (1) Prompt Optimization on GSM8k and (2) Test-Time Optimization on LeetCode Hard. After the professor reviewed the proposal, he highlighted the GSM8k target and commented: *"Am I correct that you will focus on just one type of input? That is sufficient."* Based on this feedback, we are intentionally narrowing scope to **Claim 1 only**. LeetCode Hard test-time optimization is dropped because (a) the professor confirmed one input type is sufficient, and (b) small local models (~3B–8B) would likely score near-zero on LeetCode Hard, making it infeasible to demonstrate any optimization signal.

**Scope**:
- **Core**: Reproduce the prompt optimization result on a 20/15/20 subset of GSM8k
- **Extensions (nice-to-have)**: Judge weaker vs. stronger than solver; self-judge vs. external judge

---

## 2. The TextGrad Algorithm for GSM8k Prompt Optimization

Below is the step-by-step algorithm we are implementing:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TextGrad Prompt Optimization                     │
│                                                                     │
│  INPUTS:                                                            │
│    P₀ = initial system prompt (CoT: "Think step by step...")        │
│    D_train = 20 training examples                                   │
│    D_val = 15 validation examples                                   │
│    Solver LLM (weak, ~3B)                                           │
│    Judge LLM (strong, ~8B)                                          │
│                                                                     │
│  FOR iteration = 1 to max_iterations (default 5):                   │
│                                                                     │
│    ┌─ STEP 1: FORWARD PASS ──────────────────────────────────────┐  │
│    │  Sample a mini-batch of 3 questions from D_train            │  │
│    │  For each question q_i in the batch:                        │  │
│    │    answer_i = Solver_LLM(P_current + q_i)                   │  │
│    │    correct_i = exact_match(answer_i, ground_truth_i)        │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│    ┌─ STEP 2: LOSS EVALUATION ───────────────────────────────────┐  │
│    │  For each INCORRECT answer in the batch:                    │  │
│    │    loss_i = Judge_LLM(evaluation_prompt,                    │  │
│    │                       question=q_i,                         │  │
│    │                       model_answer=answer_i,                │  │
│    │                       correct_answer=ground_truth_i)        │  │
│    │  loss_i is a natural-language critique of the answer        │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│    ┌─ STEP 3: GRADIENT (BACKWARD PASS) ──────────────────────────┐  │
│    │  Aggregate all loss critiques from the batch                │  │
│    │  gradient = Judge_LLM(gradient_prompt,                      │  │
│    │                       system_prompt=P_current,              │  │
│    │                       losses=aggregated_critiques)          │  │
│    │  gradient is feedback on how to CHANGE THE PROMPT           │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│    ┌─ STEP 4: OPTIMIZER STEP (TGD) ─────────────────────────────┐  │
│    │  P_candidate = Judge_LLM(optimizer_prompt,                  │  │
│    │                          current_prompt=P_current,          │  │
│    │                          gradient=gradient)                 │  │
│    │  P_candidate is a rewritten/improved prompt                 │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│    ┌─ STEP 5: VALIDATION GATE ──────────────────────────────────┐  │
│    │  val_acc_new = evaluate(Solver_LLM, P_candidate, D_val)    │  │
│    │  IF val_acc_new > val_acc_best:                             │  │
│    │    P_current = P_candidate                                  │  │
│    │    val_acc_best = val_acc_new                               │  │
│    │  ELSE:                                                      │  │
│    │    Keep P_current unchanged                                 │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  OUTPUT: P_final (the optimized prompt), all logs                   │
│  FINAL EVAL: evaluate(Solver_LLM, P_final, D_test)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Differences from Original Paper
| Aspect | Original Paper | Our Reproduction |
|--------|---------------|------------------|
| Solver LLM | gpt-3.5-turbo | Local ~3B model (quantized GGUF) |
| Judge LLM | gpt-4o | Local ~8B model (quantized GGUF) |
| API | OpenAI cloud API | LM Studio local API (OpenAI-compatible) |
| Train/Val/Test | DSPy splits (~200/300/1000) | Custom 20/15/20 subset (see note below) |
| Iterations | 12 | 5 (configurable up to 12) |
| Batch size | 3 | 3 (same) |
| Expected accuracy | 72.9% → 81.1% | Lower absolute, but expect upward trend |

**Why 20/15/20 instead of the proposal's 35/10/5?**
- The proposal's 35 training examples are more than needed — with batch size 3 and 5 iterations, only 15 examples are sampled (with replacement). 20 training examples provide sufficient coverage with less waste.
- The proposal's 5-example test set is too small — each question is worth 20 percentage points, making accuracy measurements essentially meaningless. 20 test examples bring granularity to 5 percentage points.
- The proposal's 10-example validation set was kept at 10 initially, but we increase it to **15** to reduce noise in the validation gate (each question = ~6.7 pp instead of 10 pp). With only 10 examples, a single borderline question flipping can cause spurious prompt acceptances/rejections.
- **Known limitation**: Even 15 validation examples are noisy. We should acknowledge this in the submission notebook and discuss how it may affect the optimization trajectory compared to the paper's ~300-example validation set.

---

## 3. Working Repository Structure

```
textgrad-gsm8k/
├── README.md                       # Optional internal docs (good for collaboration, not required in submission)
├── requirements.txt                # Python dependencies
├── config/
│   ├── config_core.yaml            # Main experiment config      [Brandon]
│   ├── config_ext_weak_judge.yaml  # Extension: weak judge       [Brandon]
│   └── config_ext_self_judge.yaml  # Extension: self-judge       [Brandon]
├── prompts/
│   ├── system_prompt_init.txt      # Initial solver prompt       [Brandon]
│   ├── evaluation_prompt.txt       # Loss function prompt        [Brandon]
│   ├── gradient_prompt.txt         # Backward pass prompt        [Brandon]
│   └── optimizer_prompt.txt        # TGD update prompt           [Brandon]
├── data/
│   ├── prepare_dataset.py          # Script to create splits     [Brandon]
│   ├── train.jsonl                 # 20 training examples        [Brandon]
│   ├── val.jsonl                   # 15 validation examples      [Brandon]
│   └── test.jsonl                  # 20 test examples            [Brandon]
├── src/
│   ├── __init__.py
│   ├── llm_client.py              # LLM API wrapper             [Roger]
│   ├── variable.py                # Variable class               [Roger]
│   ├── engine.py                  # Forward, loss, gradient      [Roger]
│   ├── optimizer.py               # TGD optimizer                [Roger]
│   ├── evaluation.py              # Answer extraction, scoring   [Roger]
│   ├── pipeline.py                # Main optimization loop       [Roger]
│   └── utils.py                   # Config, logging, templates   [Roger]
├── tests/
│   ├── test_variable.py           # Unit tests                   [Roger]
│   ├── test_evaluation.py         # Answer extraction tests      [Roger]
│   ├── test_engine_mock.py        # Mock LLM integration tests   [Roger]
│   └── mock_responses.json        # Canned LLM responses         [Roger]
├── logs/                           # Created at runtime
│   └── run_YYYYMMDD_HHMMSS.jsonl  # Per-run LLM call logs
├── results/                        # Created at runtime
│   └── run_YYYYMMDD_HHMMSS/
│       ├── metrics.json            # Accuracy per iteration
│       ├── prompts_history.json    # Prompt at each iteration
│       └── details.jsonl           # Per-question results
├── analysis/
│   └── analyze_results.py         # Visualization & analysis     [Brandon]
└── submission/
    ├── submission_experiment.ipynb # Main notebook for final hand-in        [Both]
    ├── submission_analysis.ipynb   # Results, plots, discussion             [Brandon]
    └── submission_appendix.ipynb   # Optional methods/setup appendix         [Both]
```

**Submission format note**: The professor now wants `.ipynb` files rather than a GitHub/repo submission. We will still use `git` during development because it is the safest way to collaborate and merge work, but the **final hand-in artifact** is the notebook(s) under `submission/`.

---

## 4. Shared Interface Contracts

These contracts are the handshake between Brandon's deliverables and Roger's code. **Both must follow these formats exactly.**

### 4.1 Config File Schema (YAML)

```yaml
# config/config_core.yaml
experiment_name: "core_3b_solver_8b_judge"

# LLM settings
# Why http://127.0.0.1:1234/v1?
#   - 127.0.0.1 is the loopback IP (same as "localhost") — it points to Brandon's own machine
#   - 1234 is LM Studio's default server port
#   - /v1 is required by the OpenAI Python SDK to identify the API version path
#   - LM Studio exposes an OpenAI-compatible REST API at this address, so Roger's code
#     uses the standard `openai` Python library pointed here instead of to OpenAI's cloud.
#     This means no code changes are needed if we ever switch to a real OpenAI API key —
#     just swap the api_base_url and api_key fields below.
api_base_url: "http://127.0.0.1:1234/v1"    # LM Studio local server (Brandon's PC)
solver_model: "model-name-for-solver"         # LM Studio model identifier
judge_model: "model-name-for-judge"           # LM Studio model identifier
api_key: "lm-studio"                          # LM Studio accepts any non-empty string

# Generation settings
solver_temperature: 0.0          # Deterministic for reproducibility
solver_max_tokens: 512           # Max tokens for solver response
judge_temperature: 0.7           # Some creativity for feedback
judge_max_tokens: 1024           # Judge needs more space for critiques

# Optimization settings
batch_size: 3                    # Questions per iteration
max_iterations: 5                # Optimization iterations
sampling_with_replacement: true  # Match paper's sampling strategy
random_seed: 42                  # Seed for batch sampling RNG (reproducibility)

# Paths
prompt_dir: "prompts/"
data_dir: "data/"
log_dir: "logs/"
results_dir: "results/"

# Dataset files (relative to data_dir)
train_file: "train.jsonl"
val_file: "val.jsonl"
test_file: "test.jsonl"

# Extension flags
self_judge: false    # If true, solver also acts as judge
# (when self_judge is true, judge_model is ignored and solver_model is used for everything)
```

### 4.2 Dataset Format (JSONL)

Each line in `train.jsonl`, `val.jsonl`, `test.jsonl`:

```json
{
  "id": "gsm8k_train_0042",
  "question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
  "answer_text": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = <<9*2=18>>$18 every day at the farmer\u2019s market.\n#### 18",
  "ground_truth": 18
}
```

**Fields**:
- `id`: unique identifier
- `question`: the math problem text
- `answer_text`: full solution with reasoning (from GSM8k)
- `ground_truth`: the extracted numerical answer (integer or float)

### 4.3 Prompt Template Format

Plain text files with Python `str.format()` / f-string style `{placeholder}` variables.

**Example** (`prompts/evaluation_prompt.txt`):
```
Below is a math question, the student's answer, and the correct answer.

Question: {question}

Student's Answer: {model_answer}

Correct Answer: {ground_truth}

Critically evaluate the student's reasoning. Identify specific errors in their approach.
Explain what went wrong and why the answer is incorrect.
```

**Placeholder variables available per template**:

| Template | Available Placeholders |
|----------|----------------------|
| `system_prompt_init.txt` | *(none — this is static initial text)* |
| `evaluation_prompt.txt` | `{question}`, `{model_answer}`, `{ground_truth}` |
| `gradient_prompt.txt` | `{system_prompt}`, `{aggregated_losses}` |
| `optimizer_prompt.txt` | `{current_prompt}`, `{gradient_feedback}` |

### 4.4 Logging Format (JSONL)

Every LLM call gets logged as one JSON line in `logs/run_*.jsonl`:

```json
{
  "timestamp": "2026-03-20T14:30:00",
  "iteration": 2,
  "step": "forward",
  "role": "solver",
  "model": "qwen2.5-3b-q4",
  "messages": [
    {"role": "system", "content": "You will answer a math question..."},
    {"role": "user", "content": "Janet's ducks lay 16 eggs..."}
  ],
  "response": "Janet sells 16 - 3 - 4 = 9 eggs...\nAnswer: $18",
  "tokens_prompt": 145,
  "tokens_completion": 89,
  "latency_ms": 2340
}
```

### 4.5 Results Format

**`results/run_*/metrics.json`**:
```json
{
  "experiment_name": "core_3b_solver_8b_judge",
  "config_file": "config/config_core.yaml",
  "iterations": [
    {
      "iteration": 0,
      "phase": "baseline",
      "val_accuracy": 0.30,
      "test_accuracy": 0.25,
      "num_val": 15,
      "num_test": 20
    },
    {
      "iteration": 1,
      "phase": "optimization",
      "train_batch_accuracy": 0.33,
      "val_accuracy": 0.40,
      "val_improved": true,
      "prompt_updated": true
    }
  ],
  "final_test_accuracy": 0.45,
  "initial_test_accuracy": 0.25
}
```

**`results/run_*/prompts_history.json`**:
```json
[
  {
    "iteration": 0,
    "prompt": "You will answer a mathematical reasoning question. Think step by step...",
    "val_accuracy": 0.30
  },
  {
    "iteration": 1,
    "prompt": "You will answer a mathematical reasoning question. Restate the problem...",
    "gradient_feedback": "The prompt should instruct the model to break down calculations...",
    "val_accuracy": 0.40
  }
]
```

**`results/run_*/details.jsonl`** — Per-question results for every evaluation run (baseline, each iteration's validation, final test). One JSON line per question:
```json
{
  "iteration": 0,
  "split": "test",
  "id": "gsm8k_test_0012",
  "question": "Janet's ducks lay 16 eggs per day...",
  "ground_truth": 18,
  "model_response": "Let me solve step by step... Answer: $18",
  "extracted_answer": 18,
  "is_correct": true
}
```

**Fields**:
- `iteration`: which optimization iteration (0 = baseline)
- `split`: which dataset split was being evaluated (`"train_batch"`, `"val"`, or `"test"`)
- `id`: the question's unique ID from the dataset JSONL
- `question`: the problem text
- `ground_truth`: the correct numerical answer
- `model_response`: the solver's full text response
- `extracted_answer`: the number extracted from the response (or `null` if extraction failed)
- `is_correct`: whether extracted_answer matched ground_truth

> Brandon's per-question analysis (which questions improved/regressed after optimization) depends on this file. Roger's pipeline must write it during every evaluation call (baseline, validation, final test).

---

## 5. Integration Workflow

This is how Brandon's and Roger's work come together:

```
Step 1: SHARED DEVELOPMENT WORKFLOW
├── Roger → commits src/ and tests/ to a shared git repo
├── Brandon → commits prompts/, data/, config/ to the same git repo
├── Shared git repo can be a private GitHub repo, another private remote, or a local/shared git setup
└── Both pull/merge from git as they iterate

Note: git remains the recommended collaboration tool even though the final course submission is notebook-only.

Step 2: BRANDON PREPARES ENVIRONMENT
├── Downloads models in LM Studio (solver ~3B, judge ~8B)
├── Starts LM Studio server on localhost:1234
├── Edits config_core.yaml with exact model names from LM Studio
└── Runs: pip install -r requirements.txt

Step 3: BRANDON RUNS EXPERIMENTS
├── python src/pipeline.py --config config/config_core.yaml
├── Pipeline loads Brandon's config → loads Brandon's prompts → loads Brandon's data
├── Pipeline calls Roger's code (engine, optimizer, evaluation)
├── Roger's code calls LLMs via Brandon's LM Studio server
├── All calls logged to logs/
└── Results saved to results/

Step 4: BRANDON ANALYZES
├── python analysis/analyze_results.py --results results/run_*/
├── Generates plots and tables for the submission notebooks
└── Writes the notebook narrative/discussion

Step 5: EXTENSIONS (if time permits)
├── Brandon creates config_ext_weak_judge.yaml (same model for both roles)
├── Brandon creates config_ext_self_judge.yaml (self_judge: true)
├── Reruns pipeline with each config
└── Compares results across configs

Step 6: PACKAGE FINAL SUBMISSION
├── Move essential setup, method, results, and discussion into `submission/*.ipynb`
├── Do not rely on README/report files for required grading content
└── Submit `.ipynb` files only
```

---

## 6. Experiment Matrix

### Core Experiment (Must complete)
| Config | Solver | Judge | Iterations | Expected Outcome |
|--------|--------|-------|------------|-----------------|
| `config_core.yaml` | ~3B model | ~8B model | 5 | Accuracy improves from baseline; prompt becomes more detailed |

### Extension Experiments (Nice-to-have)

**Important design principle**: each extension changes exactly **one variable** from the core experiment so results are attributable.

| Config | Solver | Judge | Variable Changed | Purpose |
|--------|--------|-------|-----------------|---------|
| Core (baseline) | ~3B | ~8B (external) | — | Reference result |
| `config_ext_weak_judge.yaml` | ~3B | ~3B (external) | Judge strength ↓ | Show that a weaker judge gives less/no improvement |
| `config_ext_self_judge.yaml` | ~3B | ~3B (self) | Judge mode (self) | Show whether self-judging works when model critiques its own output |

Note: the self-judge extension uses the **same ~3B model** as solver and judge (not 8B), so that the only difference from `config_ext_weak_judge` is external-judge vs. self-judge. This avoids confounding judge mode with model strength.

### Metrics to Report
- Accuracy per iteration (line plot)
- Initial prompt vs. final prompt text (side-by-side)
- Per-question accuracy improvement/regression
- Extension comparison (bar chart: core vs. weak-judge vs. self-judge)

---

## 7. Submission Outline (Mapped to Evaluation Rubric)

| Section | Rubric Item | Owner |
|---------|------------|-------|
| 1. Introduction & Paper Summary | Description of paper's methods, main claims | Brandon |
| 2. Method Description | What we implemented, how TextGrad works | Both |
| 3. Implementation Details | Decisions not in paper (model choices, dataset size, prompts) | Both |
| 4. Experimental Setup | Dataset, models, config, compute resources | Brandon |
| 5. Results | Accuracy plots, prompt evolution, comparison tables | Brandon |
| 6. Discussion | Do results support the paper's claims? Limitations. **If extensions were run**, include extension results here as subsections (not a separate section). | Both |
| 7. Code Documentation | Well-documented code, repo structure, and notebook setup/method notes | Roger |

> **Note**: The final submission is notebook-based, so the rubric-facing content above should appear in notebook markdown/code cells. A README can still exist for internal collaboration, but the grader should not need it.
>
> The submission structure is designed to stand on its own with just the core experiment (sections 1–6). Extension results (weak-judge, self-judge) are folded into Discussion as optional subsections. If extensions are not completed, the Discussion section still works – it simply discusses the core result without extension comparisons.

---

## 8. Dependencies

```
# requirements.txt
openai>=1.0.0        # OpenAI-compatible client — used to talk to LM Studio's local API
                     # at http://127.0.0.1:1234/v1. LM Studio mirrors the OpenAI REST
                     # API format, so the standard openai Python SDK works out of the box
                     # without writing any raw HTTP code. This also means the same codebase
                     # works against the real OpenAI cloud by just changing the config URL.
pyyaml>=6.0          # Config file parsing
datasets>=2.0.0      # HuggingFace datasets (for GSM8k download)
matplotlib>=3.7.0    # Plotting for analysis
```

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Local 3B model is too weak for GSM8k | Try a 7-8B model as solver instead; document the finding |
| Judge model gives poor/generic feedback | Iterate on prompt templates; try different judge models |
| LM Studio API is slow/unstable | Add retry logic, increase timeouts in config |
| Validation accuracy never improves | This is still a valid result — document that smaller models struggle to produce useful textual gradients; compare with paper's GPT-4o results |
| Context window overflow | Monitor token counts in logs; truncate if needed; use smaller batch |
| Non-determinism across runs | The judge uses temperature 0.7, so two identical runs may produce different gradients and different final prompts. Mitigations: (1) `random_seed` in config controls batch sampling, (2) solver uses temperature 0 for deterministic answers, (3) run the core experiment **at least twice** with different seeds (e.g., 42 and 123) and report both trajectories in the submission notebook. Acknowledge remaining non-determinism from the judge in the discussion. |
