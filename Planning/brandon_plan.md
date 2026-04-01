# Brandon's Deliverables — TextGrad GSM8k Reproduction

> **Your role**: Prompt designer, dataset curator, experiment runner, analyst, submission writer.
> Roger builds the engine; you provide the fuel (prompts, data, config) and drive the car (run experiments, analyze results).

---

## 1. Prompt Templates to Design

You need to create **4 prompt template files** in the `prompts/` folder. These are the most critical part of your contribution — the quality of these prompts directly determines whether the TextGrad loop produces useful feedback and prompt improvements.

### 1.1 `prompts/system_prompt_init.txt` — Initial Solver Prompt

**Purpose**: The starting system prompt that the solver LLM uses to answer math questions. This is the variable being optimized. The TextGrad loop will iteratively improve this prompt.

**No placeholders** — this is the initial static text.

**Design guidance**:
- Start with the paper's initial prompt as a baseline: a Chain-of-Thought instruction telling the model to think step by step
- Must instruct the model to output its final answer in a parseable format: `Answer: $VALUE`
- Keep it concise — small models have limited context windows

**Example starting point** (adapt for small local models):
```
You will answer a mathematical reasoning question. Think step by step.
The last line of your response should be of the following format:
'Answer: $VALUE' where VALUE is a numerical value.
```

**Tips for local small models**:
- Be more explicit about the output format (small models are less instruction-following)
- Consider adding: "Show your work clearly. Use simple arithmetic."
- Test this prompt manually in LM Studio first before running the full pipeline

---

### 1.2 `prompts/evaluation_prompt.txt` — Loss Function (Judge Evaluates Answer)

**Purpose**: The judge LLM uses this prompt to evaluate the solver's answer. It produces a natural-language critique — this is the "loss signal" in TextGrad terminology.

**Placeholders**: `{question}`, `{model_answer}`, `{ground_truth}`

**Design guidance**:
- The judge should critically analyze the solver's reasoning step by step
- It should identify specific mathematical errors (wrong operations, incorrect arithmetic, missing steps)
- It should explain WHY the answer is wrong, not just that it is wrong
- This is only called for INCORRECT answers (Roger's code will skip correct ones)

**Template structure**:
```
You are a math teacher evaluating a student's work.

Question: {question}

Student's Answer:
{model_answer}

Correct Answer: {ground_truth}

Carefully analyze the student's reasoning step by step.
Identify the specific errors in their approach — where did the logic
go wrong? What mathematical mistakes were made?
Be specific and constructive in your critique.
```

**Important**: The output of this prompt becomes the "loss" that feeds into the gradient step. If the critique is vague ("the answer is wrong"), the gradient will be useless. Make the judge produce **specific, actionable** feedback.

> **Integration note**: Multiple loss critiques from a batch are aggregated by Roger's code into a single string with the format:
> ```
> Feedback 1:
> (critique for question 1)
>
> ---
>
> Feedback 2:
> (critique for question 2)
> ```
> This aggregated string gets inserted into `{aggregated_losses}` in the gradient prompt. Design your gradient_prompt.txt to work well with this format.

---

### 1.3 `prompts/gradient_prompt.txt` — Backward Pass (Critique the System Prompt)

**Purpose**: Given the aggregated loss critiques from the batch, the judge produces feedback on how to **change the system prompt** to prevent those errors. This is the "textual gradient" — analogous to ∂Loss/∂Prompt.

**Placeholders**: `{system_prompt}`, `{aggregated_losses}`

**Design guidance**:
- This prompt must connect the observed errors (from the evaluation) back to deficiencies in the system prompt
- The judge should suggest what instructions to add, remove, or modify in the system prompt
- The feedback should be about the PROMPT, not about individual answers

**Template structure**:
```
You are an expert prompt engineer. A language model is using the
following system prompt to solve math problems, but it is making errors.

Current System Prompt:
"{system_prompt}"

Below are critiques of the model's incorrect answers on recent problems:

{aggregated_losses}

Based on these errors, provide specific feedback on how the system prompt
should be changed to help the model avoid these types of mistakes.
Focus on what instructions should be added, modified, or clarified
in the system prompt. Do NOT solve the individual problems — instead,
suggest general improvements to the prompt's instructions.
```

---

### 1.4 `prompts/optimizer_prompt.txt` — TGD Update (Rewrite the Prompt)

**Purpose**: Given the current prompt and the gradient feedback, the judge rewrites the prompt to incorporate the suggestions. This is the optimizer step — analogous to `P_new = P_old - lr * gradient`.

**Placeholders**: `{current_prompt}`, `{gradient_feedback}`

**Design guidance**:
- The judge must produce a complete, revised system prompt (not a diff or partial edit)
- The new prompt should incorporate the feedback while keeping the core structure
- Must preserve the output format instruction (`Answer: $VALUE`)
- Should not grow unboundedly — encourage concise improvements

**Template structure**:
```
You are an expert prompt engineer. Your task is to improve a system prompt
based on feedback about its weaknesses.

Current System Prompt:
"{current_prompt}"

Feedback on how to improve it:
{gradient_feedback}

Write an improved version of the system prompt that addresses the feedback.
The improved prompt must:
1. Incorporate the suggested improvements
2. Keep the instruction for the model to show step-by-step reasoning
3. End with the instruction that the final answer must be in the format:
   'Answer: $VALUE' where VALUE is a numerical value
4. Be concise — do not exceed 200 words

Output ONLY the new system prompt, with no additional commentary.
```

---

## 2. Dataset Preparation

### 2.1 `data/prepare_dataset.py` — Dataset Split Script

**Purpose**: Download GSM8k from HuggingFace, sample a 20/15/20 split, save as JSONL.

**What the script does**:
1. Load the GSM8k dataset: `load_dataset("openai/gsm8k", "main")`
2. From the **train** split (7,473 examples), randomly sample 35 examples → 20 train + 15 validation
3. From the **test** split (1,319 examples), randomly sample 20 examples → 20 test
4. Extract the ground truth number from each answer using regex: `#### (\-?[0-9\.,]+)`
5. Save each split as JSONL with the schema from the overall plan:
   ```json
   {"id": "gsm8k_train_0042", "question": "...", "answer_text": "...", "ground_truth": 18}
   ```
6. Use a fixed random seed (e.g., `seed=42`) for reproducibility

**Tip**: Remove commas from numbers (e.g., "1,200" → "1200") during ground truth extraction.

### 2.2 Answer Extraction Utility

You'll also need a function to extract the final numerical answer from the **solver's** response. This is used during evaluation.

**Strategy (in priority order)**:
1. Look for `Answer: $VALUE` or `Answer: VALUE` pattern (since our prompt instructs this format)
2. Fallback: extract the last number in the response
3. Fallback: return `None` (counts as incorrect)

**Regex patterns**:
```python
# Primary: match our instructed format
r"Answer:\s*\$?([\-\d,\.]+)"

# Fallback: last number in text
r"([\-\d,\.]+)\s*$"
```

> Note: Roger will implement this in `src/evaluation.py`, but you should define and test the regex patterns. Provide him with test cases (example model outputs and expected extracted answers).

---

## 3. Config Files

### 3.1 `config/config_core.yaml` — Core Experiment

```yaml
experiment_name: "core_3b_solver_8b_judge"

# Your LM Studio server is reachable at http://127.0.0.1:1234
# The /v1 suffix is required by the openai Python library (the API version path).
# Roger's code uses the `openai` Python SDK pointed at this address instead of
# writing raw HTTP requests — it's cleaner, handles retries, and parses responses
# automatically. 127.0.0.1 and localhost are identical; using the IP is more explicit.
api_base_url: "http://127.0.0.1:1234/v1"
solver_model: "SET_THIS_TO_YOUR_SOLVER_MODEL_NAME"   # Copy exact name from LM Studio
judge_model: "SET_THIS_TO_YOUR_JUDGE_MODEL_NAME"     # Copy exact name from LM Studio
api_key: "lm-studio"                                 # LM Studio ignores this but openai SDK requires it

solver_temperature: 0.0
solver_max_tokens: 512
judge_temperature: 0.7
judge_max_tokens: 1024

batch_size: 3
max_iterations: 5
sampling_with_replacement: true
random_seed: 42                                      # Controls batch sampling; change for repeated runs

prompt_dir: "prompts/"
data_dir: "data/"
log_dir: "logs/"
results_dir: "results/"

train_file: "train.jsonl"
val_file: "val.jsonl"
test_file: "test.jsonl"

self_judge: false
```

### 3.2 `config/config_ext_weak_judge.yaml` — Extension: Weak Judge

Same as core, but `judge_model` is the same weak ~3B model:
```yaml
experiment_name: "ext_weak_judge_3b_both"
judge_model: "SET_THIS_TO_YOUR_SOLVER_MODEL_NAME"   # Same as solver!
# ... rest same as core
```

### 3.3 `config/config_ext_self_judge.yaml` — Extension: Self-Judge

Uses the **same ~3B solver** as core, but with `self_judge: true` so the 3B model critiques its own output. This isolates the self-judge variable — the only difference from `config_ext_weak_judge` is external vs. self-judge mode:
```yaml
experiment_name: "ext_self_judge_3b"
solver_model: "SET_THIS_TO_YOUR_SOLVER_MODEL_NAME"   # Same 3B as core
judge_model: "SET_THIS_TO_YOUR_SOLVER_MODEL_NAME"    # Same 3B (self-judging)
self_judge: true
# ... rest same as core
```

---

## 4. LM Studio Setup

### Models to Download
- **Solver (~3B)**: Choose one quantized GGUF model. Options to try:
  - Qwen 2.5 3B Instruct (Q4_K_M quantization) — good at math
  - Llama 3.2 3B Instruct (Q4_K_M)
  - Phi-3.5 Mini 3.8B Instruct (Q4_K_M)

- **Judge (~8B)**: Choose one quantized GGUF model. Options to try:
  - Qwen 2.5 7B Instruct (Q4_K_M) — strong at reasoning
  - Llama 3.1 8B Instruct (Q4_K_M)
  - Gemma 2 9B Instruct (Q4_K_M)

### LM Studio Configuration
1. Load both models simultaneously in LM Studio
2. Start the local server — LM Studio shows **Reachable at: http://127.0.0.1:1234**
3. Note the exact model identifiers from LM Studio's model list
4. Enter those identifiers in the config YAML files under `solver_model` / `judge_model`
5. Test the connection with a quick curl (note the `/v1` path):
   ```bash
   curl http://127.0.0.1:1234/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "your-model-name", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
   ```

> **Why /v1?** The `openai` Python library that Roger's code uses expects the base URL to end with `/v1` — this is the standard OpenAI API versioning path. LM Studio routes requests correctly when this path is included. Without it, the SDK will construct incorrect endpoint URLs (e.g., `/v1/v1/chat/completions`).

### Important Notes
- LM Studio can load multiple models, but only one runs at a time — it swaps on demand
- With 16GB VRAM + 32GB RAM, both models fit with CPU offloading for the larger one
- Monitor VRAM usage during long runs; if it OOMs, use a more aggressive quantization (Q3_K_M)

---

## 5. Experiment Execution

### Running the Core Experiment
```bash
# From the repo root
python src/pipeline.py --config config/config_core.yaml
```

### Expected Behavior
1. Pipeline loads config, prompts, and dataset
2. Runs baseline evaluation on val and test sets (iteration 0)
3. Runs the optimization loop (iterations 1–5)
4. Each iteration takes ~5-15 minutes depending on model speed
5. Total core experiment: ~30-90 minutes
6. Logs written to `logs/`, results to `results/`

### Repeated Runs for Reproducibility
The judge LLM uses temperature 0.7, introducing non-determinism. Run the core experiment **at least twice** with different random seeds to show the optimization trend is consistent:
```bash
# Run 1
python src/pipeline.py --config config/config_core.yaml
# Then edit config_core.yaml: random_seed: 123
# Run 2
python src/pipeline.py --config config/config_core.yaml
```
Report both trajectories in the results section. If the trend (accuracy improving over iterations) is consistent across seeds, that strengthens the reproducibility claim even if exact numbers differ.

### Running Extensions (If Time Permits)
```bash
python src/pipeline.py --config config/config_ext_weak_judge.yaml
python src/pipeline.py --config config/config_ext_self_judge.yaml
```

### Manual Prompt Testing
Before running the full pipeline, test each prompt template manually:
1. Open LM Studio chat interface
2. Paste the initial system prompt, send a GSM8k question → check if it answers correctly
3. Paste the evaluation prompt (fill placeholders) → check if it gives useful critique
4. Paste the gradient prompt (fill placeholders) → check if it suggests prompt improvements
5. Paste the optimizer prompt (fill placeholders) → check if it produces a valid new prompt

---

## 6. Analysis & Visualization

### `analysis/analyze_results.py`

Create a Python script that reads results from `results/run_*/` and produces:

### 6.1 Accuracy Over Iterations (Line Plot)
- X-axis: iteration number (0 = baseline, 1–5 = optimization steps)
- Y-axis: validation accuracy (%)
- One line per experiment config (core, weak-judge, self-judge if run)
- Save as `analysis/accuracy_over_iterations.png`

### 6.2 Prompt Evolution Table
- Show the prompt text at each iteration alongside its validation accuracy
- Format as a markdown table or notebook-friendly table for the submission notebook
- Highlight key changes between iterations

### 6.3 Per-Question Analysis
- For the test set: which questions went from incorrect → correct after optimization?
- Which went from correct → incorrect (regressions)?
- Show 2–3 concrete examples of improved and regressed questions to illustrate what the optimized prompt helps with

### 6.4 Extension Comparison (Bar Chart) — Only if extensions were run
- X-axis: experiment config (core, weak-judge, self-judge)
- Y-axis: final test accuracy (%) and accuracy improvement from baseline
- Save as `analysis/extension_comparison.png`

### 6.5 Textual Gradient Quality Analysis
- Show 2-3 examples of textual gradients from the logs
- Analyze: are they specific? Do they lead to meaningful prompt changes?
- This qualitative analysis is important for the submission notebook discussion

---

## 7. Submission Responsibilities

Brandon owns these submission sections:

| Section | Content |
|---------|---------|
| **1. Introduction** | Brief context on TextGrad, what we're reproducing, why it matters |
| **2. Paper Summary** | The TextGrad method explained (with your own figures), main claims |
| **4. Experimental Setup** | Dataset (GSM8k subset), models (which GGUF, quantization), compute setup, hyperparameters |
| **5. Results** | Accuracy plots, prompt evolution, per-question analysis |
| **6. Discussion** | Do our results support the paper's claim? What's different with small local models? Limitations. If extensions were run, include as subsections here. |

Roger owns:
| Section | Content |
|---------|---------|
| **3. Implementation** | Architecture, code design decisions, what was built vs borrowed |
| **Code documentation** | Notebook-ready setup/method notes, docstrings, inline comments |

Both collaborate on:
| Section | Content |
|---------|---------|
| **2. Method description** | How we instantiated TextGrad for our setup |
| **7. Discussion** | Implementation decisions not in the paper |

> **Submission format note**: The final hand-in is `.ipynb` only. That means the important setup, method, results, and discussion text must live inside notebook markdown/code cells. A README can still exist for your own workflow, but the grader should not need it.

---

## 8. Deliverables Checklist

- [ ] `prompts/system_prompt_init.txt` — tested manually in LM Studio
- [ ] `prompts/evaluation_prompt.txt` — tested manually in LM Studio
- [ ] `prompts/gradient_prompt.txt` — tested manually in LM Studio
- [ ] `prompts/optimizer_prompt.txt` — tested manually in LM Studio
- [ ] `data/prepare_dataset.py` — runs and produces JSONL files
- [ ] `data/train.jsonl` (20 examples)
- [ ] `data/val.jsonl` (15 examples)
- [ ] `data/test.jsonl` (20 examples)
- [ ] `config/config_core.yaml` — filled with actual model names
- [ ] `config/config_ext_weak_judge.yaml`
- [ ] `config/config_ext_self_judge.yaml`
- [ ] LM Studio running with both models loaded
- [ ] Manual prompt testing completed and documented
- [ ] Core experiment run 1 completed (seed 42)
- [ ] Core experiment run 2 completed (seed 123) — verifies optimization trend is reproducible
- [ ] `analysis/analyze_results.py` — produces all plots
- [ ] Extension experiments run (if time permits)
- [ ] `submission/submission_experiment.ipynb` — final notebook prepared
- [ ] `submission/submission_analysis.ipynb` — plots + discussion prepared
- [ ] `submission/submission_appendix.ipynb` — optional appendix notebook if needed
- [ ] Notebook markdown cells include setup/method/results discussion (do not rely on README alone)
- [ ] Answer extraction regex patterns documented and tested (give test cases to Roger)
