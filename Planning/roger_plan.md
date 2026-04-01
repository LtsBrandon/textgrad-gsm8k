# Roger's Deliverables — TextGrad GSM8k Reproduction

> **Your role**: Core pipeline implementer. You build the TextGrad engine from scratch.
> Brandon provides the prompts, dataset, config files, and LM Studio runtime.
> Your code loads his files, runs the optimization loop, and outputs structured logs/results.

> **Reference material**: The [`textgrad`](https://github.com/zou-group/textgrad) pip package is the paper's official library. You may reference its source code to understand the algorithm, and borrow small utility patterns, but **the core optimization loop must be your own implementation**. This is critical for the "Technical Contribution" rubric grade.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      pipeline.py (main)                      │
│  Loads config → Loads data → Runs optimization loop          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  FOR each iteration:                                 │    │
│  │    1. Sample batch from train set                    │    │
│  │    2. engine.forward_pass(batch, current_prompt)     │───►│ llm_client.py
│  │    3. engine.compute_loss(incorrect_answers)         │───►│ (calls Solver LLM)
│  │    4. engine.compute_gradient(prompt, losses)        │───►│
│  │    5. optimizer.step(prompt, gradient) → new_prompt  │───►│ (calls Judge LLM)
│  │    6. evaluation.validate(new_prompt, val_set)       │───►│
│  │    7. If improved: update prompt                     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Uses: variable.py, utils.py                                 │
│  Outputs: logs/*.jsonl, results/run_*/                       │
└──────────────────────────────────────────────────────────────┘
```

**Data flow**:
```
Brandon's files                  Roger's code                    Output
─────────────                    ───────────                     ──────
config/*.yaml    ──────►  utils.load_config()
prompts/*.txt    ──────►  utils.load_prompt_template()
data/*.jsonl     ──────►  utils.load_dataset()
                                    │
                                    ▼
                            pipeline.run() ──────────►  logs/*.jsonl
                                    │                   results/run_*/
                                    ▼
                          LM Studio API (Brandon's PC)
```

---

## 2. Modules to Implement

### 2.1 `src/variable.py` — The Variable Class

This is the TextGrad analog of a PyTorch tensor. It holds a text value and accumulates textual gradients.

```python
class Variable:
    """A text variable that can accumulate textual gradients.

    Analogous to a PyTorch tensor with requires_grad=True.
    The 'value' is a string (e.g., a system prompt).
    The 'gradients' are a list of natural-language feedback strings.
    """

    def __init__(
        self,
        value: str,
        role_description: str,
        requires_grad: bool = False
    ):
        """
        Args:
            value: The text content of this variable
            role_description: What this variable represents
                (e.g., "system prompt for a math-solving LLM")
            requires_grad: Whether this variable should collect gradients
        """
        self.value = value
        self.role_description = role_description
        self.requires_grad = requires_grad
        self.gradients: list[str] = []  # accumulated feedback

    def add_gradient(self, feedback: str) -> None:
        """Append a textual gradient (feedback) to this variable."""
        self.gradients.append(feedback)

    def get_aggregated_gradient(self) -> str:
        """Combine all accumulated gradients into one string."""
        if not self.gradients:
            return ""
        aggregated = "\n\n---\n\n".join(
            [f"Feedback {i+1}:\n{g}" for i, g in enumerate(self.gradients)]
        )
        return aggregated

    def zero_grad(self) -> None:
        """Clear accumulated gradients (call at start of each iteration)."""
        self.gradients = []

    def update(self, new_value: str) -> None:
        """Update the variable's value (after optimizer step)."""
        self.value = new_value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Variable(role='{self.role_description}', value='{self.value[:50]}...')"
```

**Build from scratch**: Yes. This is simple but core to the design.

---

### 2.2 `src/llm_client.py` — LLM API Wrapper

Thin wrapper around the `openai` Python SDK, pointed at LM Studio's local server.

**Why the `openai` library instead of raw HTTP?**
LM Studio exposes a local REST API at `http://127.0.0.1:1234` that is fully compatible with
the OpenAI API format. Rather than writing raw `requests.post(...)` calls — which require
manually constructing JSON bodies, parsing responses, handling errors, and extracting token
counts — we use the standard `openai` Python SDK and just redirect it to `127.0.0.1:1234/v1`
instead of OpenAI's cloud. Benefits:
- Clean, readable call syntax (`client.chat.completions.create(...)`)
- Automatic response parsing (`response.choices[0].message.content`)
- Token usage automatically available (`response.usage.prompt_tokens`)
- Built-in error types for structured exception handling
- Zero code changes needed if Brandon later wants to test against real OpenAI API

```python
from openai import OpenAI
import time
import json

class LLMClient:
    """Wrapper for OpenAI-compatible API calls.

    Uses the `openai` Python SDK pointed at LM Studio's local API server
    (http://127.0.0.1:1234/v1) instead of OpenAI's cloud. LM Studio mirrors
    the OpenAI REST API format exactly, so the SDK works without modification.

    Handles:
    - Chat completion calls to solver and judge models
    - Automatic retry on failure
    - Logging every call to a JSONL file
    """

    def __init__(
        self,
        api_base_url: str,        # Brandon's LM Studio: "http://127.0.0.1:1234/v1"
                                  # The /v1 suffix is required by the openai SDK
        api_key: str = "lm-studio",  # LM Studio ignores the key but the SDK requires one
        log_file: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        """Initialize the client."""
        ...

    def chat(
        self,
        model: str,               # Model identifier in LM Studio
        messages: list[dict],      # [{"role": "system"|"user", "content": "..."}]
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            model: Model name as registered in LM Studio
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            The assistant's response text (str)

        Raises:
            RuntimeError: After max_retries failures
        """
        ...

    def _log_call(
        self,
        model: str,
        messages: list[dict],
        response: str,
        tokens_prompt: int,
        tokens_completion: int,
        latency_ms: float,
        iteration: int | None = None,
        step: str | None = None
    ) -> None:
        """Write one JSON line to the log file."""
        ...
```

**Logging format** (matches overall_plan.md contract):
```json
{
  "timestamp": "2026-03-20T14:30:00",
  "iteration": 2,
  "step": "forward",
  "role": "solver",
  "model": "qwen2.5-3b-q4",
  "messages": [...],
  "response": "...",
  "tokens_prompt": 145,
  "tokens_completion": 89,
  "latency_ms": 2340
}
```

**Build from scratch**: Yes. It's simple but you need the logging.
**May reference**: The `textgrad` library's engine classes for how they handle the openai client.

---

### 2.3 `src/engine.py` — The TextGrad Engine

This is the core module — it implements the forward pass, loss computation, and backward pass (gradient generation).

```python
class TextGradEngine:
    """Implements forward pass, loss evaluation, and gradient computation.

    This is the heart of TextGrad. It coordinates:
    1. Forward pass: solver LLM answers questions using current prompt
    2. Loss: judge LLM evaluates incorrect answers
    3. Gradient: judge LLM critiques the system prompt based on losses
    """

    def __init__(
        self,
        llm_client: LLMClient,
        solver_model: str,
        judge_model: str,
        solver_temperature: float,
        solver_max_tokens: int,
        judge_temperature: float,
        judge_max_tokens: int,
        prompt_templates: dict[str, str],  # loaded template strings
        self_judge: bool = False
    ):
        """
        Args:
            llm_client: The LLM API client
            solver_model: Model name for the solver (forward pass)
            judge_model: Model name for the judge (loss + gradient)
            solver_temperature: Temperature for solver calls
            solver_max_tokens: Max tokens for solver responses
            judge_temperature: Temperature for judge calls
            judge_max_tokens: Max tokens for judge responses
            prompt_templates: Dict with keys:
                'evaluation_prompt' -> template string with {question}, {model_answer}, {ground_truth}
                'gradient_prompt' -> template string with {system_prompt}, {aggregated_losses}
            self_judge: If True, solver_model is used for ALL calls
                (judge_model parameter is ignored; solver_model acts as
                both solver and judge for forward, loss, gradient, and optimizer)
        """
        # Implementation note: when self_judge is True, any method that
        # would normally use self.judge_model should use self.solver_model instead.
        # Simplest approach: in __init__, do:
        #   if self_judge:
        #       self.judge_model = self.solver_model
        ...

    def forward_pass(
        self,
        system_prompt: Variable,
        batch: list[dict],         # list of dataset items
        iteration: int
    ) -> list[dict]:
        """Run the solver on a batch of questions.

        For each question in the batch:
        1. Send [system_prompt, question] to the solver LLM
        2. Extract the numerical answer from the response
        3. Compare with ground truth (exact match)

        Args:
            system_prompt: The current system prompt Variable
            batch: List of dataset items (each has 'question', 'ground_truth')
            iteration: Current iteration number (for logging)

        Returns:
            List of result dicts:
            [
                {
                    "question": "...",
                    "ground_truth": 42,
                    "model_response": "Let me solve step by step... Answer: $42",
                    "extracted_answer": 42,
                    "is_correct": True
                },
                ...
            ]
        """
        ...

    def compute_loss(
        self,
        incorrect_results: list[dict],
        iteration: int
    ) -> list[str]:
        """Have the judge evaluate each incorrect answer.

        For each incorrect result:
        1. Fill the evaluation_prompt template with question, answer, ground truth
        2. Send to judge LLM
        3. Collect the critique (natural language loss)

        Args:
            incorrect_results: Results from forward_pass where is_correct=False
            iteration: Current iteration number (for logging)

        Returns:
            List of critique strings (one per incorrect answer)
        """
        ...

    def compute_gradient(
        self,
        system_prompt: Variable,
        loss_critiques: list[str],
        iteration: int
    ) -> str:
        """Generate the textual gradient for the system prompt.

        Given the aggregated critiques from the batch, ask the judge:
        "How should the system prompt be changed to avoid these errors?"

        This is the BACKWARD PASS — it produces feedback targeted at
        the prompt (the variable being optimized), not at individual answers.

        Args:
            system_prompt: The current system prompt Variable
            loss_critiques: List of critique strings from compute_loss
            iteration: Current iteration number (for logging)

        Returns:
            The textual gradient string (feedback on how to change the prompt)
        """
        ...
```

**Build from scratch**: Yes. This is the primary technical contribution.

---

### 2.4 `src/optimizer.py` — Textual Gradient Descent (TGD)

```python
class TGDOptimizer:
    """Textual Gradient Descent — rewrites a Variable using its gradients.

    Analogous to SGD in PyTorch: takes the current value and gradient,
    produces an updated value.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        judge_model: str,
        judge_temperature: float,
        judge_max_tokens: int,
        optimizer_template: str   # loaded template with {current_prompt}, {gradient_feedback}
    ):
        """
        Args:
            llm_client: The LLM API client
            judge_model: Model name for the judge (does the rewriting)
            judge_temperature: Temperature for the rewrite call
            judge_max_tokens: Max tokens for the new prompt
            optimizer_template: Template string for the optimizer prompt
        """
        ...

    def step(
        self,
        variable: Variable,
        iteration: int
    ) -> str:
        """Perform one TGD update step.

        1. Get the aggregated gradient from the variable
        2. Fill the optimizer template with current value + gradient
        3. Send to judge LLM
        4. Return the new value (does NOT update the variable — pipeline decides)

        Args:
            variable: The Variable to optimize (must have gradients)
            iteration: Current iteration number (for logging)

        Returns:
            The proposed new value (string) for the variable
        """
        ...
```

**Build from scratch**: Yes.

---

### 2.5 `src/evaluation.py` — Answer Extraction & Scoring

```python
import re

def extract_answer(response: str) -> float | None:
    """Extract the numerical answer from a solver's response.

    Tries multiple patterns in order:
    1. 'Answer: $VALUE' or 'Answer: VALUE' (our instructed format)
    2. '#### VALUE' (GSM8k native format, in case model mimics training data)
    3. Last number in the response (fallback)

    Args:
        response: The full text response from the solver LLM

    Returns:
        The extracted number as a float, or None if no number found
    """
    ...


def exact_match(predicted: float | None, ground_truth: float) -> bool:
    """Check if the predicted answer matches ground truth.

    Handles floating point comparison with small tolerance.

    Args:
        predicted: Extracted answer (or None if extraction failed)
        ground_truth: The correct answer from the dataset

    Returns:
        True if answers match, False otherwise
    """
    ...


def evaluate_prompt(
    engine: 'TextGradEngine',
    system_prompt: 'Variable',
    dataset: list[dict],
    iteration: int,
    split_name: str = "val"
) -> tuple[float, list[dict]]:
    """Evaluate a system prompt on a dataset split.

    Runs the solver on every question and computes accuracy.

    Args:
        engine: The TextGrad engine (for making solver calls)
        system_prompt: The prompt to evaluate
        dataset: List of dataset items
        iteration: Current iteration (for logging)
        split_name: Name of the split (for logging: "val", "test")

    Returns:
        Tuple of (accuracy: float between 0-1, detailed_results: list[dict])
        where each dict in detailed_results has the fields defined in the
        details.jsonl contract (see overall_plan.md section 4.5):
        iteration, split, id, question, ground_truth, model_response,
        extracted_answer, is_correct
    """
    ...
```

**Build from scratch**: Yes, but you may reference the `textgrad` library's evaluation patterns and standard GSM8k evaluation scripts for the regex.

---

### 2.6 `src/pipeline.py` — Main Orchestrator

```python
"""Main entry point. Ties everything together."""

import argparse

def run_experiment(config_path: str) -> None:
    """Run a complete TextGrad prompt optimization experiment.

    Steps:
    1. Load config (YAML)
    2. Load prompt templates from prompts/ directory
    3. Load dataset splits (train, val, test JSONL)
    4. Initialize LLMClient, TextGradEngine, TGDOptimizer
    5. Create the system prompt Variable (from system_prompt_init.txt)
    5b. Initialize the batch-sampling RNG using config's `random_seed`
    6. Run baseline evaluation on val and test sets (iteration 0)
    7. FOR each iteration 1..max_iterations:
        a. Sample a mini-batch (batch_size) from train set
           (with replacement if configured)
        b. Forward pass: solver answers the batch
        c. If all correct: skip optimization this iteration, continue
        d. Compute loss: judge critiques incorrect answers
        e. Compute gradient: judge suggests prompt improvements
        f. Add gradient to the system prompt Variable
        g. Optimizer step: judge rewrites the prompt
        h. Validate: evaluate new prompt on val set
        i. If val accuracy improved: update prompt, record as best
        j. Zero gradients on the variable
        k. Log all results
    8. Final evaluation on test set with the best prompt
    9. Save all results to results/run_YYYYMMDD_HHMMSS/:
       - metrics.json: accuracy per iteration (see overall_plan.md section 4.5)
       - prompts_history.json: prompt text + gradient at each iteration
       - details.jsonl: per-question results for every evaluation run
         (one JSON line per question, with fields: iteration, split, id,
          question, ground_truth, model_response, extracted_answer, is_correct)
         This file is critical — Brandon's per-question analysis depends on it.

    Args:
        config_path: Path to the experiment config YAML file
    """
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextGrad Prompt Optimization")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    run_experiment(args.config)
```

**Build from scratch**: Yes.

---

### 2.7 `src/utils.py` — Utilities

```python
import yaml
import json
from pathlib import Path
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as dict."""
    ...


def load_dataset(file_path: str) -> list[dict]:
    """Load a JSONL dataset file.

    Each line is a JSON object with keys: id, question, answer_text, ground_truth

    Returns:
        List of dicts
    """
    ...


def load_prompt_template(file_path: str) -> str:
    """Load a prompt template file as a string.

    Returns:
        The template string with {placeholder} variables intact
    """
    ...


def fill_template(template: str, **kwargs) -> str:
    """Fill a prompt template with values.

    Example:
        fill_template("Hello {name}", name="World") -> "Hello World"
    """
    ...


def create_run_dir(base_dir: str) -> Path:
    """Create a timestamped run directory.

    Returns:
        Path to the created directory (e.g., results/run_20260320_143000/)
    """
    ...


def sample_batch(
    dataset: list[dict],
    batch_size: int,
    with_replacement: bool = True,
    rng=None
) -> list[dict]:
    """Sample a mini-batch from the dataset.

    Args:
        dataset: Full training dataset
        batch_size: Number of examples to sample
        with_replacement: Whether to sample with replacement
        rng: Random number generator (for reproducibility)

    Returns:
        List of sampled dataset items
    """
    ...


def save_json(data: dict | list, file_path: str) -> None:
    """Save data as formatted JSON."""
    ...


def append_jsonl(data: dict, file_path: str) -> None:
    """Append one JSON line to a JSONL file."""
    ...
```

**Build from scratch**: Yes, these are all straightforward utilities.

---

## 3. What to Build vs. Borrow

| Component | Build from Scratch | May Reference textgrad Library |
|-----------|-------------------|-------------------------------|
| `Variable` class | **Yes** — core abstraction | Check their `Variable` class for design patterns |
| `LLMClient` | **Yes** — simple wrapper | Check their engine classes for openai SDK usage |
| `TextGradEngine` (forward, loss, gradient) | **Yes** — this is the main contribution | Check `textgrad/autograd/` for how they implement backward |
| `TGDOptimizer` | **Yes** — core abstraction | Check `textgrad/optimizer/` for their TGD implementation |
| `evaluation.py` (answer extraction) | **Yes** | Borrow regex patterns from standard GSM8k eval scripts |
| `pipeline.py` (orchestration) | **Yes** | Check their example notebooks for the optimization loop structure |
| `utils.py` (config, I/O) | **Yes** | Standard Python utilities, nothing to borrow |
| Prompt templates | **No** — Brandon provides these | N/A |
| Dataset files | **No** — Brandon provides these | N/A |
| Config files | **No** — Brandon provides these | N/A |

**What counts as "borrowing"**: Looking at their code to understand the algorithm, then writing your own implementation. You are NOT copying their files. You ARE writing code that follows the same logical flow.

---

## 4. Testing Strategy

Since you're developing on Google Colab Free (no GPU), you need a mock-based testing approach.

### 4.1 Unit Tests

**`tests/test_variable.py`**:
```python
def test_variable_creation():
    v = Variable("test prompt", role_description="system prompt", requires_grad=True)
    assert v.value == "test prompt"
    assert v.requires_grad is True
    assert v.gradients == []

def test_gradient_accumulation():
    v = Variable("prompt", role_description="test", requires_grad=True)
    v.add_gradient("feedback 1")
    v.add_gradient("feedback 2")
    assert len(v.gradients) == 2
    agg = v.get_aggregated_gradient()
    assert "feedback 1" in agg
    assert "feedback 2" in agg

def test_zero_grad():
    v = Variable("prompt", role_description="test", requires_grad=True)
    v.add_gradient("feedback")
    v.zero_grad()
    assert v.gradients == []
```

**`tests/test_evaluation.py`**:
```python
def test_extract_answer_standard():
    assert extract_answer("The answer is Answer: $42") == 42.0

def test_extract_answer_no_dollar():
    assert extract_answer("Answer: 18") == 18.0

def test_extract_answer_gsm8k_format():
    assert extract_answer("Some reasoning\n#### 55") == 55.0

def test_extract_answer_fallback_last_number():
    assert extract_answer("After calculation, we get 73.") == 73.0

def test_extract_answer_no_number():
    assert extract_answer("I don't know") is None

def test_exact_match_true():
    assert exact_match(42.0, 42) is True

def test_exact_match_false():
    assert exact_match(41.0, 42) is False

def test_exact_match_none():
    assert exact_match(None, 42) is False
```

### 4.2 Mock-Based Integration Tests

**`tests/test_engine_mock.py`**:

Create a `MockLLMClient` that returns canned responses, then test the full engine flow:

```python
class MockLLMClient:
    """Returns predefined responses for testing without a real LLM."""

    def __init__(self, responses: dict[str, str]):
        """
        Args:
            responses: Mapping from keywords in the prompt to response text.
                       When a prompt contains a keyword, return the corresponding response.
        """
        self.responses = responses
        self.call_log = []

    def chat(self, model, messages, temperature=0, max_tokens=512):
        prompt_text = " ".join(m["content"] for m in messages)
        self.call_log.append({"model": model, "messages": messages})
        for keyword, response in self.responses.items():
            if keyword in prompt_text:
                return response
        return "Default mock response. Answer: $0"
```

**`tests/mock_responses.json`**:
```json
{
  "solver_correct": "Let me solve step by step. 16 - 3 - 4 = 9. 9 * 2 = 18.\nAnswer: $18",
  "solver_incorrect": "16 - 3 = 13. 13 - 4 = 8. I think it's 8.\nAnswer: $8",
  "evaluation_critique": "The student made an error in the final step. They computed 16-3-4=8 but forgot to multiply by the price per egg ($2). The correct calculation is (16-3-4)*2 = 18.",
  "gradient_feedback": "The system prompt should instruct the model to carefully identify ALL operations needed, not just subtraction. It should tell the model to re-read the question to ensure no steps are missed.",
  "optimizer_new_prompt": "You will answer a mathematical reasoning question. Read the question carefully and identify every operation needed. Think step by step, and verify that you have addressed every part of the question. The last line of your response should be: 'Answer: $VALUE' where VALUE is a numerical value."
}
```

**Test the full pipeline with mocks**:
```python
def test_full_iteration_with_mock():
    """Test one complete optimization iteration using mock LLM."""
    client = MockLLMClient(MOCK_RESPONSES)
    engine = TextGradEngine(client, ...)
    optimizer = TGDOptimizer(client, ...)

    prompt = Variable("Think step by step...", requires_grad=True)
    batch = [{"question": "...", "ground_truth": 18}]

    # Forward pass
    results = engine.forward_pass(prompt, batch, iteration=1)
    assert len(results) == 1

    # Loss
    incorrect = [r for r in results if not r["is_correct"]]
    critiques = engine.compute_loss(incorrect, iteration=1)

    # Gradient
    gradient = engine.compute_gradient(prompt, critiques, iteration=1)
    prompt.add_gradient(gradient)

    # Optimizer step
    new_prompt = optimizer.step(prompt, iteration=1)
    assert isinstance(new_prompt, str)
    assert len(new_prompt) > 0
```

### 4.3 Testing on Google Colab

1. Clone the shared git repo in Colab
2. `pip install openai pyyaml`
3. Run unit tests: `!python -m pytest tests/ -v`
4. All tests use mocks – no GPU or LLM server needed
5. Once tests pass, push commits to the shared git remote (private GitHub/private remote/local shared repo all fine) so Brandon can run with real LLMs

---

## 5. Development Workflow

```
Roger's workflow:
1. Clone the shared git repo
2. Create a branch: git checkout -b feature/core-engine
3. Implement modules one at a time in this order:
   a. variable.py + test_variable.py
   b. llm_client.py (with logging)
   c. evaluation.py + test_evaluation.py
   d. engine.py + test_engine_mock.py
   e. optimizer.py
   f. utils.py
   g. pipeline.py
4. Run all tests in Colab: !python -m pytest tests/ -v
5. Push commits/branch to the shared git remote
6. Brandon pulls/merges, plugs in his prompts/data/config, runs on LM Studio
7. If issues: iterate on the branch, push fixes
```

**Workflow note**: Even though the final course submission is notebook-only, keep using `git` during development. It is still the best way to manage parallel work and merge changes safely.

### Implementation Order (recommended)

| Order | Module | Dependencies | Estimated Effort |
|-------|--------|-------------|-----------------|
| 1 | `variable.py` | None | Small — ~30 min |
| 2 | `utils.py` | None | Small — ~45 min |
| 3 | `evaluation.py` | None | Medium — ~1 hr (regex edge cases) |
| 4 | `llm_client.py` | `utils.py` | Medium — ~1 hr (retry logic, logging) |
| 5 | `engine.py` | `llm_client`, `variable`, `evaluation`, `utils` | Large — ~2-3 hrs (core logic) |
| 6 | `optimizer.py` | `llm_client`, `variable`, `utils` | Medium — ~1 hr |
| 7 | `pipeline.py` | Everything above | Large — ~2-3 hrs (orchestration, result saving) |
| 8 | Tests | All modules | Ongoing — write alongside each module |

**Total estimated effort**: ~10-14 hours of focused coding

---

## 6. Integration Checklist

When your code is ready for Brandon to use, verify:

- [ ] `pip install openai pyyaml datasets matplotlib` — all dependencies listed in `requirements.txt`
- [ ] `python -m pytest tests/ -v` — all tests pass
- [ ] `python src/pipeline.py --config config/config_core.yaml` — runs without crashing (with mock or real LLM)
- [ ] Config loading works — reads all fields from YAML correctly
- [ ] Prompt template loading works — reads `.txt` files, fills `{placeholders}` correctly
- [ ] Dataset loading works — reads JSONL, parses all fields
- [ ] LLM client works — connects to `http://127.0.0.1:1234/v1` (Brandon's LM Studio, he tests this)
- [ ] Logging works — every LLM call produces a JSONL line with the correct format
- [ ] Results are saved — `metrics.json`, `prompts_history.json`, and `details.jsonl` in the run directory
- [ ] Answer extraction handles edge cases — `$42`, `42`, `#### 42`, no number found
- [ ] Self-judge mode works — when `self_judge: true`, solver model is used for judge calls too

---

## 7. Submission Responsibilities

Roger owns these submission sections:

| Section | Content |
|---------|---------|
| **3. Implementation Details** | Architecture diagram, module descriptions, design decisions |
| **Code documentation** | Notebook-ready setup instructions/method notes, docstrings in every module |

You should also write notebook-ready text about:
- What you built from scratch vs. what you referenced from the textgrad library
- Key implementation decisions (e.g., how you handled the gradient aggregation, retry logic)
- Challenges encountered during implementation

> **Submission format note**: Since the final hand-in is `.ipynb` only, any setup/usage explanation that the grader needs should be easy for Brandon to copy into notebook markdown cells. A README is still useful internally, but it is no longer the primary submission surface.

---

## 8. Key Gotchas and Tips

1. **LM Studio model names**: The model name in the config must EXACTLY match what LM Studio reports. Brandon will provide these after loading models.

2. **Response parsing**: Small models are messy. The solver might not follow the "Answer: $VALUE" format perfectly. Build robust fallback extraction in `evaluation.py`.

3. **Empty batches**: If all 3 questions in a batch are answered correctly, there are no losses to compute gradients from. Handle this gracefully — skip the optimization step for that iteration.

4. **Prompt growing too long**: The optimizer might keep adding text to the prompt. The optimizer template (Brandon's) tells the judge to keep it under 200 words, but verify in your code that the prompt doesn't explode. Consider adding a max length check.

5. **API timeouts**: Local models can be slow. Set generous timeouts (60+ seconds per call) in the LLM client. LM Studio sometimes takes a while on first calls.

6. **Temperature 0 for solver**: Important for reproducibility. The solver should be deterministic. The judge should have some temperature (0.7) for creative feedback.

7. **Logging is critical**: Brandon needs the logs for analysis and submission notebook figures. Don't skip the logging implementation — it's as important as the algorithm itself.
