# src/pipeline.py

"""
Main pipeline for TextGrad-style prompt optimization.

This script:
1. loads config
2. loads prompt templates
3. loads dataset splits
4. initializes client / engine / optimizer
5. evaluates the initial prompt
6. runs iterative prompt optimization
7. saves metrics, prompt history, and detailed evaluation results
"""

import argparse
import copy
import json
import random
from pathlib import Path

from engine import TextGradEngine
from evaluation import evaluate_prompt
from llm_client import LLMClient
from optimizer import TGDOptimizer
from utils import (
    append_jsonl,
    create_run_dir,
    load_config,
    load_dataset,
    load_prompt_template,
    sample_batch,
    save_json,
)
from variable import Variable


def _get(config: dict, *keys, default=None):
    """
    Safely read nested config values.

    Example:
        _get(config, "models", "solver_model")
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_prompt_templates(config: dict) -> dict[str, str]:
    """
    Load prompt template files from config.

    Expected config structure can be either:
    nested:
        prompts:
            system_prompt_init: ...
            evaluation_prompt: ...
            gradient_prompt: ...
            optimizer_prompt: ...
    or 
    flat:
        system_prompt_init: ...
        evaluation_prompt: ...
        gradient_prompt: ...
        optimizer_prompt: ...

    """
    prompts_cfg = config.get("prompts", {})

    system_prompt_init_path = prompts_cfg.get(
        "system_prompt_init",
        config.get("system_prompt_init"),
    )
    evaluation_prompt_path = prompts_cfg.get(
        "evaluation_prompt",
        config.get("evaluation_prompt"),
    )
    gradient_prompt_path = prompts_cfg.get(
        "gradient_prompt",
        config.get("gradient_prompt"),
    )
    optimizer_prompt_path = prompts_cfg.get(
        "optimizer_prompt",
        config.get("optimizer_prompt"),
    )

    if not system_prompt_init_path:
        raise ValueError("Missing path for system_prompt_init in config")
    if not evaluation_prompt_path:
        raise ValueError("Missing path for evaluation_prompt in config")
    if not gradient_prompt_path:
        raise ValueError("Missing path for gradient_prompt in config")
    if not optimizer_prompt_path:
        raise ValueError("Missing path for optimizer_prompt in config")

    return {
        "system_prompt_init": load_prompt_template(system_prompt_init_path),
        "evaluation_prompt": load_prompt_template(evaluation_prompt_path),
        "gradient_prompt": load_prompt_template(gradient_prompt_path),
        "optimizer_prompt": load_prompt_template(optimizer_prompt_path),
    }


def _load_datasets(config: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load train / val / test datasets from config.

    Expected config structure can be either:
    nested:
        data:
            train_path: ...
            val_path: ...
            test_path: ...
    or
    flat:
        train_path: ...
        val_path: ...
        test_path: ...
    """
    data_cfg = config.get("data", {})

    train_path = data_cfg.get("train_path", config.get("train_path"))
    val_path = data_cfg.get("val_path", config.get("val_path"))
    test_path = data_cfg.get("test_path", config.get("test_path"))

    if not train_path:
        raise ValueError("Missing train_path in config")
    if not val_path:
        raise ValueError("Missing val_path in config")
    if not test_path:
        raise ValueError("Missing test_path in config")

    train_data = load_dataset(train_path)
    val_data = load_dataset(val_path)
    test_data = load_dataset(test_path)

    return train_data, val_data, test_data


def _save_details(details: list[dict], details_path: Path) -> None:
    """
    Append a list of evaluation result dicts to a JSONL file.
    """
    for item in details:
        append_jsonl(item, str(details_path))


def run_experiment(config_path: str) -> None:
    """
    Run a complete TextGrad prompt optimization experiment.
    """
    # -------------------------
    # 1. load config
    # -------------------------
    config = load_config(config_path)

    # -------------------------
    # 2. create run directory
    # -------------------------
    results_base_dir = _get(config, "output", "results_dir", default="results")
    run_dir = create_run_dir(results_base_dir)

    metrics_path = run_dir / "metrics.json"
    prompts_history_path = run_dir / "prompts_history.json"
    details_path = run_dir / "details.jsonl"

    # -------------------------
    # 3. load templates + data
    # -------------------------
    prompt_templates = _load_prompt_templates(config)
    train_data, val_data, test_data = _load_datasets(config)

    # -------------------------
    # 4a. init LLM client
    # -------------------------
    api_cfg = config.get("api", {})
    log_file = _get(config, "output", "llm_log_file", default=str(run_dir / "llm_calls.jsonl"))

    llm_client = LLMClient(
        api_base_url=api_cfg.get("api_base_url", config.get("api_base_url")),
        api_key=api_cfg.get("api_key", config.get("api_key", "lm-studio")),
        log_file=log_file,
        max_retries=api_cfg.get("max_retries", config.get("max_retries", 3)),
        retry_delay=api_cfg.get("retry_delay", config.get("retry_delay", 5.0)),
    )

    # -------------------------
    # 4b. init engine / optimizer
    # -------------------------
    model_cfg = config.get("models", {})
    solver_model = model_cfg.get("solver_model", config.get("solver_model"))
    judge_model = model_cfg.get("judge_model", config.get("judge_model"))
    self_judge = model_cfg.get("self_judge", config.get("self_judge", False))

    if not solver_model:
        raise ValueError("Missing solver_model in config")
    if not judge_model and not self_judge:
        raise ValueError("Missing judge_model in config")

    solver_temperature = _get(config, "generation", "solver_temperature", default=config.get("solver_temperature", 0.0))
    solver_max_tokens = _get(config, "generation", "solver_max_tokens", default=config.get("solver_max_tokens", 512))
    judge_temperature = _get(config, "generation", "judge_temperature", default=config.get("judge_temperature", 0.7))
    judge_max_tokens = _get(config, "generation", "judge_max_tokens", default=config.get("judge_max_tokens", 512))

    engine = TextGradEngine(
        llm_client=llm_client,
        solver_model=solver_model,
        judge_model=judge_model if judge_model else solver_model,
        solver_temperature=solver_temperature,
        solver_max_tokens=solver_max_tokens,
        judge_temperature=judge_temperature,
        judge_max_tokens=judge_max_tokens,
        prompt_templates={
            "evaluation_prompt": prompt_templates["evaluation_prompt"],
            "gradient_prompt": prompt_templates["gradient_prompt"],
        },
        self_judge=self_judge,
    )

    optimizer_model = solver_model if self_judge else judge_model
    optimizer = TGDOptimizer(
        llm_client=llm_client,
        judge_model=optimizer_model,
        judge_temperature=judge_temperature,
        judge_max_tokens=judge_max_tokens,
        optimizer_template=prompt_templates["optimizer_prompt"],
    )

    # -------------------------
    # 5. create system prompt variable
    # -------------------------
    system_prompt = Variable(
        value=prompt_templates["system_prompt_init"],
        role_description="system prompt for math-solving LLM",
        requires_grad=True,
    )

    best_prompt_text = system_prompt.value
    best_val_accuracy = -1.0

    metrics = []
    prompts_history = []

    # -------------------------
    # 5b. init RNG
    # -------------------------
    random_seed = config.get("random_seed", 42)
    rng = random.Random(random_seed)

    # -------------------------
    # 6. baseline evaluation
    # -------------------------
    print("Running baseline evaluation...")

    val_accuracy, val_details = evaluate_prompt(
        engine=engine,
        system_prompt=system_prompt,
        dataset=val_data,
        iteration=0,
        split_name="val",
    )
    _save_details(val_details, details_path)

    test_accuracy, test_details = evaluate_prompt(
        engine=engine,
        system_prompt=system_prompt,
        dataset=test_data,
        iteration=0,
        split_name="test",
    )
    _save_details(test_details, details_path)

    best_val_accuracy = val_accuracy
    best_prompt_text = system_prompt.value

    metrics.append(
        {
            "iteration": 0,
            "stage": "baseline",
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "accepted": True,
        }
    )

    prompts_history.append(
        {
            "iteration": 0,
            "prompt_before": None,
            "gradient": "",
            "candidate_prompt": system_prompt.value,
            "accepted_prompt": system_prompt.value,
            "val_accuracy": val_accuracy,
            "accepted": True,
        }
    )

    print(f"[Iteration 0] val={val_accuracy:.4f}, test={test_accuracy:.4f}")

    # -------------------------
    # 7. optimization loop
    # -------------------------
    opt_cfg = config.get("optimization", {})
    max_iterations = opt_cfg.get("max_iterations", config.get("max_iterations", 5))
    batch_size = opt_cfg.get("batch_size", config.get("batch_size", 4))
    with_replacement = opt_cfg.get(
        "with_replacement",
        config.get("with_replacement", True),
    )

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Iteration {iteration} ===")

        system_prompt.zero_grad()
        prompt_before = system_prompt.value

        # sample train batch
        batch = sample_batch(
            dataset=train_data,
            batch_size=batch_size,
            with_replacement=with_replacement,
            rng=rng,
        )

        # forward pass
        batch_results = engine.forward_pass(
            system_prompt=system_prompt,
            batch=batch,
            iteration=iteration,
        )

        num_correct = sum(r["is_correct"] for r in batch_results)
        train_batch_accuracy = num_correct / len(batch_results) if batch_results else 0.0

        incorrect_results = [r for r in batch_results if not r["is_correct"]]

        # if batch already perfect, skip optimization
        if not incorrect_results:
            val_accuracy_iter, val_details_iter = evaluate_prompt(
                engine=engine,
                system_prompt=system_prompt,
                dataset=val_data,
                iteration=iteration,
                split_name="val",
            )
            _save_details(val_details_iter, details_path)

            accepted = False  # no update happened
            metrics.append(
                {
                    "iteration": iteration,
                    "stage": "skip_all_correct",
                    "train_batch_accuracy": train_batch_accuracy,
                    "val_accuracy": val_accuracy_iter,
                    "best_val_accuracy": best_val_accuracy,
                    "accepted": accepted,
                }
            )

            prompts_history.append(
                {
                    "iteration": iteration,
                    "prompt_before": prompt_before,
                    "gradient": "",
                    "candidate_prompt": prompt_before,
                    "accepted_prompt": system_prompt.value,
                    "val_accuracy": val_accuracy_iter,
                    "accepted": accepted,
                    "note": "Skipped optimization because all batch answers were correct.",
                }
            )

            print(
                f"All batch examples correct. "
                f"train_batch_acc={train_batch_accuracy:.4f}, val={val_accuracy_iter:.4f}"
            )
            continue

        # compute loss critiques
        loss_critiques = engine.compute_loss(
            incorrect_results=incorrect_results,
            iteration=iteration,
        )

        # compute gradient
        gradient = engine.compute_gradient(
            system_prompt=system_prompt,
            loss_critiques=loss_critiques,
            iteration=iteration,
        )

        if gradient.strip():
            system_prompt.add_gradient(gradient)

        # optimizer proposes a new prompt
        candidate_prompt = optimizer.step(
            variable=system_prompt,
            iteration=iteration,
        )

        # evaluate candidate prompt on validation set
        candidate_variable = Variable(
            value=candidate_prompt,
            role_description=system_prompt.role_description,
            requires_grad=False,
        )

        val_accuracy_candidate, val_details_candidate = evaluate_prompt(
            engine=engine,
            system_prompt=candidate_variable,
            dataset=val_data,
            iteration=iteration,
            split_name="val",
        )
        _save_details(val_details_candidate, details_path)

        # accept only if validation improves
        accepted = val_accuracy_candidate > best_val_accuracy

        if accepted:
            system_prompt.update(candidate_prompt)
            best_prompt_text = candidate_prompt
            best_val_accuracy = val_accuracy_candidate

        metrics.append(
            {
                "iteration": iteration,
                "stage": "optimization",
                "train_batch_accuracy": train_batch_accuracy,
                "num_incorrect_in_batch": len(incorrect_results),
                "num_critiques": len(loss_critiques),
                "val_accuracy": val_accuracy_candidate,
                "best_val_accuracy": best_val_accuracy,
                "accepted": accepted,
            }
        )

        prompts_history.append(
            {
                "iteration": iteration,
                "prompt_before": prompt_before,
                "gradient": gradient,
                "candidate_prompt": candidate_prompt,
                "accepted_prompt": system_prompt.value,
                "val_accuracy": val_accuracy_candidate,
                "accepted": accepted,
            }
        )

        print(
            f"train_batch_acc={train_batch_accuracy:.4f}, "
            f"candidate_val={val_accuracy_candidate:.4f}, "
            f"best_val={best_val_accuracy:.4f}, "
            f"accepted={accepted}"
        )

        system_prompt.zero_grad()

    # -------------------------
    # 8. final test evaluation using best prompt
    # -------------------------
    print("\nRunning final test evaluation with best prompt...")

    best_variable = Variable(
        value=best_prompt_text,
        role_description="best system prompt",
        requires_grad=False,
    )

    final_test_accuracy, final_test_details = evaluate_prompt(
        engine=engine,
        system_prompt=best_variable,
        dataset=test_data,
        iteration=max_iterations + 1,
        split_name="test",
    )
    _save_details(final_test_details, details_path)

    metrics.append(
        {
            "iteration": max_iterations + 1,
            "stage": "final_test",
            "val_accuracy": best_val_accuracy,
            "test_accuracy": final_test_accuracy,
            "accepted": True,
        }
    )

    # -------------------------
    # 9. save outputs
    # -------------------------
    save_json(metrics, str(metrics_path))
    save_json(prompts_history, str(prompts_history_path))

    summary = {
        "config_path": config_path,
        "run_dir": str(run_dir),
        "best_val_accuracy": best_val_accuracy,
        "final_test_accuracy": final_test_accuracy,
        "best_prompt": best_prompt_text,
    }
    save_json(summary, str(run_dir / "summary.json"))

    print("\nDone.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextGrad Prompt Optimization")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    run_experiment(args.config)