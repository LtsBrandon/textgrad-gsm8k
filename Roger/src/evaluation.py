# src/evaluation.py

import re


def extract_answer(response: str) -> float | None:
    """
    Extract the numerical answer from a solver response.

    Tries:
    1. "Answer: $VALUE"
    2. "Answer: VALUE"
    3. "#### VALUE"
    4. last number in the response
    """
    # try "Answer: $42" or "Answer: 42"
    match = re.search(r"Answer:\s*\$?\s*(-?\d+(?:\.\d+)?)", response)
    if match:
        return float(match.group(1))

    # try GSM8k style "#### 42"
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", response)
    if match:
        return float(match.group(1))

    # fallback: take the last number that appears in the response
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    if numbers:
        return float(numbers[-1])

    return None


def exact_match(predicted: float | None, ground_truth: float) -> bool:
    """
    Check whether predicted answer matches ground truth.
    """
    if predicted is None:
        return False

    return abs(predicted - ground_truth) < 1e-6


def evaluate_prompt(
    engine,
    system_prompt,
    dataset: list[dict],
    iteration: int,
    split_name: str = "val"
) -> tuple[float, list[dict]]:
    """
    Evaluate a system prompt on a dataset split.

    Args:
        engine: TextGradEngine instance
        system_prompt: Variable containing the current prompt
        dataset: list of dataset examples
        iteration: current iteration number
        split_name: dataset split name ("val" or "test")

    Returns:
        (accuracy, detailed_results)
    """
    # run the solver on the whole dataset
    results = engine.forward_pass(
        system_prompt=system_prompt,
        batch=dataset,
        iteration=iteration
    )

    # count correct predictions
    num_correct = sum(result["is_correct"] for result in results)
    accuracy = num_correct / len(dataset) if dataset else 0.0

    detailed_results = []

    # expected pipeline output
    for item, result in zip(dataset, results):
        detailed_results.append(
            {
                "iteration": iteration,
                "split": split_name,
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "model_response": result["model_response"],
                "extracted_answer": result["extracted_answer"],
                "is_correct": result["is_correct"],
            }
        )

    return accuracy, detailed_results