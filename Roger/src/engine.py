# src/engine.py

from evaluation import extract_answer, exact_match
from utils import fill_template


class TextGradEngine:
    """
    Simple TextGrad engine.

    It does the following things:
    1. forward_pass: solver answers questions
    2. compute_loss: judge critiques incorrect answers
    3. compute_gradient: judge suggests how to improve the system prompt
    """

    def __init__(
        self,
        llm_client,
        solver_model: str,
        judge_model: str,
        solver_temperature: float,
        solver_max_tokens: int,
        judge_temperature: float,
        judge_max_tokens: int,
        prompt_templates: dict[str, str],
        self_judge: bool = False,
    ):
        """
        Initialize the engine.

        Args:
            llm_client: LLMClient instance
            solver_model: model used for solving questions
            judge_model: model used for judging / feedback
            solver_temperature: temperature for solver calls
            solver_max_tokens: max output tokens for solver
            judge_temperature: temperature for judge calls
            judge_max_tokens: max output tokens for judge
            prompt_templates: dictionary of loaded prompt templates
            self_judge: if True, use solver model for judge calls too
        """
        self.llm_client = llm_client
        self.solver_model = solver_model
        self.judge_model = solver_model if self_judge else judge_model

        self.solver_temperature = solver_temperature
        self.solver_max_tokens = solver_max_tokens
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens

        self.prompt_templates = prompt_templates

    def forward_pass(
        self,
        system_prompt,
        batch: list[dict],
        iteration: int
    ) -> list[dict]:
        """
        Run the solver on a batch of questions.

        For each example:
        - send [system prompt, user question] to solver
        - extract numerical answer
        - compare against ground truth

        Returns:
            list of result dicts
        """
        results = []

        for item in batch:
            question = item["question"]
            ground_truth = item["ground_truth"]

            messages = [
                {"role": "system", "content": system_prompt.value},
                {"role": "user", "content": question},
            ]

            model_response = self.llm_client.chat(
                model=self.solver_model,
                messages=messages,
                temperature=self.solver_temperature,
                max_tokens=self.solver_max_tokens,
                iteration=iteration,
                step="forward",
                role="solver",
            )

            extracted_answer = extract_answer(model_response)
            is_correct = exact_match(extracted_answer, ground_truth)

            results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_response": model_response,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                }
            )

        return results

    def compute_loss(
        self,
        incorrect_results: list[dict],
        iteration: int
    ) -> list[str]:
        """
        Ask the judge to critique each incorrect answer.

        Returns:
            list of critique strings
        """
        if not incorrect_results:
            return []

        evaluation_template = self.prompt_templates["evaluation_prompt"]
        critiques = []

        for result in incorrect_results:
            eval_prompt = fill_template(
                evaluation_template,
                question=result["question"],
                model_answer=result["model_response"],
                ground_truth=result["ground_truth"],
            )

            messages = [
                {"role": "user", "content": eval_prompt}
            ]

            critique = self.llm_client.chat(
                model=self.judge_model,
                messages=messages,
                temperature=self.judge_temperature,
                max_tokens=self.judge_max_tokens,
                iteration=iteration,
                step="loss",
                role="judge",
            )

            critiques.append(critique)

        return critiques

    def compute_gradient(
        self,
        system_prompt,
        loss_critiques: list[str],
        iteration: int
    ) -> str:
        """
        Ask the judge how the system prompt should be improved.

        Returns:
            one textual gradient string
        """
        if not loss_critiques:
            return ""

        gradient_template = self.prompt_templates["gradient_prompt"]

        aggregated_losses = "\n\n---\n\n".join(
            f"Critique {i+1}:\n{critique}"
            for i, critique in enumerate(loss_critiques)
        )

        gradient_prompt = fill_template(
            gradient_template,
            system_prompt=system_prompt.value,
            aggregated_losses=aggregated_losses,
        )

        messages = [
            {"role": "user", "content": gradient_prompt}
        ]

        gradient = self.llm_client.chat(
            model=self.judge_model,
            messages=messages,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_tokens,
            iteration=iteration,
            step="gradient",
            role="judge",
        )

        return gradient