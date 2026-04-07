# src/optimizer.py

from utils import fill_template


class TGDOptimizer:
    """
    Textual Gradient Descent optimizer.

    This optimizer does one thing:
    - take the current prompt
    - take the aggregated textual gradient (feedback)
    - ask the judge model to rewrite the prompt

    It returns the proposed new prompt text.
    The caller decides whether to actually update the Variable.
    """

    def __init__(
        self,
        llm_client,
        judge_model: str,
        judge_temperature: float,
        judge_max_tokens: int,
        optimizer_template: str,
    ):
        """
        Initialize the optimizer.

        Args:
            llm_client: LLMClient instance
            judge_model: model used to rewrite the prompt
            judge_temperature: temperature for optimizer rewrite calls
            judge_max_tokens: max tokens for rewritten prompt
            optimizer_template: template string with placeholders:
                {current_prompt}
                {gradient_feedback}
        """
        self.llm_client = llm_client
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens
        self.optimizer_template = optimizer_template

    def step(self, variable, iteration: int) -> str:
        """
        Perform one optimization step.

        Process:
        1. collect all gradients stored in the Variable
        2. fill the optimizer template
        3. send it to the judge model
        4. return the rewritten prompt text

        Args:
            variable: Variable object containing current prompt + gradients
            iteration: current optimization iteration

        Returns:
            Proposed new prompt string
        """
        gradient_feedback = variable.get_aggregated_gradient()

        # if there is no feedback, keep the current value unchanged
        if not gradient_feedback.strip():
            return variable.value

        optimizer_prompt = fill_template(
            self.optimizer_template,
            current_prompt=variable.value,
            gradient_feedback=gradient_feedback,
        )

        messages = [
            {"role": "user", "content": optimizer_prompt}
        ]

        new_prompt = self.llm_client.chat(
            model=self.judge_model,
            messages=messages,
            temperature=self.judge_temperature,
            max_tokens=self.judge_max_tokens,
            iteration=iteration,
            step="optimizer",
            role="judge",
        )

        return new_prompt.strip()