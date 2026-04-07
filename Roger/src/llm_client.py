from openai import OpenAI
import time
import json
from datetime import datetime
from pathlib import Path


class LLMClient:
    """
    Wrapper for OpenAI-compatible API calls.

    This client is designed to work with LM Studio (local server),
    which exposes an OpenAI-compatible API.

    Responsibilities:
    - Send chat completion requests
    - Retry on transient failures
    - Log every request/response to a JSONL file
    """

    def __init__(
        self,
        api_base_url: str,
        api_key: str = "lm-studio",
        log_file: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        """
        Initialize the client.

        Args:
            api_base_url: Base URL for LM Studio
            api_key: Dummy key (required by SDK but ignored by LM Studio)
            log_file: Optional path to JSONL log file
            max_retries: Number of retry attempts for failed calls
            retry_delay: Delay (seconds) between retries
        """

        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key
        )

        # logging and retry config
        self.log_file = Path(log_file) if log_file else None
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # ensure log directory exists (if logging is enabled)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        iteration: int | None = None,
        step: str | None = None,
        role: str | None = None,
    ) -> str:
        """
        Send a chat completion request and return the response text.

        This method:
        1. Calls the LLM
        2. Retries on failure
        3. Logs the full interaction

        Args:
            model: Model name registered in LM Studio
            messages: Chat messages (OpenAI format)
            temperature: Sampling temperature
            max_tokens: Max tokens for response
            iteration: Current optimization iteration (for logging)
            step: Pipeline step ("forward", "loss", "gradient", etc.)
            role: Logical role ("solver" or "judge")

        Returns:
            Assistant response text

        Raises:
            RuntimeError if all retry attempts fail
        """

        last_error = None

        # retry loop
        for attempt in range(self.max_retries):
            start = time.time()  # track latency

            try:
                # send request to LLM
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                
                latency_ms = (time.time() - start) * 1000
                text = response.choices[0].message.content.strip()
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)

                # log this call (for analysis/debugging)
                self._log_call(
                    model=model,
                    messages=messages,
                    response=text,
                    tokens_prompt=prompt_tokens,
                    tokens_completion=completion_tokens,
                    latency_ms=latency_ms,
                    iteration=iteration,
                    step=step,
                    role=role
                )

                return text  # success

            except Exception as e:
                # save error and retry
                last_error = e
                time.sleep(self.retry_delay)

        # raise error
        raise RuntimeError(f"LLM call failed: {last_error}")

    def _log_call(
        self,
        model: str,
        messages: list[dict],
        response: str,
        tokens_prompt: int,
        tokens_completion: int,
        latency_ms: float,
        iteration: int | None = None,
        step: str | None = None,
        role: str | None = None
    ) -> None:
        """
        Write one JSON line to the log file.

        Each line corresponds to ONE LLM call.

        Logging is critical for:
        - debugging model behavior
        - analyzing performance
        - generating plots/results for the report
        """

        if self.log_file is None:
            return

        # construct log record
        record = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "step": step,
            "role": role,
            "model": model,
            "messages": messages,
            "response": response,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "latency_ms": round(latency_ms, 2),
        }

        # append to JSONL file (one line per call)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")