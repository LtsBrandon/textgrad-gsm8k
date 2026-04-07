import yaml
import json
import random
from pathlib import Path
from datetime import datetime


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> list[dict]:
    """
    Load a dataset from a JSONL file.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    return data


def load_prompt_template(file_path: str) -> str:
    """
    Load a prompt template from a text file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fill_template(template: str, **kwargs) -> str:
    """
    Fill a template string with values.
    """
    return template.format(**kwargs)


def create_run_dir(base_dir: str) -> Path:
    """
    Create a timestamped directory to store experiment results.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sample_batch(
    dataset: list[dict],
    batch_size: int,
    with_replacement: bool = True,
    rng=None
) -> list[dict]:
    """
    Sample a mini-batch from the dataset.
    """
    if not dataset:
        raise ValueError("dataset cannot be empty")

    rng = rng or random

    if with_replacement:
        return [rng.choice(dataset) for _ in range(batch_size)]

    if batch_size > len(dataset):
        raise ValueError(
            "batch_size cannot exceed dataset size when sampling without replacement"
        )

    return rng.sample(dataset, batch_size)


def save_json(data: dict | list, file_path: str) -> None:
    """
    Save data as a formatted JSON file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_jsonl(data: dict, file_path: str) -> None:
    """
    Append a single record to a JSONL file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")