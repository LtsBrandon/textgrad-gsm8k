import json
import random
import re
from pathlib import Path

from datasets import load_dataset


SEED = 679
TRAIN_COUNT = 20
VAL_COUNT = 15
TEST_COUNT = 20
GROUND_TRUTH_PATTERN = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")


def parse_ground_truth(answer_text: str) -> int | float:
    match = GROUND_TRUTH_PATTERN.search(answer_text)
    if not match:
        raise ValueError(f"Could not extract ground truth from answer_text: {answer_text!r}")

    numeric_text = match.group(1).replace(",", "")
    value = float(numeric_text)
    return int(value) if value.is_integer() else value


def make_record(split_name: str, source_index: int, example: dict) -> dict:
    return {
        "id": f"gsm8k_{split_name}_{source_index:04d}",
        "question": example["question"],
        "answer_text": example["answer"],
        "ground_truth": parse_ground_truth(example["answer"]),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    rng = random.Random(SEED)

    dataset = load_dataset("openai/gsm8k", "main")
    source_train = dataset["train"]
    source_test = dataset["test"]

    sampled_train_indices = rng.sample(range(len(source_train)), TRAIN_COUNT + VAL_COUNT)
    sampled_test_indices = rng.sample(range(len(source_test)), TEST_COUNT)

    train_indices = sampled_train_indices[:TRAIN_COUNT]
    val_indices = sampled_train_indices[TRAIN_COUNT:]

    train_records = [make_record("train", idx, source_train[idx]) for idx in train_indices]
    val_records = [make_record("val", idx, source_train[idx]) for idx in val_indices]
    test_records = [make_record("test", idx, source_test[idx]) for idx in sampled_test_indices]

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    write_jsonl(output_dir / "test.jsonl", test_records)


if __name__ == "__main__":
    main()
