"""Configuration registries for data generation pipeline."""

import os
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DATASET_CONFIGS",
    "DatasetConfig",
    "resolve_normalize_fn",
]


NormalizeFn = Callable[[dict], dict]


@dataclass(kw_only=True)
class DatasetConfig:
    """Configuration for loading a dataset.

    The conversation schema can be described declaratively so a normalizer is
    generated from config instead of hand-written (see ``resolve_normalize_fn``):

    - ``conversations_column``: name of the column already holding a list of
      conversation turns; it is renamed to the canonical ``"conversations"``.
    - ``prompt_column`` / ``answer_column``: build a two-turn (user, assistant)
      conversation from a prompt/response pair.

    ``normalize_fn`` remains an explicit escape hatch for schemas that can't be
    expressed with the hints above (e.g. multi-modal datasets).
    """

    name: str
    hf_path: str
    subset: str | None = None
    split: str
    filter_fn: Callable[[dict], bool] | None = None
    conversations_column: str = "conversations"
    prompt_column: str | None = None
    answer_column: str | None = None
    normalize_fn: NormalizeFn | None = None


def _make_rename_normalizer(source_column: str) -> NormalizeFn:
    """Build a normalizer that renames ``source_column`` to ``"conversations"``."""

    def normalize(example: dict) -> dict:
        if "conversations" in example or source_column not in example:
            return example
        return {"conversations": example[source_column]}

    normalize.__name__ = f"normalize_rename_{source_column}"
    return normalize


def _make_prompt_response_normalizer(
    prompt_column: str,
    answer_column: str,
) -> NormalizeFn:
    """Build a normalizer that turns a prompt/answer pair into a conversation."""

    def normalize(example: dict) -> dict:
        if "conversations" in example:
            return example
        return {
            "conversations": [
                {"role": "user", "content": example[prompt_column]},
                {"role": "assistant", "content": example[answer_column]},
            ]
        }

    normalize.__name__ = f"normalize_{prompt_column}_{answer_column}"
    return normalize


def resolve_normalize_fn(config: DatasetConfig) -> NormalizeFn | None:
    """Resolve the normalizer for a dataset config.

    An explicit ``normalize_fn`` always wins; otherwise one is generated from the
    declarative schema hints. Returns ``None`` when the dataset is already in the
    canonical ``conversations`` schema (the ``messages``-column case is handled
    generically during ingestion, so it needs no per-dataset normalizer).
    """
    if config.normalize_fn is not None:
        return config.normalize_fn
    if config.prompt_column is not None and config.answer_column is not None:
        return _make_prompt_response_normalizer(
            config.prompt_column, config.answer_column
        )
    if config.conversations_column != "conversations":
        return _make_rename_normalizer(config.conversations_column)
    return None


def get_coco_dir():
    return os.getenv("COCO_DIR") or "coco/"


def _parse_sharegpt4v_part(part: str, image_path: str):
    if part == "<image>":
        return {"type": "image", "path": image_path}

    return {"type": "text", "text": part}


def _parse_sharegpt4v_user_content(content: str, image_path: str):
    return [_parse_sharegpt4v_part(part, image_path) for part in content.split("\n")]


def _parse_sharegpt4v_assistant_content(content: str):
    return [{"type": "text", "text": content}]


def _filter_sharegpt4v_coco(example: dict) -> bool:
    return example["image"].startswith("coco/")


def _normalize_sharegpt4v_coco(example: dict) -> dict:
    coco_dir = get_coco_dir()
    image_path = os.path.join(coco_dir, example["image"].removeprefix("coco/"))

    if not os.path.exists(image_path):
        state_str = "set to" if os.getenv("COCO_DIR") else "default"

        raise ValueError(
            f"No image found at <{image_path}>. "
            f"Please download COCO 2017 Train Images from "
            f"<http://images.cocodataset.org/zips/train2017.zip> and place the "
            f"extracted folder under `COCO_DIR` ({state_str}: `{coco_dir}`)."
        )

    messages = [
        (
            turn
            | {
                "value": (
                    _parse_sharegpt4v_user_content(turn["value"], image_path)
                    if turn["from"] in ("human", "user")
                    else _parse_sharegpt4v_assistant_content(turn["value"])
                )
            }
        )
        for turn in example["conversations"]
    ]

    return {"conversations": messages}


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sharegpt": DatasetConfig(
        name="sharegpt",
        hf_path="Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
    ),
    "ultrachat": DatasetConfig(
        name="ultrachat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        # 'messages' column is renamed to 'conversations' automatically.
    ),
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_path="openai/gsm8k",
        subset="main",
        split="train",
        prompt_column="question",
        answer_column="answer",
    ),
    # NOTE: You need to serve vLLM with `--allowed-local-media-path /path/to/coco`
    "sharegpt4v_coco": DatasetConfig(
        name="sharegpt4v_coco",
        hf_path="Lin-Chen/ShareGPT4V",
        subset="ShareGPT4V",
        split="train",
        filter_fn=_filter_sharegpt4v_coco,
        normalize_fn=_normalize_sharegpt4v_coco,
    ),
}
