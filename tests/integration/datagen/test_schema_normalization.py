"""Unit tests for declarative schema normalization (config-generated normalizers,
role aliases, and the automatic messages -> conversations rename)."""

import pytest
from datasets import Dataset as HFDataset

from speculators.data_generation.configs import (
    DATASET_CONFIGS,
    DatasetConfig,
    resolve_normalize_fn,
)
from speculators.data_generation.preprocessing import (
    ROLE_ALIASES,
    _normalize_conversation,
    _normalize_turn,
    _rename_messages_to_conversations,
)


def _cfg(**overrides) -> DatasetConfig:
    base = {"name": "t", "hf_path": "x", "split": "train"}
    return DatasetConfig(**{**base, **overrides})


# ---------------------------------------------------------------------------
# _normalize_turn / ROLE_ALIASES
# ---------------------------------------------------------------------------


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("raw_role", "expected"),
    [
        ("human", "user"),
        ("user", "user"),
        ("gpt", "assistant"),
        ("assistant", "assistant"),
        ("system", "system"),
        ("tool", "tool"),
        ("observation", "tool"),
    ],
)
def test_normalize_turn_role_aliases(raw_role, expected):
    assert _normalize_turn({"from": raw_role, "value": "x"})["role"] == expected
    assert _normalize_turn({"role": raw_role, "content": "x"})["role"] == expected


@pytest.mark.sanity
def test_observation_maps_to_tool_not_user():
    # observation is a tool result; mapping it to user would break the tool-turn
    # contract (it can still carry tool_call_id).
    assert ROLE_ALIASES["observation"] == "tool"


@pytest.mark.sanity
def test_normalize_turn_unknown_role_returns_none():
    assert _normalize_turn({"role": "narrator", "content": "x"}) is None


@pytest.mark.sanity
def test_normalize_turn_preserves_thinking_and_reasoning_independently():
    result = _normalize_turn(
        {
            "role": "assistant",
            "content": "answer",
            "thinking": "T",
            "reasoning_content": "R",
        }
    )
    assert result["thinking"] == "T"
    assert result["reasoning_content"] == "R"


@pytest.mark.sanity
def test_normalize_turn_preserves_tool_fields():
    result = _normalize_turn(
        {
            "role": "tool",
            "content": "result",
            "tool_calls": [{"id": "c1"}],
            "tool_call_id": "c1",
        }
    )
    assert result["tool_calls"] == [{"id": "c1"}]
    assert result["tool_call_id"] == "c1"


@pytest.mark.sanity
def test_normalize_conversation_empty_is_safe():
    # turn_dropout calls random.randint(1, len(conv)); an empty conv must not raise.
    assert _normalize_conversation([], turn_dropout=True) == []


# ---------------------------------------------------------------------------
# resolve_normalize_fn (config-generated normalizers)
# ---------------------------------------------------------------------------


@pytest.mark.sanity
def test_resolve_normalize_fn_default_is_none():
    # Canonical 'conversations' schema (and the 'messages' case, handled by the
    # automatic rename) needs no generated normalizer.
    assert resolve_normalize_fn(_cfg()) is None


@pytest.mark.sanity
def test_resolve_normalize_fn_prompt_answer_builds_conversation():
    fn = resolve_normalize_fn(_cfg(prompt_column="question", answer_column="answer"))
    assert fn is not None
    assert fn({"question": "Q", "answer": "A"})["conversations"] == [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]


@pytest.mark.sanity
def test_resolve_normalize_fn_prompt_answer_passthrough_if_conversations():
    fn = resolve_normalize_fn(_cfg(prompt_column="q", answer_column="a"))
    existing = {"conversations": [{"role": "user", "content": "hi"}]}
    assert fn(existing) == existing


@pytest.mark.sanity
def test_resolve_normalize_fn_conversations_column_rename():
    fn = resolve_normalize_fn(_cfg(conversations_column="dialogue"))
    assert fn is not None
    out = fn({"dialogue": [{"role": "user", "content": "hi"}]})
    assert out["conversations"] == [{"role": "user", "content": "hi"}]


@pytest.mark.sanity
def test_resolve_normalize_fn_explicit_fn_is_escape_hatch():
    def custom(example: dict) -> dict:
        return {"conversations": []}

    assert resolve_normalize_fn(_cfg(normalize_fn=custom)) is custom


@pytest.mark.sanity
def test_gsm8k_preset_uses_prompt_answer_hints():
    cfg = DATASET_CONFIGS["gsm8k"]
    assert cfg.normalize_fn is None
    assert cfg.prompt_column == "question"
    assert cfg.answer_column == "answer"


@pytest.mark.sanity
def test_ultrachat_preset_needs_no_normalizer():
    cfg = DATASET_CONFIGS["ultrachat"]
    assert cfg.normalize_fn is None
    assert resolve_normalize_fn(cfg) is None


# ---------------------------------------------------------------------------
# automatic messages -> conversations rename
# ---------------------------------------------------------------------------


@pytest.mark.sanity
def test_rename_messages_to_conversations():
    ds = HFDataset.from_dict({"messages": [[{"role": "user", "content": "hi"}]]})
    out = _rename_messages_to_conversations(ds)
    assert "conversations" in out.column_names
    assert "messages" not in out.column_names


@pytest.mark.sanity
def test_rename_drops_messages_when_both_present():
    ds = HFDataset.from_dict(
        {
            "messages": [[{"role": "user", "content": "hi"}]],
            "conversations": [[{"role": "user", "content": "hi"}]],
        }
    )
    out = _rename_messages_to_conversations(ds)
    assert "messages" not in out.column_names
    assert "conversations" in out.column_names


@pytest.mark.sanity
def test_rename_noop_when_no_messages():
    ds = HFDataset.from_dict({"conversations": [[{"role": "user", "content": "hi"}]]})
    assert _rename_messages_to_conversations(ds).column_names == ["conversations"]
