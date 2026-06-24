import pytest
import torch

from scripts.train import parse_args
from speculators.train.trainer import (
    TrainerConfig,
    _prepare_metrics_for_logging,
    _resolve_scheduler_steps,
)


def make_config(**overrides) -> TrainerConfig:
    return TrainerConfig(
        lr=1e-4,
        num_epochs=5,
        save_path="checkpoint",
        **overrides,
    )


def test_scheduler_steps_default_to_one_percent_of_training_steps():
    warmup_steps, total_steps = _resolve_scheduler_steps(make_config(), 20)

    assert total_steps == 100
    assert warmup_steps == 1


def test_scheduler_warmup_ratio_uses_scheduler_total_steps():
    warmup_steps, total_steps = _resolve_scheduler_steps(
        make_config(scheduler_total_steps=200, scheduler_warmup_ratio=0.1),
        20,
    )

    assert total_steps == 200
    assert warmup_steps == 20


def test_scheduler_warmup_steps_take_precedence_over_ratio():
    with pytest.warns(UserWarning, match="using scheduler_warmup_steps"):
        warmup_steps, total_steps = _resolve_scheduler_steps(
            make_config(scheduler_warmup_steps=0, scheduler_warmup_ratio=0.1),
            20,
        )

    assert total_steps == 100
    assert warmup_steps == 0


def test_scheduler_warmup_ratio_must_be_between_zero_and_one():
    with pytest.raises(ValueError, match="scheduler_warmup_ratio"):
        _resolve_scheduler_steps(make_config(scheduler_warmup_ratio=1.1), 20)


def test_scheduler_type_rejects_unsupported_values(monkeypatch):
    # --verifier-name-or-path is supplied so the only parse failure is the rejected
    # --scheduler-type choice (not the missing required verifier arg).
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--verifier-name-or-path", "x", "--scheduler-type", "constant"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_prepare_metrics_for_logging_normalizes_sum_total_pairs():
    metrics = {
        "loss_sum": torch.tensor(6.0),
        "loss_total": torch.tensor(3.0),
        "full_acc_sum": torch.tensor(8.0),
        "full_acc_total": torch.tensor(10.0),
        "position_1_acc_sum": torch.tensor(4.0),
        "position_1_acc_total": torch.tensor(5.0),
    }

    assert _prepare_metrics_for_logging(metrics) == {
        "loss": 2.0,
        "full_acc": 0.8,
        "position_1_acc": 0.8,
    }


def test_prepare_metrics_for_logging_averages_plain_metrics_by_world_size():
    metrics = {
        "throughput": torch.tensor(12.0),
    }

    assert _prepare_metrics_for_logging(metrics, world_size=3) == {
        "throughput": 4.0,
    }
