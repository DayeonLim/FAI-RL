"""Tests for ZeRO-3 partitioned-load activation in BaseTrainer.

The crux of the fix is that ``from_pretrained`` must see ``is_deepspeed_zero3_enabled()``
== True so it loads the model sharded per rank instead of fully on every rank
(which OOMs the host on large models). That end-to-end behavior requires a GPU +
deepspeed and is exercised on the cluster; here we lock down the pure decision
logic that gates it: ``BaseTrainer._deepspeed_zero_stage``.
"""

import json

from core.trainer_base import BaseTrainer


def test_stage_from_zero3_path(tmp_path):
    cfg = tmp_path / "zero3.json"
    cfg.write_text(json.dumps({"zero_optimization": {"stage": 3}}))
    assert BaseTrainer._deepspeed_zero_stage(str(cfg)) == 3


def test_stage_from_zero1_path(tmp_path):
    cfg = tmp_path / "zero1.json"
    cfg.write_text(json.dumps({"zero_optimization": {"stage": 1}}))
    assert BaseTrainer._deepspeed_zero_stage(str(cfg)) == 1


def test_stage_from_dict():
    assert BaseTrainer._deepspeed_zero_stage({"zero_optimization": {"stage": 3}}) == 3


def test_none_when_unset():
    assert BaseTrainer._deepspeed_zero_stage(None) is None
    assert BaseTrainer._deepspeed_zero_stage("") is None


def test_none_when_no_zero_block():
    assert BaseTrainer._deepspeed_zero_stage({"train_batch_size": "auto"}) is None


def test_none_when_file_missing():
    assert BaseTrainer._deepspeed_zero_stage("/nonexistent/zero3.json") is None


def test_none_when_unparseable(tmp_path):
    cfg = tmp_path / "broken.json"
    cfg.write_text("{not json")
    assert BaseTrainer._deepspeed_zero_stage(str(cfg)) is None


def test_real_repo_zero3_config_is_stage_3():
    """The shipped ZeRO-3 config the platform points VLM LoRA at must be detected."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    cfg = repo_root / "configs" / "deepspeed" / "zero3_config.json"
    assert BaseTrainer._deepspeed_zero_stage(str(cfg)) == 3
