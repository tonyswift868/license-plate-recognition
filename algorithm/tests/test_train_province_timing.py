"""Timing proxies preserve inference results and never add calls."""
from __future__ import annotations

import pytest

from lpr_tools.train.province_inference_timing import instrument_method, summarize


def test_timed_method_returns_original_result_and_records_once():
    result = object()

    class Reader:
        def __init__(self):
            self.calls = 0

        def predict_proba(self, batch):
            self.calls += 1
            assert batch == [7] * 7
            return result

    reader = Reader()
    records = []
    instrument_method(reader, "predict_proba", records,
                      batch_size=lambda args, _kwargs: len(args[0]))
    actual = reader.predict_proba([7] * 7)
    assert actual is result
    assert reader.calls == 1
    assert len(records) == 1 and records[0]["batch_size"] == 7
    assert records[0]["elapsed_ms"] >= 0


def test_timed_method_propagates_the_same_exception_and_records_once():
    error = ValueError("fixture failure")

    class Reader:
        def read(self, _crop):
            raise error

    reader = Reader()
    records = []
    instrument_method(reader, "read", records,
                      batch_size=lambda _args, _kwargs: 1)
    with pytest.raises(ValueError) as raised:
        reader.read(object())
    assert raised.value is error
    assert len(records) == 1 and records[0]["batch_size"] == 1


def test_summary_retains_every_n1_n7_n8_call_including_first():
    records = [
        {"batch_size": 1, "elapsed_ms": 8.0},
        {"batch_size": 7, "elapsed_ms": 7.0},
        {"batch_size": 8, "elapsed_ms": 6.0},
        {"batch_size": 1, "elapsed_ms": 2.0},
    ]
    summary = summarize(records, buckets=(1, 7, 8))
    assert summary["by_batch_size"]["1"]["all_calls_elapsed_ms"] == [8.0, 2.0]
    assert summary["by_batch_size"]["1"]["mean_ms"] == 5.0
    assert summary["by_batch_size"]["1"]["p95_nearest_rank_ms"] == 8.0
    assert summary["by_batch_size"]["7"]["calls"] == 1
    assert summary["by_batch_size"]["8"]["calls"] == 1
