"""Benchmark results and the uncertainty around them.

At the sample sizes anyone runs locally, two models a few points apart are
usually tied. These tests pin down the two places that claim can be lost: the
interval arithmetic, and `parse_results`, which has to find a score inside
lm-eval's JSON without silently reporting a *different* metric than the one
that was asked for — which is how two runs of the "same" benchmark stop being
comparable.
"""
from __future__ import annotations

import pytest

import harness


def result(score: float, stderr: float | None, task_key: str = "gsm8k") -> harness.Result:
    return harness.Result(
        task_key=task_key,
        metric_key="exact_match,strict-match",
        score=score,
        stderr=stderr,
        samples=50,
        model="local:llama3.1",
        limit=50,
        num_fewshot=5,
    )


class TestTaskCatalogue:
    def test_task_keys_are_unique(self):
        keys = [t.key for t in harness.TASKS]
        assert len(keys) == len(set(keys))

    def test_get_task_resolves_a_known_key(self):
        assert harness.get_task("gsm8k").key == "gsm8k"

    def test_every_task_declares_the_metric_it_reads(self):
        for task in harness.TASKS:
            assert task.metric_key, f"{task.key} has no metric_key"

    def test_every_task_carries_a_blurb_and_a_source(self):
        for task in harness.TASKS:
            assert task.blurb.strip(), f"{task.key} has no blurb"
            assert task.source_url.startswith("http"), f"{task.key} has no source"

    def test_every_default_limit_is_positive(self):
        for task in harness.TASKS:
            assert task.default_limit is None or task.default_limit > 0


class TestInterval:
    def test_percent_is_the_score_scaled_and_rounded(self):
        assert result(0.6234, None).percent == 62.3

    def test_margin_is_the_95_percent_half_width_in_points(self):
        assert result(0.62, 0.04).margin == pytest.approx(0.04 * harness.Z_95 * 100, abs=0.05)

    def test_a_missing_stderr_has_no_margin(self):
        assert result(0.62, None).margin is None

    def test_the_interval_travels_with_the_number(self):
        assert result(0.62, 0.04).interval_label == "62.0% ± 7.8"

    def test_a_score_without_an_interval_is_shown_bare(self):
        assert result(0.62, None).interval_label == "62.0%"

    def test_a_zero_stderr_still_reports_a_margin(self):
        assert result(0.62, 0.0).margin == 0.0


class TestTiesWith:
    def test_overlapping_intervals_are_a_tie(self):
        # 61.2 ± 7.8 against 58.9 ± 7.8: a 2.3-point gap inside a 15-point
        # combined margin is not a result.
        assert result(0.612, 0.04).ties_with(result(0.589, 0.04)) is True

    def test_clearly_separated_intervals_are_not_a_tie(self):
        assert result(0.90, 0.01).ties_with(result(0.40, 0.01)) is False

    def test_the_comparison_is_symmetric(self):
        a, b = result(0.62, 0.03), result(0.55, 0.04)
        assert a.ties_with(b) == b.ties_with(a)

    def test_a_result_ties_with_itself(self):
        a = result(0.62, 0.04)
        assert a.ties_with(a) is True

    def test_touching_intervals_count_as_tied(self):
        # The conservative direction: exactly-touching is treated as
        # indistinguishable rather than as a win.
        a = result(0.60, 0.01)   # margin 1.96 points
        b = result(0.6392, 0.01)  # gap 3.92 == 1.96 + 1.96
        assert a.ties_with(b) is True

    def test_no_tie_claim_is_made_without_intervals(self):
        # Without a stderr there is no basis for either claim, and False here
        # means "not established as tied", not "significantly different".
        assert result(0.62, None).ties_with(result(0.55, 0.04)) is False
        assert result(0.62, 0.04).ties_with(result(0.55, None)) is False


class TestAsDict:
    def test_carries_the_interval_alongside_the_score(self):
        payload = result(0.62, 0.04).as_dict()
        assert payload["score_percent"] == 62.0
        assert payload["margin_95"] == pytest.approx(7.8, abs=0.05)
        assert payload["samples"] == 50

    def test_records_the_run_shape_that_makes_scores_comparable(self):
        payload = result(0.62, 0.04).as_dict()
        assert payload["limit"] == 50
        assert payload["num_fewshot"] == 5
        assert payload["metric"] == "exact_match,strict-match"


class TestParseResults:
    def test_reads_the_requested_metric_and_its_stderr(self):
        payload = {
            "results": {"gsm8k": {
                "exact_match,strict-match": 0.62,
                "exact_match_stderr,strict-match": 0.04,
            }},
            "n-samples": {"gsm8k": {"effective": 50, "original": 1319}},
            "configs": {"gsm8k": {"num_fewshot": 5}},
            "config": {"model": "local:llama3.1", "limit": 50},
        }
        parsed = harness.parse_results(payload, harness.get_task("gsm8k"))
        assert parsed.score == 0.62
        assert parsed.stderr == 0.04
        assert parsed.metric_key == "exact_match,strict-match"

    def test_records_the_effective_sample_count_not_the_full_task_size(self):
        # A 50-question run must not report 1319 samples; the interval depends
        # on what was actually scored.
        payload = {
            "results": {"gsm8k": {"exact_match,strict-match": 0.62}},
            "n-samples": {"gsm8k": {"effective": 50, "original": 1319}},
        }
        assert harness.parse_results(payload, harness.get_task("gsm8k")).samples == 50

    def test_falls_back_to_the_original_count_when_effective_is_absent(self):
        payload = {
            "results": {"gsm8k": {"exact_match,strict-match": 0.62}},
            "n-samples": {"gsm8k": {"original": 1319}},
        }
        assert harness.parse_results(payload, harness.get_task("gsm8k")).samples == 1319

    def test_a_task_group_reporting_under_another_key_is_still_read(self):
        # Some lm-eval task groups report under a variant key; a single
        # unambiguous block is accepted.
        payload = {"results": {"gsm8k_cot": {"exact_match,strict-match": 0.5}}}
        assert harness.parse_results(payload, harness.get_task("gsm8k")).score == 0.5

    def test_an_unrequested_metric_is_used_but_reported_under_its_own_name(self):
        # Reporting a different metric silently is how two runs of the "same"
        # benchmark stop being comparable, so the substituted name must
        # survive onto the Result.
        payload = {"results": {"gsm8k": {"flexible_match,none": 0.71}}}
        parsed = harness.parse_results(payload, harness.get_task("gsm8k"))
        assert parsed.score == 0.71
        assert parsed.metric_key == "flexible_match,none"

    def test_a_stderr_is_not_mistaken_for_the_score(self):
        payload = {"results": {"gsm8k": {
            "exact_match_stderr,strict-match": 0.04,
            "flexible_match,none": 0.71,
        }}}
        parsed = harness.parse_results(payload, harness.get_task("gsm8k"))
        assert parsed.score == 0.71

    def test_empty_results_raise_and_say_what_was_reported(self):
        with pytest.raises(harness.HarnessUnavailable) as excinfo:
            harness.parse_results({"results": {}}, harness.get_task("gsm8k"))
        assert "nothing" in str(excinfo.value)

    def test_a_missing_results_block_raises(self):
        with pytest.raises(harness.HarnessUnavailable):
            harness.parse_results({}, harness.get_task("gsm8k"))

    def test_several_unrelated_blocks_raise_rather_than_guessing(self):
        # With two candidates and neither matching, picking one would be a
        # coin flip presented as a measurement.
        payload = {"results": {"hellaswag": {"acc,none": 0.5},
                               "arc_challenge": {"acc,none": 0.4}}}
        with pytest.raises(harness.HarnessUnavailable) as excinfo:
            harness.parse_results(payload, harness.get_task("gsm8k"))
        assert "hellaswag" in str(excinfo.value)

    def test_a_block_with_no_numeric_metric_raises(self):
        payload = {"results": {"gsm8k": {"alias": "gsm8k"}}}
        with pytest.raises(harness.HarnessUnavailable) as excinfo:
            harness.parse_results(payload, harness.get_task("gsm8k"))
        assert "No numeric metric" in str(excinfo.value)

    def test_a_missing_stderr_leaves_the_result_without_an_interval(self):
        payload = {"results": {"gsm8k": {"exact_match,strict-match": 0.62}}}
        parsed = harness.parse_results(payload, harness.get_task("gsm8k"))
        assert parsed.stderr is None
        assert parsed.interval_label == "62.0%"

    def test_missing_sample_counts_default_to_zero_rather_than_raising(self):
        payload = {"results": {"gsm8k": {"exact_match,strict-match": 0.62}}}
        assert harness.parse_results(payload, harness.get_task("gsm8k")).samples == 0

    def test_the_raw_block_is_retained_for_inspection(self):
        block = {"exact_match,strict-match": 0.62, "alias": "gsm8k"}
        parsed = harness.parse_results({"results": {"gsm8k": block}},
                                       harness.get_task("gsm8k"))
        assert parsed.raw == block
