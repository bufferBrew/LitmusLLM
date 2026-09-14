"""Speed and cost measurement.

Two things here are worth defending with tests. The percentile function is
hand-written to avoid a numpy dependency, so it needs to agree with numpy's
default method or the p90 figures quietly diverge from what anyone comparing
against another tool expects. And the labels encode a distinction the numbers
alone lose: a local run that genuinely cost nothing must not render the same
as a cloud run nobody could price.
"""
from __future__ import annotations

import pytest

import perf


class TestPercentile:
    def test_empty_input_has_no_percentile(self):
        assert perf.percentile([], 50) is None

    def test_a_single_value_is_its_own_percentile(self):
        assert perf.percentile([7.0], 50) == 7.0
        assert perf.percentile([7.0], 99) == 7.0

    def test_median_of_an_odd_series(self):
        assert perf.percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_median_of_an_even_series_interpolates(self):
        assert perf.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5

    def test_p0_and_p100_are_the_extremes(self):
        values = [5.0, 1.0, 9.0, 3.0]
        assert perf.percentile(values, 0) == 1.0
        assert perf.percentile(values, 100) == 9.0

    def test_input_order_does_not_matter(self):
        assert perf.percentile([3.0, 1.0, 2.0], 50) == perf.percentile([1.0, 2.0, 3.0], 50)

    def test_the_input_sequence_is_not_mutated(self):
        values = [3.0, 1.0, 2.0]
        perf.percentile(values, 50)
        assert values == [3.0, 1.0, 2.0]

    @pytest.mark.parametrize("p,expected", [(50, 2.5), (90, 3.7), (99, 3.97)])
    def test_interpolation_matches_numpys_default_method(self, p, expected):
        # numpy.percentile([1,2,3,4], p) with the default 'linear' method.
        assert perf.percentile([1.0, 2.0, 3.0, 4.0], p) == pytest.approx(expected)

    def test_identical_values_give_that_value_at_every_percentile(self):
        assert perf.percentile([2.0, 2.0, 2.0], 90) == 2.0


class TestSummarise:
    def test_an_empty_series_is_marked_unmeasured(self):
        dist = perf.summarise("Latency", "ms", [])
        assert dist.n == 0
        assert dist.measured is False
        assert dist.mean is None

    def test_unmeasured_entries_are_dropped_not_counted_as_zero(self):
        # A None is "we did not measure this call", not "this call took 0ms".
        # Counting it as zero would drag every percentile down.
        dist = perf.summarise("Latency", "ms", [1.0, None, 3.0])
        assert dist.n == 2
        assert dist.mean == 2.0

    def test_a_series_of_only_nones_is_unmeasured(self):
        assert perf.summarise("Latency", "ms", [None, None]).measured is False

    def test_records_the_full_distribution(self):
        dist = perf.summarise("Latency", "ms", [1.0, 2.0, 3.0, 4.0])
        assert dist.minimum == 1.0
        assert dist.maximum == 4.0
        assert dist.p50 == 2.5
        assert dist.mean == 2.5

    def test_label_and_unit_are_carried_through(self):
        dist = perf.summarise("Time to first token", "ms", [1.0])
        assert dist.label == "Time to first token"
        assert dist.unit == "ms"

    def test_as_dict_rounds_for_display_and_renames_the_extremes(self):
        dist = perf.summarise("Latency", "ms", [1.0, 2.3456])
        payload = dist.as_dict()
        assert payload["mean"] == 1.67
        assert payload["min"] == 1.0
        assert payload["max"] == 2.35

    def test_as_dict_keeps_nones_as_null_for_an_unmeasured_series(self):
        payload = perf.summarise("Latency", "ms", []).as_dict()
        assert payload["n"] == 0
        assert payload["p50"] is None


class TestFormatMs:
    def test_unmeasured_renders_as_a_dash(self):
        assert perf.RunPerf.format_ms(None) == "--"

    def test_sub_second_values_stay_in_milliseconds(self):
        assert perf.RunPerf.format_ms(250) == "250 ms"

    def test_milliseconds_are_shown_without_decimals(self):
        assert perf.RunPerf.format_ms(216.7) == "217 ms"

    def test_one_second_is_the_switch_point(self):
        assert perf.RunPerf.format_ms(999) == "999 ms"
        assert perf.RunPerf.format_ms(1000) == "1.00 s"

    def test_seconds_carry_two_decimals(self):
        assert perf.RunPerf.format_ms(2216) == "2.22 s"

    def test_zero_is_a_measurement_not_a_blank(self):
        assert perf.RunPerf.format_ms(0) == "0 ms"


class TestRunPerfLabels:
    def test_an_uninstrumented_run_reports_nothing_measured(self):
        run = perf.RunPerf()
        assert run.measured is False
        assert run.ttft_label == "--"
        assert run.tps_label == "--"

    def test_percentiles_are_read_from_the_stored_distributions(self):
        run = perf.RunPerf(
            calls=10,
            ttft={"p50": 320.0, "p90": 890.0},
            output_tps={"p50": 42.5, "p90": 51.0},
        )
        assert run.measured is True
        assert run.ttft_p50 == 320.0
        assert run.ttft_p90 == 890.0
        assert run.tps_p50 == 42.5
        assert run.tps_p90 == 51.0

    def test_labels_render_the_p50_figures(self):
        run = perf.RunPerf(calls=1, ttft={"p50": 1500.0}, output_tps={"p50": 42.53})
        assert run.ttft_label == "1.50 s"
        assert run.tps_label == "42.5 tok/s"

    def test_a_run_with_timings_but_no_token_counts_says_so(self):
        run = perf.RunPerf(calls=5, ttft={"p50": 100.0})
        assert run.has_tokens is False

    def test_token_counts_are_detected_from_either_direction(self):
        assert perf.RunPerf(prompt_tokens=10).has_tokens is True
        assert perf.RunPerf(completion_tokens=10).has_tokens is True
        assert perf.RunPerf().has_tokens is False

    def test_a_p90_present_without_a_p50_still_renders_a_dash(self):
        run = perf.RunPerf(calls=1, ttft={"p90": 900.0})
        assert run.ttft_label == "--"
        assert run.ttft_p90_label == "900 ms"


class TestCostLabel:
    def test_free_and_unpriced_are_not_the_same_thing(self):
        # A local run has genuinely no invoice; an unpriced cloud model is a
        # gap in our data. Rendering both as "$0.00" would turn one into the
        # other.
        assert perf.RunPerf(total_cost_usd=0.0).cost_label == "no API cost"
        assert perf.RunPerf(total_cost_usd=None).cost_label == "not priced"

    def test_sub_cent_costs_keep_four_decimals(self):
        # At two decimals every cheap run reads "$0.00", which is the free
        # label again.
        assert perf.RunPerf(total_cost_usd=0.0004).cost_label == "$0.0004"
        assert perf.RunPerf(total_cost_usd=0.0099).cost_label == "$0.0099"

    def test_costs_of_a_cent_or_more_use_two_decimals(self):
        assert perf.RunPerf(total_cost_usd=0.01).cost_label == "$0.01"
        assert perf.RunPerf(total_cost_usd=1.234).cost_label == "$1.23"
        assert perf.RunPerf(total_cost_usd=12.5).cost_label == "$12.50"


class TestFromRun:
    def test_a_run_recorded_before_instrumentation_yields_none(self):
        assert perf.from_run({"perf_json": None}) is None
        assert perf.from_run({}) is None

    def test_unparseable_perf_json_does_not_raise(self):
        # One corrupt run must not break the page listing every other run.
        assert perf.from_run({"perf_json": "not json"}) is None
