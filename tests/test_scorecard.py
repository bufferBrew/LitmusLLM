"""The comparison table.

The scorecard puts quality, speed, cost and serving details side by side so a
model can be chosen under a real constraint. Its risk is presentational: rows
from incompatible evaluation setups sit adjacent and look directly rankable,
and a benchmark percentage shown without its interval reads as precision it
does not have. Both guards are tested here.
"""
from __future__ import annotations

import perf
import scorecard
import scoring


def make_row(
    run_id: int = 1,
    *,
    setup_tag: str = "abc123",
    composite: scoring.Composite | None = None,
    measured: perf.RunPerf | None = None,
    runtime: str | None = "Ollama",
    quantization: str | None = "Q4_K_M",
    vram_bytes: int | None = None,
    benchmarks: dict | None = None,
) -> scorecard.Row:
    row = scorecard.Row(
        run_id=run_id,
        model="local:llama3.1:8b",
        model_type="local",
        runtime=runtime,
        quantization=quantization,
        vram_bytes=vram_bytes,
        dataset="Starter set",
        judge="llama3.1:latest",
        status="completed",
        started_at="2026-09-01T10:00:00",
        setup_tag=setup_tag,
        composite=composite,
        measured=measured,
    )
    if benchmarks:
        row.benchmarks = benchmarks
    return row


def composite_of(*averages: float) -> scoring.Composite:
    labels = ["Answer Relevancy", "Faithfulness", "Contextual Recall"]
    return scoring.composite([
        {"metric_name": labels[i], "average_score": avg, "scored_cases": 50}
        for i, avg in enumerate(averages)
    ])


class TestScoreColumn:
    def test_a_scored_run_reports_its_composite(self):
        row = make_row(composite=composite_of(0.8, 0.8))
        assert row.score == 80.0

    def test_an_unscored_run_reports_no_score_rather_than_zero(self):
        # A failed or still-running row must not sort as the worst model.
        assert make_row(composite=None).score is None

    def test_thin_evidence_travels_onto_the_row(self):
        assert make_row(composite=composite_of(0.8)).thin is True
        assert make_row(composite=composite_of(0.8, 0.7)).thin is False

    def test_an_unscored_row_is_not_flagged_as_thin(self):
        assert make_row(composite=None).thin is False


class TestServingColumns:
    def test_vram_is_shown_in_gigabytes(self):
        assert make_row(vram_bytes=5_400_000_000).vram_label == "5.40 GB"

    def test_unreported_vram_is_a_dash_not_zero(self):
        # Only Ollama reports residency; a blank means "not reported", and
        # "0.00 GB" would read as a model using no memory.
        assert make_row(vram_bytes=None).vram_label == "--"
        assert make_row(vram_bytes=0).vram_label == "--"

    def test_the_serving_column_joins_runtime_and_quantization(self):
        assert make_row(runtime="Ollama", quantization="Q4_K_M").served_label == (
            "Ollama · Q4_K_M"
        )

    def test_a_partially_known_serving_setup_shows_what_is_known(self):
        assert make_row(runtime="Ollama", quantization=None).served_label == "Ollama"
        assert make_row(runtime=None, quantization="Q8_0").served_label == "Q8_0"

    def test_an_unknown_serving_setup_is_a_dash(self):
        assert make_row(runtime=None, quantization=None).served_label == "--"


class TestSpeedAndCostColumns:
    def test_an_uninstrumented_run_dashes_every_measured_column(self):
        row = make_row(measured=None)
        assert row.tps_label == "--"
        assert row.ttft_label == "--"
        assert row.cost_label == "--"

    def test_measured_columns_render_from_the_stored_distributions(self):
        row = make_row(measured=perf.RunPerf(
            calls=10,
            ttft={"p50": 320.0},
            output_tps={"p50": 42.5},
            total_cost_usd=0.0,
        ))
        assert row.ttft_label == "320 ms"
        assert row.tps_label == "42.5 tok/s"
        assert row.cost_label == "no API cost"


class TestBenchmarkColumn:
    def test_a_benchmark_score_carries_its_interval(self):
        # A bare '62.5%' from a 50-question run reads as precision it does not
        # have, so the interval travels with the number everywhere.
        row = make_row(benchmarks={"gsm8k": {"score": 0.625, "stderr": 0.04, "samples": 50}})
        assert row.benchmark_label("gsm8k") == "62.5% ± 7.8"

    def test_a_score_without_a_stderr_is_shown_bare(self):
        row = make_row(benchmarks={"gsm8k": {"score": 0.625, "stderr": None, "samples": 50}})
        assert row.benchmark_label("gsm8k") == "62.5%"

    def test_an_unrun_benchmark_is_a_dash(self):
        assert make_row().benchmark_label("gsm8k") == "--"

    def test_a_row_present_but_unscored_is_a_dash(self):
        row = make_row(benchmarks={"gsm8k": {"score": None, "samples": 0}})
        assert row.benchmark_label("gsm8k") == "--"

    def test_sample_count_is_available_alongside_the_score(self):
        row = make_row(benchmarks={"gsm8k": {"score": 0.6, "stderr": 0.04, "samples": 50}})
        assert row.benchmark_samples("gsm8k") == 50

    def test_sample_count_for_an_unrun_benchmark_is_none(self):
        assert make_row().benchmark_samples("gsm8k") is None


class TestSetupGroups:
    def test_rows_are_bucketed_by_comparability_tag(self):
        card = scorecard.Scorecard(rows=[
            make_row(1, setup_tag="aaa111"),
            make_row(2, setup_tag="aaa111"),
            make_row(3, setup_tag="bbb222"),
        ])
        groups = card.setup_groups
        assert set(groups) == {"aaa111", "bbb222"}
        assert len(groups["aaa111"]) == 2
        assert len(groups["bbb222"]) == 1

    def test_only_groups_with_more_than_one_row_are_comparable(self):
        # A single observation in its own setup is not a comparison, however
        # adjacent it sits in the table.
        card = scorecard.Scorecard(rows=[
            make_row(1, setup_tag="aaa111"),
            make_row(2, setup_tag="aaa111"),
            make_row(3, setup_tag="bbb222"),
        ])
        assert card.comparable_groups == 1

    def test_a_table_of_all_distinct_setups_has_nothing_comparable(self):
        card = scorecard.Scorecard(rows=[
            make_row(1, setup_tag="aaa111"),
            make_row(2, setup_tag="bbb222"),
        ])
        assert card.comparable_groups == 0

    def test_an_empty_table_has_no_groups(self):
        card = scorecard.Scorecard()
        assert card.setup_groups == {}
        assert card.comparable_groups == 0

    def test_grouping_preserves_row_order_within_a_group(self):
        rows = [make_row(1, setup_tag="a"), make_row(2, setup_tag="b"),
                make_row(3, setup_tag="a")]
        assert [r.run_id for r in scorecard.Scorecard(rows=rows).setup_groups["a"]] == [1, 3]


class TestAnyMeasured:
    def test_a_table_of_uninstrumented_runs_reports_nothing_measured(self):
        card = scorecard.Scorecard(rows=[make_row(1), make_row(2)])
        assert card.any_measured is False

    def test_one_instrumented_run_is_enough_to_show_the_speed_columns(self):
        card = scorecard.Scorecard(rows=[
            make_row(1),
            make_row(2, measured=perf.RunPerf(calls=5, output_tps={"p50": 30.0})),
        ])
        assert card.any_measured is True

    def test_a_run_with_a_perf_record_but_no_calls_does_not_count(self):
        card = scorecard.Scorecard(rows=[make_row(1, measured=perf.RunPerf())])
        assert card.any_measured is False


class TestCsvExport:
    def test_the_header_row_is_written(self):
        text = scorecard.to_csv(scorecard.Scorecard())
        assert text.splitlines()[0].startswith(scorecard.HEADERS[0])

    def test_each_row_becomes_a_line(self):
        card = scorecard.Scorecard(
            rows=[make_row(1, composite=composite_of(0.8, 0.8)), make_row(2)],
            metric_labels=["Answer Relevancy", "Faithfulness"],
        )
        lines = [line for line in scorecard.to_csv(card).splitlines() if line.strip()]
        assert len(lines) == 3  # header + two runs

    def test_the_comparability_tag_is_exported_alongside_the_scores(self):
        card = scorecard.Scorecard(rows=[make_row(1, setup_tag="zz9900")])
        assert "zz9900" in scorecard.to_csv(card)
