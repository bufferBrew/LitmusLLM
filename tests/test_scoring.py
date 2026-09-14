"""The Litmus Score.

This is the number the whole app exists to produce, so the tests here are
about the ways an aggregate can lie: a flipped metric ranking the worst model
best, an out-of-range judge score dragging the composite outside 0-100, a
flattering case count hiding that one metric saw almost no evidence, and two
runs configured differently being presented as rankable.
"""
from __future__ import annotations

import scoring


def row(metric_name: str, average_score: float | None, scored_cases: int = 20) -> dict:
    """One row in the shape `database.get_run_summary` returns."""
    return {
        "metric_name": metric_name,
        "average_score": average_score,
        "scored_cases": scored_cases,
    }


# ---------------------------------------------------------------------------
# goodness: direction handling
# ---------------------------------------------------------------------------

class TestGoodness:
    def test_higher_is_better_metrics_pass_through_unchanged(self):
        assert scoring.goodness("Answer Relevancy", 0.82) == 0.82

    def test_inverted_metrics_are_flipped_so_one_is_always_good(self):
        # Toxicity 0.0 is the *good* outcome, so it must become 1.0 before any
        # averaging. Without this the most toxic model wins the composite.
        assert scoring.goodness("Toxicity", 0.0) == 1.0
        assert scoring.goodness("Toxicity", 1.0) == 0.0
        assert scoring.goodness("Bias", 0.25) == 0.75
        assert scoring.goodness("Hallucination", 0.10) == 0.90

    def test_all_three_inverted_metrics_in_the_catalogue_are_flipped(self):
        import metrics_catalog

        inverted = [m.label for m in metrics_catalog.METRICS if not m.higher_is_better]
        assert inverted, "catalogue should carry at least one inverted metric"
        for label in inverted:
            assert scoring.goodness(label, 0.0) == 1.0, f"{label} was not flipped"

    def test_none_average_stays_none(self):
        assert scoring.goodness("Answer Relevancy", None) is None

    def test_out_of_range_judge_scores_are_clamped(self):
        # Judges occasionally return 1.2 or -0.1; neither may push a composite
        # past the ends of the 0-100 scale.
        assert scoring.goodness("Answer Relevancy", 1.4) == 1.0
        assert scoring.goodness("Answer Relevancy", -0.3) == 0.0
        assert scoring.goodness("Toxicity", -0.2) == 1.0
        assert scoring.goodness("Toxicity", 1.5) == 0.0

    def test_unknown_metric_is_treated_as_higher_is_better(self):
        # A metric not in the catalogue must still score rather than crash the
        # run; assuming the common direction is the safe default.
        assert scoring.goodness("Some Custom Metric", 0.6) == 0.6


# ---------------------------------------------------------------------------
# composite: the aggregate itself
# ---------------------------------------------------------------------------

class TestComposite:
    def test_single_metric_scales_to_0_100(self):
        card = scoring.composite([row("Answer Relevancy", 0.82)])
        assert card is not None
        assert card.score == 82.0

    def test_unweighted_mean_across_metrics(self):
        card = scoring.composite([
            row("Answer Relevancy", 0.90),
            row("Faithfulness", 0.70),
        ])
        assert card.score == 80.0

    def test_inverted_metric_is_flipped_before_averaging(self):
        # Raw mean would be (0.9 + 0.1) / 2 = 0.5 -> 50.0, which would rank a
        # clean model the same as a toxic one. Flipped: (0.9 + 0.9) / 2 = 90.0.
        card = scoring.composite([
            row("Answer Relevancy", 0.90),
            row("Toxicity", 0.10),
        ])
        assert card.score == 90.0

    def test_a_toxic_model_scores_below_a_clean_one(self):
        clean = scoring.composite([row("Answer Relevancy", 0.8), row("Toxicity", 0.05)])
        toxic = scoring.composite([row("Answer Relevancy", 0.8), row("Toxicity", 0.95)])
        assert clean.score > toxic.score

    def test_score_is_rounded_to_one_decimal(self):
        card = scoring.composite([
            row("Answer Relevancy", 0.8123),
            row("Faithfulness", 0.7456),
        ])
        assert card.score == round(card.score, 1)
        assert card.score == 77.9

    def test_metrics_with_no_scored_cases_are_excluded(self):
        card = scoring.composite([
            row("Answer Relevancy", 0.80, scored_cases=20),
            row("Faithfulness", 0.20, scored_cases=0),
        ])
        assert card.metric_count == 1
        assert card.score == 80.0

    def test_metrics_with_a_none_average_are_excluded(self):
        card = scoring.composite([
            row("Answer Relevancy", 0.80),
            row("Faithfulness", None),
        ])
        assert card.metric_count == 1
        assert card.score == 80.0

    def test_returns_none_when_nothing_scored(self):
        assert scoring.composite([]) is None
        assert scoring.composite(None) is None
        assert scoring.composite([row("Answer Relevancy", None)]) is None
        assert scoring.composite([row("Answer Relevancy", 0.9, scored_cases=0)]) is None

    def test_cases_reports_the_weakest_link_not_the_maximum(self):
        # A composite is only as well-evidenced as its thinnest metric.
        card = scoring.composite([
            row("Answer Relevancy", 0.80, scored_cases=200),
            row("Faithfulness", 0.70, scored_cases=3),
        ])
        assert card.cases == 3

    def test_contribution_records_both_raw_and_flipped_values(self):
        card = scoring.composite([row("Toxicity", 0.2, scored_cases=12)])
        contribution = card.contributions[0]
        assert contribution.label == "Toxicity"
        assert contribution.raw == 0.2
        assert contribution.good == 0.8
        assert contribution.inverted is True
        assert contribution.cases == 12

    def test_contribution_for_a_normal_metric_is_not_marked_inverted(self):
        card = scoring.composite([row("Answer Relevancy", 0.5)])
        assert card.contributions[0].inverted is False

    def test_clamping_keeps_the_composite_inside_the_scale(self):
        card = scoring.composite([
            row("Answer Relevancy", 1.8),
            row("Faithfulness", 1.6),
        ])
        assert card.score == 100.0


# ---------------------------------------------------------------------------
# thin evidence: refusing to let a small run look authoritative
# ---------------------------------------------------------------------------

class TestThinEvidence:
    def test_a_well_evidenced_composite_is_not_flagged(self):
        card = scoring.composite([
            row("Answer Relevancy", 0.8, scored_cases=50),
            row("Faithfulness", 0.7, scored_cases=50),
        ])
        assert card.thin is False
        assert card.caveat == ""

    def test_a_single_metric_composite_is_flagged(self):
        card = scoring.composite([row("Answer Relevancy", 0.8, scored_cases=50)])
        assert card.thin is True
        assert "only one metric" in card.caveat

    def test_too_few_cases_is_flagged(self):
        card = scoring.composite([
            row("Answer Relevancy", 0.8, scored_cases=5),
            row("Faithfulness", 0.7, scored_cases=5),
        ])
        assert card.thin is True
        assert "only 5 test cases" in card.caveat

    def test_the_case_threshold_is_inclusive_at_the_boundary(self):
        just_under = scoring.composite([
            row("Answer Relevancy", 0.8, scored_cases=scoring.THIN_EVIDENCE_CASES - 1),
            row("Faithfulness", 0.7, scored_cases=scoring.THIN_EVIDENCE_CASES - 1),
        ])
        at_threshold = scoring.composite([
            row("Answer Relevancy", 0.8, scored_cases=scoring.THIN_EVIDENCE_CASES),
            row("Faithfulness", 0.7, scored_cases=scoring.THIN_EVIDENCE_CASES),
        ])
        assert just_under.thin is True
        assert at_threshold.thin is False

    def test_both_reasons_are_reported_together(self):
        card = scoring.composite([row("Answer Relevancy", 0.8, scored_cases=4)])
        assert card.caveat == "only one metric and only 4 test cases"

    def test_a_single_case_is_described_in_the_singular(self):
        card = scoring.composite([
            row("Answer Relevancy", 0.8, scored_cases=1),
            row("Faithfulness", 0.7, scored_cases=1),
        ])
        assert "only 1 test case" in card.caveat
        assert "test cases" not in card.caveat

    def test_one_thin_metric_flags_the_whole_composite(self):
        # 200 cases on one metric does not rescue a composite whose other
        # metric saw two.
        card = scoring.composite([
            row("Answer Relevancy", 0.8, scored_cases=200),
            row("Faithfulness", 0.7, scored_cases=2),
        ])
        assert card.thin is True


class TestBands:
    def test_band_boundaries(self):
        def band_for(score: float) -> str:
            return scoring.composite([
                row("Answer Relevancy", score),
                row("Faithfulness", score),
            ]).band

        assert band_for(0.95) == "strong"
        assert band_for(0.80) == "strong"
        assert band_for(0.79) == "fair"
        assert band_for(0.60) == "fair"
        assert band_for(0.59) == "weak"
        assert band_for(0.40) == "weak"
        assert band_for(0.39) == "poor"
        assert band_for(0.0) == "poor"


# ---------------------------------------------------------------------------
# comparability: which scores may be ranked against each other
# ---------------------------------------------------------------------------

class TestComparabilityKey:
    def test_identical_setups_produce_the_same_tag(self):
        a = scoring.comparability_key(1, ["answer_relevancy", "faithfulness"], "llama3.1")
        b = scoring.comparability_key(1, ["answer_relevancy", "faithfulness"], "llama3.1")
        assert a == b

    def test_metric_order_does_not_change_the_tag(self):
        # Order is a UI artefact, not part of the evaluation setup.
        a = scoring.comparability_key(1, ["answer_relevancy", "faithfulness"], "llama3.1")
        b = scoring.comparability_key(1, ["faithfulness", "answer_relevancy"], "llama3.1")
        assert a == b

    def test_a_different_dataset_changes_the_tag(self):
        a = scoring.comparability_key(1, ["answer_relevancy"], "llama3.1")
        b = scoring.comparability_key(2, ["answer_relevancy"], "llama3.1")
        assert a != b

    def test_a_different_judge_changes_the_tag(self):
        a = scoring.comparability_key(1, ["answer_relevancy"], "llama3.1")
        b = scoring.comparability_key(1, ["answer_relevancy"], "qwen2.5")
        assert a != b

    def test_an_added_metric_changes_the_tag(self):
        a = scoring.comparability_key(1, ["answer_relevancy"], "llama3.1")
        b = scoring.comparability_key(1, ["answer_relevancy", "bias"], "llama3.1")
        assert a != b

    def test_metrics_stored_as_a_json_string_match_the_list_form(self):
        # The database column holds JSON text; callers pass whichever they have.
        as_list = scoring.comparability_key(1, ["answer_relevancy", "bias"], "llama3.1")
        as_json = scoring.comparability_key(1, '["answer_relevancy", "bias"]', "llama3.1")
        assert as_list == as_json

    def test_an_unparseable_metric_string_is_treated_as_one_metric(self):
        key = scoring.comparability_key(1, "answer_relevancy", "llama3.1")
        assert key == scoring.comparability_key(1, ["answer_relevancy"], "llama3.1")

    def test_missing_metrics_and_judge_still_produce_a_tag(self):
        assert len(scoring.comparability_key(1, None, None)) == 6

    def test_tag_is_six_hex_characters(self):
        key = scoring.comparability_key(1, ["answer_relevancy"], "llama3.1")
        assert len(key) == 6
        assert all(c in "0123456789abcdef" for c in key)


class TestComparabilityNote:
    def test_note_names_the_dataset_judge_and_metric_count(self):
        note = scoring.comparability_note({
            "dataset_name": "Support tickets",
            "metrics": ["answer_relevancy", "faithfulness"],
            "judge_model": "llama3.1:latest",
        })
        assert "Support tickets" in note
        assert "2 metric" in note
        assert "llama3.1:latest" in note

    def test_note_handles_metrics_stored_as_json_text(self):
        note = scoring.comparability_note({
            "dataset_name": "Support tickets",
            "metrics": '["answer_relevancy", "faithfulness", "bias"]',
            "judge_model": "llama3.1",
        })
        assert "3 metric" in note

    def test_note_falls_back_to_generic_wording_when_fields_are_missing(self):
        note = scoring.comparability_note({})
        assert "the same dataset" in note
        assert "the same judge" in note
