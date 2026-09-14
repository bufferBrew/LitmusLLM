"""The metric catalogue.

The catalogue is data, not logic, but two things in it are load-bearing enough
to test: the score *direction* (which `scoring` relies on to avoid crowning the
most toxic model) and the *required fields* (which decide whether a metric is
skipped for a case or allowed to reach DeepEval and raise).
"""
from __future__ import annotations

import pytest

import metrics_catalog as mc


class TestCatalogueIntegrity:
    def test_every_key_is_unique(self):
        keys = [m.key for m in mc.METRICS]
        assert len(keys) == len(set(keys))

    def test_every_label_is_unique(self):
        # `scoring.goodness` looks a metric up by label, so a duplicate label
        # would silently resolve to the wrong direction.
        labels = [m.label for m in mc.METRICS]
        assert len(labels) == len(set(labels))

    def test_by_key_index_matches_the_tuple(self):
        assert set(mc.METRICS_BY_KEY) == {m.key for m in mc.METRICS}
        assert len(mc.METRICS_BY_KEY) == len(mc.METRICS)

    def test_every_metric_has_a_non_blank_label_and_description(self):
        for m in mc.METRICS:
            assert m.label.strip(), f"{m.key} has no label"
            assert m.description.strip(), f"{m.key} has no description"

    def test_every_threshold_is_within_the_score_range(self):
        for m in mc.METRICS:
            assert 0.0 <= m.threshold <= 1.0, f"{m.key} threshold out of range"

    def test_every_category_is_non_blank(self):
        for m in mc.METRICS:
            assert m.category.strip(), f"{m.key} has no category"

    def test_inverted_metrics_say_so_in_their_notes(self):
        # The direction is invisible in the UI unless the tooltip states it,
        # and a user reading "Toxicity 0.05" needs to know that is good.
        for m in mc.METRICS:
            if not m.higher_is_better:
                assert "lower is better" in m.notes.lower(), (
                    f"{m.key} is inverted but its notes do not say so"
                )

    def test_the_safety_metrics_are_the_inverted_ones(self):
        inverted = {m.key for m in mc.METRICS if not m.higher_is_better}
        assert inverted == {"toxicity", "bias", "hallucination"}

    def test_required_fields_are_drawn_from_the_known_test_case_fields(self):
        known = {"expected_output", "context", "retrieval_context",
                 "tools_called", "expected_tools"}
        for m in mc.METRICS:
            assert set(m.requires) <= known, f"{m.key} requires an unknown field"

    def test_default_selection_exists_in_the_catalogue(self):
        for key in mc.DEFAULT_METRIC_KEYS:
            assert key in mc.METRICS_BY_KEY

    def test_the_default_metric_needs_no_extra_dataset_columns(self):
        # The first-load default must work on a bare input/output dataset,
        # otherwise a new user's first run skips every case.
        for key in mc.DEFAULT_METRIC_KEYS:
            assert mc.get_metric(key).requires == ()


class TestRequirementLabel:
    def test_a_metric_needing_nothing_says_so(self):
        assert mc.get_metric("answer_relevancy").requirement_label == (
            "Works with input + output alone"
        )

    def test_a_single_requirement_is_named_in_plain_english(self):
        assert mc.get_metric("faithfulness").requirement_label == "Needs context"

    def test_multiple_requirements_are_joined(self):
        assert mc.get_metric("contextual_recall").requirement_label == (
            "Needs expected output + context"
        )

    def test_context_is_not_listed_twice_when_both_aliases_are_required(self):
        # 'context' and 'retrieval_context' both render as "context"; the
        # de-dupe stops the label reading "context + context".
        for m in mc.METRICS:
            label = m.requirement_label
            assert label.count("context") <= 1, f"{m.key}: {label}"

    def test_tool_metrics_name_the_trace_and_the_expectation(self):
        label = mc.get_metric("tool_correctness").requirement_label
        assert "tool-call trace" in label
        assert "expected tools" in label


class TestGetMetric:
    def test_returns_the_spec_for_a_known_key(self):
        assert mc.get_metric("bias").label == "Bias"

    def test_unknown_key_raises_with_the_known_keys_listed(self):
        with pytest.raises(KeyError) as excinfo:
            mc.get_metric("not_a_metric")
        message = str(excinfo.value)
        assert "not_a_metric" in message
        assert "answer_relevancy" in message


class TestResolveMetrics:
    def test_resolves_keys_to_specs(self):
        specs = mc.resolve_metrics(["bias", "toxicity"])
        assert {s.key for s in specs} == {"bias", "toxicity"}

    def test_preserves_catalogue_order_not_argument_order(self):
        # The UI shows metrics in catalogue order regardless of click order.
        forwards = mc.resolve_metrics(["answer_relevancy", "faithfulness"])
        backwards = mc.resolve_metrics(["faithfulness", "answer_relevancy"])
        assert [s.key for s in forwards] == [s.key for s in backwards]

    def test_duplicates_are_collapsed(self):
        specs = mc.resolve_metrics(["bias", "bias", "bias"])
        assert len(specs) == 1

    def test_unknown_keys_raise_rather_than_being_skipped(self):
        with pytest.raises(KeyError) as excinfo:
            mc.resolve_metrics(["bias", "nope"])
        assert "nope" in str(excinfo.value)

    def test_empty_selection_resolves_to_nothing(self):
        assert mc.resolve_metrics([]) == []


class TestCategories:
    def test_every_metric_appears_in_exactly_one_category(self):
        grouped = mc.categories()
        flattened = [m.key for specs in grouped.values() for m in specs]
        assert sorted(flattened) == sorted(m.key for m in mc.METRICS)

    def test_category_names_match_the_specs(self):
        for name, specs in mc.categories().items():
            for spec in specs:
                assert spec.category == name


class TestAsDicts:
    def test_one_entry_per_metric(self):
        assert len(mc.as_dicts()) == len(mc.METRICS)

    def test_entries_carry_the_fields_the_api_promises(self):
        expected = {"key", "label", "description", "threshold", "higher_is_better",
                    "requires", "requirement_label", "category", "advanced", "notes"}
        for entry in mc.as_dicts():
            assert set(entry) == expected

    def test_requires_is_a_list_so_it_survives_json(self):
        for entry in mc.as_dicts():
            assert isinstance(entry["requires"], list)

    def test_direction_survives_serialisation(self):
        by_key = {e["key"]: e for e in mc.as_dicts()}
        assert by_key["toxicity"]["higher_is_better"] is False
        assert by_key["answer_relevancy"]["higher_is_better"] is True


class TestMissingRequirements:
    def test_a_metric_needing_nothing_is_never_skipped(self):
        spec = mc.get_metric("answer_relevancy")
        assert mc.missing_requirements(spec, {}) == []

    def test_reports_a_missing_expected_output(self):
        spec = mc.get_metric("g_eval_correctness")
        assert mc.missing_requirements(spec, {"input": "q"}) == ["expected_output"]

    def test_satisfied_requirements_report_nothing(self):
        spec = mc.get_metric("g_eval_correctness")
        assert mc.missing_requirements(spec, {"expected_output": "a"}) == []

    def test_retrieval_context_is_satisfied_by_the_context_column(self):
        # Dataset rows only ever have a 'context' key; the metric's
        # 'retrieval_context' requirement must map onto it.
        spec = mc.get_metric("faithfulness")
        assert mc.missing_requirements(spec, {"context": ["chunk"]}) == []

    def test_an_empty_context_list_counts_as_missing(self):
        spec = mc.get_metric("faithfulness")
        assert mc.missing_requirements(spec, {"context": []}) == ["retrieval_context"]

    def test_an_empty_string_counts_as_missing(self):
        spec = mc.get_metric("g_eval_correctness")
        assert mc.missing_requirements(spec, {"expected_output": ""}) == ["expected_output"]

    def test_every_unmet_requirement_is_reported_at_once(self):
        spec = mc.get_metric("contextual_recall")
        missing = mc.missing_requirements(spec, {})
        assert set(missing) == {"expected_output", "retrieval_context"}

    def test_tool_metrics_report_each_missing_side(self):
        spec = mc.get_metric("tool_correctness")
        assert mc.missing_requirements(
            spec, {"tools_called": ["search"]}
        ) == ["expected_tools"]
