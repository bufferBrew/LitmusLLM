"""Published reference data.

These are numbers somebody else measured. The module's whole job is keeping
them separate from LitmusLLM's own scores and attached to their provenance, so
the tests here are mostly integrity checks on the seed table plus the family
hint that decides which published rows are shown next to a local model.
"""
from __future__ import annotations

import benchmarks


class TestFamilyForLocalModel:
    def test_matches_an_ollama_tag_to_its_published_family(self):
        assert benchmarks.family_for_local_model("qwen2.5:14b") == "Qwen"
        assert benchmarks.family_for_local_model("llama3.1:8b") == "Llama"
        assert benchmarks.family_for_local_model("gemma2:2b") == "Gemma"

    def test_the_tag_suffix_is_ignored(self):
        assert benchmarks.family_for_local_model("llama3.1:latest") == "Llama"
        assert benchmarks.family_for_local_model("llama3.1") == "Llama"

    def test_matching_is_case_insensitive(self):
        assert benchmarks.family_for_local_model("Llama3.1:8B") == "Llama"
        assert benchmarks.family_for_local_model("QWEN2.5:14b") == "Qwen"

    def test_mixtral_maps_to_the_mistral_family(self):
        assert benchmarks.family_for_local_model("mixtral:8x7b") == "Mistral"

    def test_an_unknown_family_returns_nothing_rather_than_guessing(self):
        # A blank is honest; pointing a user at an unrelated vendor's rows is
        # worse than showing none.
        assert benchmarks.family_for_local_model("solar:10.7b") is None
        assert benchmarks.family_for_local_model("") is None

    def test_a_family_name_embedded_mid_tag_does_not_match(self):
        # The hint is a prefix rule; 'my-llama-fork' is not a Llama release.
        assert benchmarks.family_for_local_model("my-llama-fork:7b") is None

    def test_every_hint_target_exists_in_the_seed_table(self):
        families = {row["family"] for row in benchmarks.SEED_ROWS}
        for prefix, family in benchmarks.FAMILY_HINTS.items():
            assert family in families, (
                f"hint '{prefix}' points at '{family}', which has no seed rows"
            )


class TestSeedRows:
    def test_the_seed_table_is_not_empty(self):
        assert len(benchmarks.SEED_ROWS) > 0

    def test_every_row_carries_the_fields_the_panel_renders(self):
        required = {"model_label", "vendor", "family", "kind", "score"}
        for row in benchmarks.SEED_ROWS:
            assert required <= set(row), f"{row.get('model_label')} is missing fields"

    def test_every_score_is_on_the_zero_to_one_hundred_index_scale(self):
        # The index is a percentage; a 0-1 value here would render as a model
        # scoring 0.6 against peers scoring 60.
        for row in benchmarks.SEED_ROWS:
            assert 0.0 <= row["score"] <= 100.0, row["model_label"]

    def test_every_model_label_and_vendor_is_non_blank(self):
        for row in benchmarks.SEED_ROWS:
            assert row["model_label"].strip()
            assert row["vendor"].strip()

    def test_kind_is_one_of_the_two_buckets_the_ui_groups_by(self):
        for row in benchmarks.SEED_ROWS:
            assert row["kind"] in {"frontier", "open_weight"}, row["model_label"]

    def test_a_model_and_variant_pair_appears_only_once(self):
        # Reasoning-effort variants are kept separate on purpose, but the same
        # pair twice would be a duplicate row, not a variant.
        pairs = [(r["model_label"], r.get("variant", "")) for r in benchmarks.SEED_ROWS]
        assert len(pairs) == len(set(pairs))

    def test_every_family_is_reachable_from_some_vendor(self):
        for row in benchmarks.SEED_ROWS:
            assert row["family"].strip(), row["model_label"]


class TestProvenance:
    def test_the_source_is_named_and_linked(self):
        assert benchmarks.SEED_SOURCE.strip()
        assert benchmarks.SEED_SOURCE_URL.startswith("https://")

    def test_the_retrieval_date_is_an_iso_date(self):
        from datetime import date

        date.fromisoformat(benchmarks.SEED_RETRIEVED)

    def test_the_index_note_states_the_scale_and_direction(self):
        # The number is meaningless to a reader who does not know it is an
        # aggregate on a 0-100 scale where higher is better.
        note = benchmarks.SEED_INDEX_NOTE.lower()
        assert "0-100" in note
        assert "higher is better" in note
