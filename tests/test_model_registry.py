"""Model identity.

`parse_model_id` is the one place that decides which runtime a prompt is sent
to, and it has to survive the fact that Ollama tags contain colons themselves.
Historical runs store their model id as text, so a change in how these parse
silently repoints an old run at a different model.
"""
from __future__ import annotations

import model_registry as registry
import runtimes


class TestLocalIds:
    def test_the_ollama_prefix_is_local_not_the_word_ollama(self):
        # 'local:' predates multi-runtime support and is kept so historical
        # runs still resolve. Renaming it would orphan every stored run.
        spec = registry.parse_model_id("local:llama3.1:latest")
        assert spec.kind == "local"
        assert spec.provider == "ollama"

    def test_only_the_first_colon_splits_the_prefix(self):
        # 'local:llama3.1:8b' must yield the tag 'llama3.1:8b', not 'llama3.1'.
        # Splitting on every colon would send prompts to a model that does not
        # exist.
        assert registry.parse_model_id("local:llama3.1:8b").name == "llama3.1:8b"

    def test_a_tag_with_several_colons_survives_intact(self):
        assert registry.parse_model_id("local:qwen2.5:14b:q4").name == "qwen2.5:14b:q4"

    def test_each_registered_runtime_prefix_resolves_to_its_runtime(self):
        for key, runtime in runtimes.RUNTIMES.items():
            spec = registry.parse_model_id(f"{runtime.prefix}:some-model")
            assert spec.kind == "local"
            assert spec.runtime == key, f"{runtime.prefix} resolved to {spec.runtime}"

    def test_the_id_is_echoed_back_unchanged_for_a_prefixed_local_model(self):
        assert registry.parse_model_id("lmstudio:foo").id == "lmstudio:foo"

    def test_local_models_never_require_an_api_key(self):
        for runtime in runtimes.RUNTIMES.values():
            spec = registry.parse_model_id(f"{runtime.prefix}:m")
            assert spec.requires_key is False
            assert spec.key_present is True

    def test_local_models_expose_an_openai_compatible_base_url(self):
        assert registry.parse_model_id("local:llama3.1").base_url is not None


class TestBareIds:
    def test_an_unprefixed_name_falls_back_to_the_default_runtime(self):
        # The judge form field accepts a bare tag; it must not be read as a
        # cloud model or rejected.
        spec = registry.parse_model_id("llama3.1:8b")
        assert spec.kind == "local"
        assert spec.name == "llama3.1:8b"
        assert spec.runtime == runtimes.RUNTIMES[registry.DEFAULT_RUNTIME].key

    def test_the_fallback_id_is_normalised_with_the_default_prefix(self):
        spec = registry.parse_model_id("llama3.1:8b")
        assert spec.id == "local:llama3.1:8b"

    def test_a_bare_name_round_trips_through_its_normalised_id(self):
        once = registry.parse_model_id("llama3.1:8b")
        twice = registry.parse_model_id(once.id)
        assert twice.name == once.name
        assert twice.runtime == once.runtime

    def test_an_unrecognised_prefix_is_treated_as_part_of_the_name(self):
        # 'mistral:7b' is an Ollama tag, not a 'mistral' runtime.
        spec = registry.parse_model_id("mistral:7b")
        assert spec.kind == "local"
        assert spec.name == "mistral:7b"


class TestCloudIds:
    def test_a_curated_model_gets_its_label_and_key_env(self):
        spec = registry.parse_model_id("cloud:anthropic/claude-opus-5")
        assert spec.kind == "cloud"
        assert spec.provider == "anthropic"
        assert spec.api_key_env == "ANTHROPIC_API_KEY"
        assert spec.label and spec.label != spec.name

    def test_an_uncurated_model_is_still_routable(self):
        # The curated list is a convenience, not an allowlist.
        spec = registry.parse_model_id("cloud:acme/some-model")
        assert spec.kind == "cloud"
        assert spec.name == "acme/some-model"

    def test_the_provider_is_inferred_from_the_litellm_prefix(self):
        spec = registry.parse_model_id("cloud:acme/some-model")
        assert spec.provider == "acme"
        assert spec.api_key_env == "ACME_API_KEY"

    def test_a_hyphenated_provider_becomes_an_underscored_env_var(self):
        spec = registry.parse_model_id("cloud:together-ai/llama")
        assert spec.api_key_env == "TOGETHER_AI_API_KEY"

    def test_a_bare_cloud_model_defaults_to_openai(self):
        spec = registry.parse_model_id("cloud:gpt-4o-mini")
        assert spec.provider == "openai"
        assert spec.api_key_env == "OPENAI_API_KEY"

    def test_cloud_models_route_through_litellm_not_a_base_url(self):
        assert registry.parse_model_id("cloud:anthropic/claude-opus-5").base_url is None

    def test_cloud_models_require_a_key(self):
        assert registry.parse_model_id("cloud:anthropic/claude-opus-5").requires_key is True

    def test_key_presence_reads_the_environment(self, monkeypatch):
        spec = registry.parse_model_id("cloud:anthropic/claude-opus-5")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert spec.key_present is False
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert spec.key_present is True

    def test_an_empty_key_counts_as_absent(self, monkeypatch):
        spec = registry.parse_model_id("cloud:anthropic/claude-opus-5")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        assert spec.key_present is False

    def test_a_bare_cloud_prefix_is_not_treated_as_a_cloud_model(self):
        # 'cloud:' with nothing after it has no model to route to.
        assert registry.parse_model_id("cloud:").kind == "local"


class TestFlagshipCards:
    def test_every_flagship_entry_has_the_fields_parse_relies_on(self):
        for entry in registry.FLAGSHIP_MODELS:
            for field in ("name", "label", "provider", "api_key_env"):
                assert entry.get(field), f"{entry.get('name')} is missing {field}"

    def test_flagship_names_are_unique(self):
        names = [m["name"] for m in registry.FLAGSHIP_MODELS]
        assert len(names) == len(set(names))

    def test_every_flagship_name_carries_a_litellm_provider_prefix(self):
        for entry in registry.FLAGSHIP_MODELS:
            assert "/" in entry["name"], f"{entry['name']} has no provider prefix"

    def test_the_declared_provider_matches_the_routing_prefix(self):
        for entry in registry.FLAGSHIP_MODELS:
            assert entry["name"].split("/")[0] == entry["provider"], entry["name"]

    def test_every_flagship_model_resolves_through_parse_model_id(self):
        for entry in registry.FLAGSHIP_MODELS:
            spec = registry.parse_model_id(f"cloud:{entry['name']}")
            assert spec.kind == "cloud"
            assert spec.label == entry["label"]
            assert spec.api_key_env == entry["api_key_env"]

    def test_cards_are_produced_for_every_flagship(self):
        assert len(registry.flagship_model_cards()) == len(registry.FLAGSHIP_MODELS)
