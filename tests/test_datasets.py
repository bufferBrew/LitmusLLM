"""Dataset ingestion.

Everything a run scores enters through here, so a parsing bug does not show up
as an error — it shows up as a plausible-looking score computed over the wrong
rows. These tests cover the ways a real CSV arrives malformed: a BOM from
Excel, Latin-1 bytes, trailing blank lines, a missing header, and columns from
some other tool's export.
"""
from __future__ import annotations

import pytest

import datasets


class TestSplitMulti:
    def test_empty_input_yields_no_values(self):
        assert datasets.split_multi("") == []
        assert datasets.split_multi(None) == []

    def test_a_single_value_stays_single(self):
        assert datasets.split_multi("one chunk") == ["one chunk"]

    def test_splits_on_pipes(self):
        assert datasets.split_multi("a|b|c") == ["a", "b", "c"]

    def test_surrounding_whitespace_is_stripped(self):
        assert datasets.split_multi(" a | b |c ") == ["a", "b", "c"]

    def test_splits_on_newlines_when_no_pipe_is_present(self):
        assert datasets.split_multi("a\nb\nc") == ["a", "b", "c"]

    def test_windows_line_endings_are_normalised(self):
        assert datasets.split_multi("a\r\nb") == ["a", "b"]

    def test_pipes_win_over_newlines(self):
        # A chunk may legitimately contain a line break, so once a pipe is
        # present it is the delimiter and newlines stay inside the value.
        assert datasets.split_multi("a|b\nc") == ["a", "b\nc"]

    def test_empty_segments_are_dropped(self):
        assert datasets.split_multi("a||b|") == ["a", "b"]
        assert datasets.split_multi("a\n\nb\n") == ["a", "b"]

    def test_a_cell_of_only_delimiters_yields_nothing(self):
        assert datasets.split_multi("|||") == []
        assert datasets.split_multi("   ") == []


class TestParseCsv:
    def test_parses_a_minimal_input_only_file(self):
        rows = datasets.parse_csv(b"input\nWhat is 2+2?\n")
        assert len(rows) == 1
        assert rows[0]["input"] == "What is 2+2?"

    def test_absent_optional_fields_are_normalised_not_omitted(self):
        # Every row carries the full key set so downstream code never has to
        # branch on whether a column was present in the upload.
        row = datasets.parse_csv(b"input\nq\n")[0]
        assert row["expected_output"] is None
        assert row["context"] == []
        assert row["tools_called"] == []
        assert row["expected_tools"] == []

    def test_parses_every_supported_column(self):
        csv_bytes = (
            b"input,expected_output,context,tools_called,expected_tools\n"
            b"q,a,c1 | c2,search,search | fetch\n"
        )
        row = datasets.parse_csv(csv_bytes)[0]
        assert row["input"] == "q"
        assert row["expected_output"] == "a"
        assert row["context"] == ["c1", "c2"]
        assert row["tools_called"] == ["search"]
        assert row["expected_tools"] == ["search", "fetch"]

    def test_header_matching_is_case_and_whitespace_insensitive(self):
        row = datasets.parse_csv(b" Input , Expected_Output \nq,a\n")[0]
        assert row["input"] == "q"
        assert row["expected_output"] == "a"

    def test_unknown_columns_are_ignored_rather_than_rejected(self):
        # Exports from other eval tools carry extra columns; rejecting them
        # would mean editing every file before it loads.
        row = datasets.parse_csv(b"input,run_id,notes\nq,42,whatever\n")[0]
        assert row["input"] == "q"
        assert "run_id" not in row

    def test_a_utf8_bom_from_excel_is_stripped(self):
        rows = datasets.parse_csv("input\nq\n".encode("utf-8-sig"))
        assert rows[0]["input"] == "q"

    def test_latin1_bytes_are_decoded_rather_than_rejected(self):
        rows = datasets.parse_csv("input\ncafé\n".encode("latin-1"))
        assert len(rows) == 1
        assert "caf" in rows[0]["input"]

    def test_non_ascii_utf8_survives_round_trip(self):
        rows = datasets.parse_csv("input\nWas kostet das Büro?\n".encode("utf-8"))
        assert rows[0]["input"] == "Was kostet das Büro?"

    def test_blank_rows_are_skipped(self):
        rows = datasets.parse_csv(b"input\nq1\n\nq2\n\n")
        assert [r["input"] for r in rows] == ["q1", "q2"]

    def test_a_row_whose_input_is_only_whitespace_is_skipped(self):
        rows = datasets.parse_csv(b"input\nq1\n   \nq2\n")
        assert [r["input"] for r in rows] == ["q1", "q2"]

    def test_cell_values_are_stripped(self):
        row = datasets.parse_csv(b"input,expected_output\n  q  ,  a  \n")[0]
        assert row["input"] == "q"
        assert row["expected_output"] == "a"

    def test_an_empty_expected_output_becomes_none_not_empty_string(self):
        # Metrics test `requires` with truthiness; an empty string would pass
        # a check it should fail.
        row = datasets.parse_csv(b"input,expected_output\nq,\n")[0]
        assert row["expected_output"] is None

    def test_quoted_commas_stay_inside_the_cell(self):
        row = datasets.parse_csv(b'input\n"Hello, world, again"\n')[0]
        assert row["input"] == "Hello, world, again"

    def test_max_rows_caps_the_import(self):
        body = b"input\n" + b"".join(b"q%d\n" % i for i in range(20))
        rows = datasets.parse_csv(body, max_rows=5)
        assert len(rows) == 5

    def test_context_split_on_newlines_within_a_quoted_cell(self):
        row = datasets.parse_csv(b'input,context\nq,"chunk one\nchunk two"\n')[0]
        assert row["context"] == ["chunk one", "chunk two"]


class TestParseCsvErrors:
    def test_an_empty_file_is_rejected(self):
        with pytest.raises(datasets.CSVFormatError) as excinfo:
            datasets.parse_csv(b"")
        assert "empty" in str(excinfo.value).lower()

    def test_a_missing_input_column_is_rejected_and_names_what_was_found(self):
        with pytest.raises(datasets.CSVFormatError) as excinfo:
            datasets.parse_csv(b"question,answer\nq,a\n")
        message = str(excinfo.value)
        assert "'input' column is missing" in message
        assert "question" in message  # tells the user what it did see

    def test_a_header_with_no_data_rows_is_rejected(self):
        with pytest.raises(datasets.CSVFormatError) as excinfo:
            datasets.parse_csv(b"input,expected_output\n")
        assert "No usable rows" in str(excinfo.value)

    def test_a_file_of_only_blank_inputs_is_rejected(self):
        with pytest.raises(datasets.CSVFormatError):
            datasets.parse_csv(b"input\n\n\n\n")

    def test_the_error_is_a_valueerror_so_callers_can_catch_broadly(self):
        assert issubclass(datasets.CSVFormatError, ValueError)


class TestRowsToCsv:
    def test_writes_the_documented_header(self):
        text = datasets.rows_to_csv([])
        assert text.splitlines()[0] == ",".join(datasets.CSV_COLUMNS)

    def test_a_parsed_file_survives_a_round_trip(self):
        original = (
            b"input,expected_output,context,tools_called,expected_tools\n"
            b"q,a,c1 | c2,search,search | fetch\n"
        )
        once = datasets.parse_csv(original)
        twice = datasets.parse_csv(datasets.rows_to_csv(once).encode("utf-8"))
        assert once == twice

    def test_multi_values_are_rejoined_with_pipes(self):
        text = datasets.rows_to_csv([{"input": "q", "context": ["a", "b"]}])
        assert "a | b" in text

    def test_missing_optional_fields_become_empty_cells_not_the_word_none(self):
        text = datasets.rows_to_csv([{"input": "q", "expected_output": None}])
        assert "None" not in text

    def test_a_comma_bearing_value_is_quoted_so_it_reparses(self):
        rows = [{"input": "Hello, world", "context": [], "tools_called": [],
                 "expected_tools": [], "expected_output": None}]
        reparsed = datasets.parse_csv(datasets.rows_to_csv(rows).encode("utf-8"))
        assert reparsed[0]["input"] == "Hello, world"


class TestDatasetCapabilities:
    def test_counts_an_empty_dataset_as_all_zeroes(self):
        assert datasets.dataset_capabilities([]) == {
            "total": 0, "with_expected_output": 0, "with_context": 0, "with_tools": 0,
        }

    def test_counts_each_optional_field_independently(self):
        rows = [
            {"input": "a", "expected_output": "x", "context": ["c"]},
            {"input": "b", "expected_output": "y"},
            {"input": "c"},
        ]
        caps = datasets.dataset_capabilities(rows)
        assert caps["total"] == 3
        assert caps["with_expected_output"] == 2
        assert caps["with_context"] == 1

    def test_empty_values_are_not_counted_as_populated(self):
        rows = [{"input": "a", "expected_output": "", "context": []}]
        caps = datasets.dataset_capabilities(rows)
        assert caps["with_expected_output"] == 0
        assert caps["with_context"] == 0

    def test_tools_require_both_sides_to_count(self):
        # Tool Call Accuracy compares called against expected; one side alone
        # cannot be scored, so it must not be advertised as available.
        only_called = [{"input": "a", "tools_called": ["search"]}]
        only_expected = [{"input": "a", "expected_tools": ["search"]}]
        both = [{"input": "a", "tools_called": ["search"], "expected_tools": ["search"]}]
        assert datasets.dataset_capabilities(only_called)["with_tools"] == 0
        assert datasets.dataset_capabilities(only_expected)["with_tools"] == 0
        assert datasets.dataset_capabilities(both)["with_tools"] == 1


class TestBuiltinDataset:
    def test_the_starter_set_is_not_empty(self):
        assert len(datasets.BUILTIN_ROWS) >= 10

    def test_every_starter_row_has_a_non_blank_input(self):
        for i, row in enumerate(datasets.BUILTIN_ROWS):
            assert row.get("input", "").strip(), f"row {i} has no input"

    def test_every_starter_row_has_an_expected_output(self):
        # Without one, the correctness and recall metrics skip the whole set.
        for i, row in enumerate(datasets.BUILTIN_ROWS):
            assert row.get("expected_output", "").strip(), f"row {i} has no expected output"

    def test_context_is_always_a_list_never_a_bare_string(self):
        # RAG metrics score chunks individually; a bare string would be scored
        # one character at a time.
        for i, row in enumerate(datasets.BUILTIN_ROWS):
            assert isinstance(row.get("context", []), list), f"row {i} context is not a list"

    def test_the_starter_set_covers_the_context_dependent_metrics(self):
        caps = datasets.dataset_capabilities(datasets.BUILTIN_ROWS)
        assert caps["with_context"] >= 5, "too few grounded cases for the RAG metrics"

    def test_inputs_are_unique(self):
        inputs = [r["input"] for r in datasets.BUILTIN_ROWS]
        assert len(inputs) == len(set(inputs))

    def test_the_starter_set_clears_the_thin_evidence_threshold(self):
        # A first-time user's first run should not be flagged as too small.
        import scoring

        assert len(datasets.BUILTIN_ROWS) >= scoring.THIN_EVIDENCE_CASES

    def test_the_starter_set_survives_a_csv_round_trip(self):
        text = datasets.rows_to_csv(datasets.BUILTIN_ROWS)
        reparsed = datasets.parse_csv(text.encode("utf-8"))
        assert len(reparsed) == len(datasets.BUILTIN_ROWS)
        assert reparsed[0]["input"] == datasets.BUILTIN_ROWS[0]["input"]
        assert reparsed[0]["context"] == datasets.BUILTIN_ROWS[0]["context"]
