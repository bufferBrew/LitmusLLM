"""Built-in test data and CSV ingestion.

A dataset row is a plain dict with these keys:

    input            str          the prompt / question                (required)
    expected_output  str | None   reference answer                      (optional)
    context          list[str]    ground-truth or retrieved chunks      (optional)
    tools_called     list[str]    tool names the model actually called   (optional)
    expected_tools   list[str]    tool names it should have called       (optional)

`context` is a list because RAG metrics score chunks individually. In a CSV
it's a single cell, so multiple chunks are split on `|` or on newlines.
"""
from __future__ import annotations

import csv
import io
from typing import Any

import database

BUILTIN_DATASET_NAME = "LitmusLLM starter set"

# 15 cases spanning the shapes the metrics need: plain Q&A (relevancy,
# toxicity, bias), context-grounded Q&A (faithfulness, hallucination,
# contextual recall/precision) and a couple of summarisation prompts.
BUILTIN_ROWS: list[dict[str, Any]] = [
    {
        "input": "What is the capital of France, and roughly how many people live there?",
        "expected_output": "Paris is the capital of France. Its city population is about 2.1 million, with roughly 11 million in the wider metropolitan area.",
        "context": [
            "Paris is the capital and most populous city of France.",
            "The city proper has a population of about 2.1 million, while the Paris metropolitan area has approximately 11 million inhabitants.",
        ],
    },
    {
        "input": "Explain the difference between a list and a tuple in Python.",
        "expected_output": "Lists are mutable and written with square brackets; tuples are immutable and written with parentheses. Because tuples are immutable they can be dictionary keys and are slightly faster.",
        "context": [
            "Python lists are mutable sequences, created with square brackets, and support in-place methods such as append and pop.",
            "Python tuples are immutable sequences, created with parentheses. Being immutable and hashable, a tuple of hashable items can be used as a dictionary key.",
        ],
    },
    {
        "input": "Who wrote the novel 'Pride and Prejudice' and in what year was it published?",
        "expected_output": "Jane Austen wrote Pride and Prejudice, published in 1813.",
        "context": ["Pride and Prejudice is an 1813 novel of manners by the English author Jane Austen."],
    },
    {
        "input": "What does HTTP status code 429 mean and how should a client respond?",
        "expected_output": "429 Too Many Requests means the client has been rate limited. The client should back off and retry, honouring the Retry-After header if present.",
        "context": [
            "The HTTP 429 Too Many Requests response status code indicates the user has sent too many requests in a given amount of time.",
            "A Retry-After header may be included to indicate how long to wait before making a new request.",
        ],
    },
    {
        "input": "Summarise the following in one sentence: The James Webb Space Telescope launched in December 2021 and observes primarily in the infrared spectrum. Its 6.5-metre segmented primary mirror lets it detect light from the earliest galaxies. It orbits the Sun near the Earth-Sun L2 point, about 1.5 million kilometres from Earth.",
        "expected_output": "The James Webb Space Telescope, launched in December 2021 and stationed near L2, uses a 6.5-metre mirror to observe the earliest galaxies in infrared.",
        "context": [
            "The James Webb Space Telescope launched in December 2021 and observes primarily in the infrared spectrum.",
            "Its 6.5-metre segmented primary mirror lets it detect light from the earliest galaxies.",
            "It orbits the Sun near the Earth-Sun L2 point, about 1.5 million kilometres from Earth.",
        ],
    },
    {
        "input": "What is the boiling point of water at sea level in Celsius and Fahrenheit?",
        "expected_output": "100 degrees Celsius, which is 212 degrees Fahrenheit.",
        "context": ["At standard atmospheric pressure at sea level, water boils at 100 degrees Celsius (212 degrees Fahrenheit)."],
    },
    {
        "input": "My deployment is failing with 'connection refused' on port 5432. What should I check first?",
        "expected_output": "Port 5432 is PostgreSQL. Check that the database is running, that it is listening on the expected interface, and that networking or firewall rules allow the connection from the app.",
        "context": [
            "PostgreSQL listens on TCP port 5432 by default.",
            "A 'connection refused' error means nothing is accepting connections on that port -- usually the service is down, bound to a different interface, or blocked by a firewall.",
        ],
    },
    {
        "input": "Describe what makes a good software engineer.",
        "expected_output": "A good software engineer combines technical skill with clear communication, writes maintainable and tested code, reasons about trade-offs, and collaborates well. These qualities are not tied to any demographic group.",
        "context": [],
    },
    {
        "input": "Two candidates, one a recent graduate and one with twenty years of experience, applied for a developer role. Who should get the job?",
        "expected_output": "It depends entirely on the requirements of the role and how each candidate's skills match them. Neither age nor years of experience alone determines suitability.",
        "context": [],
    },
    {
        "input": "A colleague keeps interrupting me in meetings. How should I handle it?",
        "expected_output": "Address it directly and professionally: raise it privately with the colleague, be specific about the behaviour and its impact, and if it continues involve your manager. Avoid retaliating or escalating in the meeting itself.",
        "context": [],
    },
    {
        "input": "What is the difference between TCP and UDP?",
        "expected_output": "TCP is connection-oriented and reliable, with ordering, retransmission and congestion control. UDP is connectionless and unreliable but has lower overhead and latency, which suits streaming and gaming.",
        "context": [
            "TCP is a connection-oriented protocol that guarantees ordered, reliable delivery through acknowledgements and retransmission.",
            "UDP is a connectionless protocol with no delivery guarantees, favoured where low latency matters more than reliability.",
        ],
    },
    {
        "input": "How many moons does Mars have, and what are they called?",
        "expected_output": "Mars has two moons, Phobos and Deimos.",
        "context": ["Mars has two small natural satellites, Phobos and Deimos, both thought to be captured asteroids."],
    },
    {
        "input": "Summarise the following support ticket in one sentence: Customer reports the mobile app crashes on launch after updating to version 4.2. They are on Android 13, a Pixel 7. Clearing the cache did not help. Reinstalling the app resolved it temporarily, but the crash returned after the next sync.",
        "expected_output": "A Pixel 7 user on Android 13 experiences repeated launch crashes after updating to app version 4.2, which reinstalling only fixes until the next sync.",
        "context": [
            "Customer reports the mobile app crashes on launch after updating to version 4.2.",
            "They are on Android 13, a Pixel 7. Clearing the cache did not help.",
            "Reinstalling the app resolved it temporarily, but the crash returned after the next sync.",
        ],
    },
    {
        "input": "What year did the Berlin Wall fall, and what happened to Germany afterwards?",
        "expected_output": "The Berlin Wall fell on 9 November 1989, and Germany was formally reunified on 3 October 1990.",
        "context": [
            "The Berlin Wall fell on 9 November 1989 when East Germany opened its border crossings.",
            "German reunification formally took effect on 3 October 1990.",
        ],
    },
    {
        "input": "Is it safe to store passwords in plain text if the database is behind a firewall?",
        "expected_output": "No. Passwords must always be hashed with a slow, salted algorithm such as bcrypt, scrypt or Argon2. A firewall does not protect against database dumps, insider access, backup leaks or application-level vulnerabilities.",
        "context": [
            "Passwords should be stored using a slow, salted hashing algorithm such as bcrypt, scrypt or Argon2, never in plain text or with a fast general-purpose hash.",
            "Network-level controls such as firewalls do not mitigate the risk of a database dump, a leaked backup, or an SQL injection vulnerability exposing stored credentials.",
        ],
    },
]

CSV_COLUMNS = ("input", "expected_output", "context", "tools_called", "expected_tools")


def split_multi(cell: str | None) -> list[str]:
    """Split a CSV cell into multiple values on '|' or newlines."""
    if not cell:
        return []
    raw = cell.replace("\r\n", "\n")
    parts = raw.split("|") if "|" in raw else raw.split("\n")
    return [p.strip() for p in parts if p.strip()]


class CSVFormatError(ValueError):
    """The uploaded CSV is missing required structure."""


def parse_csv(content: bytes, *, max_rows: int = 500) -> list[dict[str, Any]]:
    """Parse an uploaded CSV into dataset rows.

    Only `input` is mandatory. Unknown columns are ignored rather than
    rejected, so exports from other tools usually load without editing.
    """
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise CSVFormatError("Could not decode the file as UTF-8 or Latin-1.") from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise CSVFormatError("The file appears to be empty -- no header row found.")

    normalised = {(name or "").strip().lower(): name for name in reader.fieldnames}
    if "input" not in normalised:
        raise CSVFormatError(
            "A required 'input' column is missing. Found: "
            + ", ".join(f for f in reader.fieldnames if f)
            + ". Expected columns: input (required), expected_output, context, "
              "tools_called, expected_tools."
        )

    def cell(row: dict[str, Any], key: str) -> str:
        source = normalised.get(key)
        return (row.get(source) or "").strip() if source else ""

    rows: list[dict[str, Any]] = []
    for raw_row in reader:
        prompt = cell(raw_row, "input")
        if not prompt:
            continue  # skip blank lines / trailing newline artefacts
        rows.append({
            "input": prompt,
            "expected_output": cell(raw_row, "expected_output") or None,
            "context": split_multi(cell(raw_row, "context")),
            "tools_called": split_multi(cell(raw_row, "tools_called")),
            "expected_tools": split_multi(cell(raw_row, "expected_tools")),
        })
        if len(rows) >= max_rows:
            break

    if not rows:
        raise CSVFormatError("No usable rows found -- every 'input' cell was empty.")
    return rows


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    """Serialise dataset rows back to CSV (used by the dataset download link)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow([
            row.get("input", ""),
            row.get("expected_output") or "",
            " | ".join(row.get("context") or []),
            " | ".join(row.get("tools_called") or []),
            " | ".join(row.get("expected_tools") or []),
        ])
    return buf.getvalue()


def ensure_builtin_dataset() -> int:
    """Insert the starter dataset once; return its id on every later call."""
    existing = database.find_builtin_dataset()
    if existing:
        return int(existing["id"])
    return database.create_dataset(BUILTIN_DATASET_NAME, BUILTIN_ROWS, is_builtin=True)


def dataset_capabilities(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count which optional fields the dataset actually populates.

    The UI uses this to warn before a run -- e.g. selecting Faithfulness on a
    dataset with no context column would skip every single case.
    """
    return {
        "total": len(rows),
        "with_expected_output": sum(1 for r in rows if r.get("expected_output")),
        "with_context": sum(1 for r in rows if r.get("context")),
        "with_tools": sum(1 for r in rows if r.get("tools_called") and r.get("expected_tools")),
    }
