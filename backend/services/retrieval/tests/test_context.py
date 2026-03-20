"""Behavioral tests for ContextBuilder."""

from retrieval.context import ContextBuilder
from storage.models import MemoryFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fact(
    content: str,
    importance_score: float | None = None,
    user_id: str = "u1",
) -> MemoryFact:
    return MemoryFact(
        user_id=user_id,
        content=content,
        importance_score=importance_score,
    )


# ---------------------------------------------------------------------------
# Tests — build()
# ---------------------------------------------------------------------------


def test_build_returns_empty_string_when_no_memories():
    builder = ContextBuilder()

    result = builder.build([])

    assert result == ""


def test_build_formats_header_and_bullet_lines():
    facts = [_make_fact("likes coffee"), _make_fact("works at Acme")]
    builder = ContextBuilder(header="## Memories")

    result = builder.build(facts)

    assert result == "## Memories\n- likes coffee\n- works at Acme"


def test_build_uses_custom_header():
    facts = [_make_fact("hello")]
    builder = ContextBuilder(header="# User Context")

    result = builder.build(facts)

    assert result.startswith("# User Context\n")


def test_build_shows_importance_scores_when_enabled():
    facts = [_make_fact("likes tea", importance_score=0.85)]
    builder = ContextBuilder(show_importance=True)

    result = builder.build(facts)

    assert "(importance: 0.85)" in result


def test_build_hides_importance_scores_by_default():
    facts = [_make_fact("likes tea", importance_score=0.85)]
    builder = ContextBuilder()

    result = builder.build(facts)

    assert "importance" not in result


def test_build_omits_importance_tag_when_score_is_none():
    facts = [_make_fact("likes tea", importance_score=None)]
    builder = ContextBuilder(show_importance=True)

    result = builder.build(facts)

    assert "importance" not in result


def test_build_truncates_to_max_memories_preserving_order():
    facts = [_make_fact("first"), _make_fact("second"), _make_fact("third")]
    builder = ContextBuilder(max_memories=2)

    result = builder.build(facts)

    assert "- first" in result
    assert "- second" in result
    assert "third" not in result


def test_build_preserves_input_order():
    facts = [_make_fact("alpha"), _make_fact("beta"), _make_fact("gamma")]
    builder = ContextBuilder()

    result = builder.build(facts)

    lines = result.split("\n")
    assert lines[1] == "- alpha"
    assert lines[2] == "- beta"
    assert lines[3] == "- gamma"


def test_build_formats_importance_to_two_decimal_places():
    facts = [_make_fact("precise", importance_score=0.1)]
    builder = ContextBuilder(show_importance=True)

    result = builder.build(facts)

    assert "(importance: 0.10)" in result


# ---------------------------------------------------------------------------
# Tests — build_system_prompt()
# ---------------------------------------------------------------------------


def test_build_system_prompt_returns_base_prompt_unchanged_when_no_memories():
    builder = ContextBuilder()

    result = builder.build_system_prompt("You are a helpful assistant.", [])

    assert result == "You are a helpful assistant."


def test_build_system_prompt_appends_context_with_blank_line_separator():
    facts = [_make_fact("lives in NYC")]
    builder = ContextBuilder(header="## Memories")

    result = builder.build_system_prompt("You are a helpful assistant.", facts)

    assert result == (
        "You are a helpful assistant.\n"
        "\n"
        "## Memories\n"
        "- lives in NYC"
    )
