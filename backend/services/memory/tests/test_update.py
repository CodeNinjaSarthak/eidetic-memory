"""Tests for _parse_batch_result — the batch LLM response parser."""


from memory.models.memory import MemoryOperation
from memory.pipeline.update import _parse_batch_result


def test_event_none_returns_noop() -> None:
    result = {"memory": [{"event": None, "id": "0", "text": "hello"}]}

    updates = _parse_batch_result(result, expected_count=1)

    assert updates[0].operation == MemoryOperation.NOOP


def test_event_string_none_wrong_case_returns_noop() -> None:
    result = {"memory": [{"event": "None", "id": "0", "text": "hello"}]}

    updates = _parse_batch_result(result, expected_count=1)

    assert updates[0].operation == MemoryOperation.NOOP


def test_event_none_uppercase_maps_to_noop() -> None:
    result = {"memory": [{"event": "NONE"}]}

    updates = _parse_batch_result(result, expected_count=1)

    assert updates[0].operation == MemoryOperation.NOOP


def test_update_with_id_none_returns_noop() -> None:
    result = {"memory": [{"event": "UPDATE", "id": None, "text": "new content"}]}

    updates = _parse_batch_result(result, expected_count=1)

    assert updates[0].operation == MemoryOperation.NOOP


def test_update_with_id_empty_string_returns_noop() -> None:
    result = {"memory": [{"event": "UPDATE", "id": "", "text": "new content"}]}

    updates = _parse_batch_result(result, expected_count=1)

    assert updates[0].operation == MemoryOperation.NOOP


def test_update_with_missing_text_returns_noop() -> None:
    result = {"memory": [{"event": "UPDATE", "id": "42", "text": None}]}

    updates = _parse_batch_result(result, expected_count=1)

    assert updates[0].operation == MemoryOperation.NOOP


def test_valid_add_returns_add_operation() -> None:
    result = {"memory": [{"event": "ADD", "text": "User likes chess"}]}

    updates = _parse_batch_result(result, expected_count=1)

    assert updates[0].operation == MemoryOperation.ADD


def test_valid_update_returns_update_with_id_and_content() -> None:
    result = {"memory": [{"event": "UPDATE", "id": "42", "text": "new content"}]}

    updates = _parse_batch_result(result, expected_count=1)

    assert updates[0].operation == MemoryOperation.UPDATE
    assert updates[0].memory_id == "42"
    assert updates[0].updated_content == "new content"


def test_fewer_items_than_expected_pads_with_noop() -> None:
    result = {"memory": [{"event": "ADD", "text": "a"}, {"event": "ADD", "text": "b"}]}

    updates = _parse_batch_result(result, expected_count=4)

    assert len(updates) == 4
    assert updates[2].operation == MemoryOperation.NOOP
    assert updates[3].operation == MemoryOperation.NOOP


def test_missing_memory_array_returns_all_noop() -> None:
    result = {"unrelated": "value"}

    updates = _parse_batch_result(result, expected_count=3)

    assert len(updates) == 3
    assert all(u.operation == MemoryOperation.NOOP for u in updates)
