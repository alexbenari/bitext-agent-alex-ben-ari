from __future__ import annotations

import json

import pandas as pd

from bitext_agent.dataset import BitextDataset, DatasetFilter
from bitext_agent.tools import DatasetToolbox


def make_dataset() -> BitextDataset:
    return BitextDataset(
        pd.DataFrame(
            [
                {
                    "flags": "B",
                    "instruction": "I want my money back",
                    "category": "REFUND",
                    "intent": "get_refund",
                    "response": "I can help you request a refund.",
                },
                {
                    "flags": "B",
                    "instruction": "Where is my order?",
                    "category": "ORDER",
                    "intent": "track_order",
                    "response": "You can track your order from your account.",
                },
                {
                    "flags": "P",
                    "instruction": "I have a complaint",
                    "category": "FEEDBACK",
                    "intent": "complaint",
                    "response": "I am sorry to hear that and can escalate this.",
                },
                {
                    "flags": "B",
                    "instruction": "I need a refund update",
                    "category": "REFUND",
                    "intent": "track_refund",
                    "response": "I can help you track your refund.",
                },
            ]
        )
    )


def test_schema_summary_groups_intents_by_category() -> None:
    summary = make_dataset().schema_summary()

    assert summary["row_count"] == 4
    assert summary["intents_by_category"]["REFUND"] == ["get_refund", "track_refund"]


def test_filter_rows_uses_literal_substring_search_without_token_reduction() -> None:
    dataset = make_dataset()
    exact_phrase_frame = dataset.filter_rows(DatasetFilter(text_query="money back"))
    natural_language_frame = dataset.filter_rows(
        DatasetFilter(text_query="people wanting their money back")
    )

    assert len(exact_phrase_frame) == 1
    assert exact_phrase_frame.iloc[0]["intent"] == "get_refund"
    assert natural_language_frame.empty


def test_filter_rows_requires_canonical_intent_instead_of_semantic_alias() -> None:
    dataset = make_dataset()

    try:
        dataset.filter_rows(DatasetFilter(intent="refund"))
    except ValueError as error:
        assert "Unknown intent 'refund'" in str(error)
    else:
        raise AssertionError("Expected non-canonical intent alias to be rejected.")


def test_toolbox_filter_then_count_uses_filter_id() -> None:
    toolbox = DatasetToolbox(make_dataset())
    filtered = toolbox.filter_dataset(intent="get_refund")
    counted = toolbox.count_rows(filter_id=filtered["filter_id"])

    assert filtered["total_matches"] == 1
    assert counted == {"count": 1}


def test_toolbox_show_examples_supports_offset_for_follow_ups() -> None:
    toolbox = DatasetToolbox(make_dataset())

    result = toolbox.show_examples(category="REFUND", n=1, offset=1)

    assert result["offset"] == 1
    assert result["count_available"] == 2
    assert result["examples"][0]["intent"] == "track_refund"


def test_toolbox_rebuilds_filter_cache_from_remembered_criteria() -> None:
    original_toolbox = DatasetToolbox(make_dataset())
    filtered = original_toolbox.filter_dataset(category="REFUND")
    restored_toolbox = DatasetToolbox(make_dataset())

    restored_toolbox.remember_filters(
        {
            filtered["filter_id"]: {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            }
        }
    )
    result = restored_toolbox.count_rows(filter_id=filtered["filter_id"])

    assert result == {"count": 2}


def test_structured_tool_returns_json_observation_for_bad_filter_id() -> None:
    toolbox = DatasetToolbox(make_dataset())
    count_rows = next(tool for tool in toolbox.tools() if tool.name == "count_rows")

    result = count_rows.invoke({"filter_id": "missing-filter"})

    payload = json.loads(result)
    assert payload == {
        "ok": False,
        "error": {
            "type": "recoverable_tool_error",
            "message": "Unknown filter_id 'missing-filter'. Call filter_dataset before using it.",
        },
    }


def test_structured_tool_returns_json_observation_for_bad_category() -> None:
    toolbox = DatasetToolbox(make_dataset())
    filter_dataset = next(tool for tool in toolbox.tools() if tool.name == "filter_dataset")

    result = filter_dataset.invoke({"category": "MISSING"})

    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "recoverable_tool_error"
    assert "Unknown category 'MISSING'" in payload["error"]["message"]


def test_structured_tool_returns_json_observation_for_bad_intent() -> None:
    toolbox = DatasetToolbox(make_dataset())
    filter_dataset = next(tool for tool in toolbox.tools() if tool.name == "filter_dataset")

    result = filter_dataset.invoke({"intent": "refund"})

    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "recoverable_tool_error"
    assert "Unknown intent 'refund'" in payload["error"]["message"]


def test_structured_tool_returns_json_observation_for_validation_error() -> None:
    toolbox = DatasetToolbox(make_dataset())
    show_examples = next(tool for tool in toolbox.tools() if tool.name == "show_examples")

    result = show_examples.invoke({"n": 99})

    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "tool_validation_error"
    assert "less than or equal to 10" in payload["error"]["message"]


def test_toolbox_examples_by_category_returns_one_example_per_category() -> None:
    toolbox = DatasetToolbox(make_dataset())
    result = toolbox.examples_by_category(n_per_category=1)

    assert result["category_count"] == 3
    assert result["examples_by_category"]["FEEDBACK"][0]["intent"] == "complaint"
    assert result["examples_by_category"]["ORDER"][0]["intent"] == "track_order"
    assert result["examples_by_category"]["REFUND"][0]["intent"] == "get_refund"


def test_validate_category_accepts_known_dataset_category() -> None:
    dataset = BitextDataset(
        pd.DataFrame(
            [
                {
                    "flags": "B",
                    "instruction": "I need to set up shipping",
                    "category": "SHIPPING",
                    "intent": "set_up_shipping_address",
                    "response": "I can help you set up your shipping address.",
                }
            ]
        )
    )

    assert dataset.validate_category("SHIPPING") == "SHIPPING"


def test_intent_distribution_counts_category_rows() -> None:
    toolbox = DatasetToolbox(make_dataset())
    distribution = toolbox.intent_distribution(category="FEEDBACK")

    assert distribution["total_rows"] == 1
    assert distribution["intent_counts"] == {"complaint": 1}
