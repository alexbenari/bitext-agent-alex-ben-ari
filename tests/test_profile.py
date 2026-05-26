from __future__ import annotations

from types import SimpleNamespace

from bitext_agent.profile import (
    ProfileFact,
    ProfileRepository,
    extract_profile_facts,
    format_profile_answer,
)


class StaticProfileModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.content)


def test_profile_repository_stores_unique_facts_per_user(tmp_path) -> None:
    repository = ProfileRepository(tmp_path / "profiles.sqlite")

    inserted = repository.add_facts(
        "alex",
        ["You prefer refund analytics.", "You prefer refund analytics."],
        source_session_id="session-1",
    )

    assert inserted == 1
    assert repository.list_facts("alex") == [
        ProfileFact(
            fact_id=repository.list_facts("alex")[0].fact_id,
            fact="You prefer refund analytics.",
        )
    ]
    repository.close()


def test_format_profile_answer_handles_empty_and_saved_facts() -> None:
    assert format_profile_answer([]) == "I do not have any saved profile facts for you yet."

    answer = format_profile_answer(
        [
            ProfileFact(fact_id="fact-1", fact="You prefer refund analytics."),
            ProfileFact(fact_id="fact-2", fact="You use the CLI."),
        ]
    )

    assert answer == (
        "I remember:\n"
        "- You prefer refund analytics.\n"
        "- You use the CLI."
    )


def test_extract_profile_facts_passes_existing_profile_and_deduplicates() -> None:
    model = StaticProfileModel(
        '{"facts": ["You prefer refund analytics.", "You use the CLI."]}'
    )

    facts = extract_profile_facts(
        model,
        existing_facts=[
            ProfileFact(fact_id="fact-1", fact="You prefer refund analytics.")
        ],
        user_messages=["My name is Alex", "Show refund examples"],
        final_answer="Here are refund examples.",
    )

    assert facts == ["You use the CLI."]
    assert "existing_profile" in model.messages[1].content
    assert "user_messages" in model.messages[1].content
    assert "final_agent_answer" in model.messages[1].content


def test_extract_profile_facts_returns_empty_list_for_invalid_json() -> None:
    facts = extract_profile_facts(
        StaticProfileModel("not json"),
        existing_facts=[],
        user_messages=["My name is Alex"],
        final_answer="Hello Alex.",
    )

    assert facts == []
