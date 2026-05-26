"""Streamlit chat UI for the Bitext dataset analyst."""

from __future__ import annotations

from typing import Any

import streamlit as st

from bitext_agent.config import AgentSettings, DEFAULT_DATASET_CACHE, DEFAULT_MODEL
from bitext_agent.graph import BitextAgent, build_agent
from bitext_agent.streamlit_store import (
    StreamlitSessionStore,
    UiTurn,
    UnknownUiSessionError,
)
from bitext_agent.streamlit_ui import (
    new_session_id,
    trace_event_counts,
    trace_event_label,
)


def main() -> None:
    """Render and run the Streamlit chat application."""

    st.set_page_config(page_title="Bitext Dataset Analyst", page_icon=":bar_chart:")
    _render_developer_style()

    store = _session_store()
    _initialize_active_session(store)

    st.title("Bitext Dataset Analyst")
    st.caption("LangGraph dataset agent console")
    _render_sidebar(store)
    _render_active_session_header(store)

    session_id = st.session_state.active_session_id
    for turn in store.list_turns(session_id):
        _render_turn(turn)

    question = st.chat_input("Ask about the Bitext dataset")
    if question:
        _handle_question(store, session_id, question)


@st.cache_resource
def _session_store() -> StreamlitSessionStore:
    return StreamlitSessionStore()


def _initialize_active_session(store: StreamlitSessionStore) -> None:
    if "agents" not in st.session_state:
        st.session_state.agents = {}
    if "session_error" not in st.session_state:
        st.session_state.session_error = ""
    if "active_session_id" in st.session_state:
        return

    sessions = store.list_sessions()
    if sessions:
        st.session_state.active_session_id = sessions[0].session_id
        return

    session_id = new_session_id()
    store.create_session(session_id)
    st.session_state.active_session_id = session_id


def _render_sidebar(store: StreamlitSessionStore) -> None:
    with st.sidebar:
        st.header("Session")
        sessions = store.list_sessions()
        active_session_id = st.session_state.active_session_id
        session_ids = [session.session_id for session in sessions]

        if session_ids:
            selected_session_id = st.selectbox(
                "Previous sessions",
                session_ids,
                index=session_ids.index(active_session_id)
                if active_session_id in session_ids
                else 0,
                format_func=_session_label(sessions),
            )
            if selected_session_id != active_session_id:
                _switch_session(store, selected_session_id)

        with st.form("resume_session_form"):
            requested_session_id = st.text_input("Session ID", value=active_session_id)
            submitted = st.form_submit_button("Resume session", use_container_width=True)
        if submitted:
            try:
                _switch_session(store, requested_session_id.strip())
            except UnknownUiSessionError as error:
                st.session_state.session_error = str(error)

        if st.session_state.session_error:
            st.error(st.session_state.session_error)

        if st.button("Start new session", use_container_width=True):
            session_id = new_session_id()
            store.create_session(session_id)
            _switch_session(store, session_id)

        st.divider()
        st.subheader("Runtime")
        st.caption(f"Model: `{DEFAULT_MODEL}`")
        st.caption(f"Dataset: `{DEFAULT_DATASET_CACHE}`")
        st.caption(f"Active: `{active_session_id}`")


def _render_active_session_header(store: StreamlitSessionStore) -> None:
    session = store.require_session(st.session_state.active_session_id)
    turns = store.list_turns(session.session_id)
    user_turns = sum(1 for turn in turns if turn.role == "user")
    assistant_turns = sum(1 for turn in turns if turn.role == "assistant")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Session", session.title)
    col_b.metric("Turns", user_turns + assistant_turns)
    col_c.metric("Last used", _compact_timestamp(session.last_used_at))


def _handle_question(
    store: StreamlitSessionStore,
    session_id: str,
    question: str,
) -> None:
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        try:
            agent = _agent_for_session(session_id)
            final_state, trace_events = _run_agent_with_status(agent, question)
        except RuntimeError as error:
            st.error(str(error))
            return

        answer = final_state.get("final_answer") or "I did not receive a final answer."
        st.markdown(answer)
        _render_run_details(final_state, trace_events)

    store.update_title_from_question(session_id, question)
    store.add_turn(session_id, "user", question)
    store.add_turn(session_id, "assistant", answer, trace_events)
    st.rerun()


def _run_agent_with_status(
    agent: BitextAgent,
    question: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final_state: dict[str, Any] = {}
    trace_events: list[dict[str, Any]] = []
    with st.status("Graph execution", expanded=True) as status:
        for event in agent.stream(question):
            if event.get("type") == "final":
                final_state = event.get("state", {})
            else:
                trace_events.append(event)
                status.write(trace_event_label(event))
        status.update(label="Graph execution complete", state="complete", expanded=False)
    return final_state, trace_events


def _agent_for_session(session_id: str) -> BitextAgent:
    agents: dict[str, BitextAgent] = st.session_state.agents
    if session_id not in agents:
        settings = AgentSettings.from_env(
            session_id=session_id,
            resume_existing_session=False,
            user_id="default",
        )
        agents[session_id] = build_agent(settings)
    return agents[session_id]


def _switch_session(store: StreamlitSessionStore, session_id: str) -> None:
    store.require_session(session_id)
    st.session_state.active_session_id = session_id
    st.session_state.session_error = ""
    st.rerun()


def _render_turn(turn: UiTurn) -> None:
    with st.chat_message(turn.role):
        st.markdown(turn.content)
        if turn.trace_events:
            _render_run_details({}, turn.trace_events)


def _render_run_details(
    final_state: dict[str, Any],
    trace_events: list[dict[str, Any]],
) -> None:
    counts = trace_event_counts(trace_events)
    route = final_state.get("query_type") or _route_from_trace(trace_events) or "unknown"

    with st.expander("Run details", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Route", route)
        col_b.metric("Tools", counts["tool_call"])
        col_c.metric("Observations", counts["observation"])
        col_d.metric("Errors", counts["error"])

        if final_state.get("observability_path"):
            st.code(final_state["observability_path"], language="text")

        summary_tab, raw_tab = st.tabs(["Trace", "Raw events"])
        with summary_tab:
            for event in trace_events:
                st.code(trace_event_label(event), language="text")
        with raw_tab:
            st.json(trace_events)


def _session_label(sessions):
    labels = {
        session.session_id: (
            f"{session.title} | {session.session_id[-8:]} | "
            f"{_compact_timestamp(session.last_used_at)}"
        )
        for session in sessions
    }
    return lambda session_id: labels.get(session_id, session_id)


def _route_from_trace(trace_events: list[dict[str, Any]]) -> str | None:
    for event in trace_events:
        if event.get("type") == "router":
            route = event.get("query_type")
            return str(route) if route else None
    return None


def _compact_timestamp(value: str) -> str:
    if "T" not in value:
        return value
    date_part, time_part = value.split("T", 1)
    return f"{date_part} {time_part[:5]}"


def _render_developer_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0e1116;
            color: #e6edf3;
        }
        [data-testid="stSidebar"] {
            background: #111820;
            border-right: 1px solid #263241;
        }
        [data-testid="stMetric"] {
            background: #151b23;
            border: 1px solid #263241;
            border-radius: 6px;
            padding: 10px 12px;
        }
        div[data-testid="stChatMessage"] {
            border: 1px solid #263241;
            background: #111820;
            border-radius: 6px;
            padding: 8px;
        }
        div[data-testid="stExpander"] {
            border-color: #263241;
            background: #0f151d;
        }
        code {
            color: #7ee787;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
