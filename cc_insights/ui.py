"""Minimal Streamlit analyst UI."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `cc_insights` importable when launched via `streamlit run cc_insights/ui.py`
# from the project root (Streamlit adds the script's dir to sys.path, not the
# parent package's dir, so we add it explicitly here).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from cc_insights.app import build_orchestrator, build_session_store


st.set_page_config(page_title="cc-insights", layout="wide")
st.title("Contact-centre Insights")


@st.cache_resource
def _get():
    return build_orchestrator(), build_session_store()


orch, sess = _get()

if "session_id" not in st.session_state:
    st.session_state.session_id = sess.ensure_session(None)

question = st.text_input("Ask the dataset a question:")
if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            answer, qid, latency_ms = orch.ask(question.strip())
        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()
    sess.record(st.session_state.session_id, question.strip(), answer.model_dump())
    st.subheader(answer.headline)
    st.write(answer.body)
    st.caption(f"confidence: {answer.confidence} · latency: {latency_ms} ms · query_id: {qid}")
    if answer.evidence:
        st.markdown("### Evidence")
        for ev in answer.evidence:
            header = f"{ev.conv_id} — {ev.reason}"
            with st.expander(header):
                meta_bits = []
                if ev.retrieved_from:
                    meta_bits.append(f"retrieved from: `{ev.retrieved_from}`")
                if ev.chunk_type:
                    meta_bits.append(f"chunk: `{ev.chunk_type}`")
                if meta_bits:
                    st.caption(" · ".join(meta_bits))
                st.markdown("**Why it was cited**")
                st.write(ev.snippet)
                if ev.turns:
                    st.markdown(f"**Full conversation ({len(ev.turns)} turns)**")
                    for t in ev.turns:
                        sent = f" _({t.sentiment})_" if t.sentiment else ""
                        st.markdown(f"**{t.turn_index:>2} · {t.role}**{sent}")
                        st.write(t.text)
                else:
                    st.caption("(full transcript unavailable)")
    if answer.caveats:
        st.markdown("### Caveats")
        for c in answer.caveats:
            st.write(f"- {c}")
    with st.expander("Plan + tools"):
        st.json(answer.plan.model_dump())
        st.write("tools used:", answer.used_tools)
