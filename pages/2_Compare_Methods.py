"""
Page 2: Compare Methods & ROUGE Evaluation Dashboard
Side-by-side comparative analysis of Extractive (TF-IDF) vs Abstractive (Transformer) summarization.
"""

import streamlit as st
import pandas as pd
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.evaluation.rouge_metrics import compute_rouge_scores
from src.utils.text_statistics import compute_compression_ratio

st.set_page_config(
    page_title="Compare Methods & ROUGE | TextSummarize AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Method Comparison & ROUGE Evaluation")
st.caption("Empirical evaluation comparing TF-IDF sentence salience against Transformer abstractive generation.")

# Retrieve data from session state
doc_text = st.session_state.get("document_text", "")
ext_summary = st.session_state.get("extractive_summary", "")
abs_summary = st.session_state.get("abstractive_summary", "")
ref_summary = st.session_state.get("reference_summary", "")
ext_meta = st.session_state.get("extractive_metadata", {})
abs_meta = st.session_state.get("abstractive_metadata", {})

if not ext_summary and not abs_summary:
    st.info("💡 No summaries generated yet. Please visit the **Summarize** page first to enter text and generate summaries.")
    if st.button("👉 Go to Summarization Workspace"):
        st.switch_page("pages/1_Summarize.py")
    st.stop()

# 1. Summary Overview Cards
st.markdown("### 🔍 Generated Outputs Inspection")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 🔹 Extractive Output (TF-IDF)")
    st.info(ext_summary if ext_summary else "[Not generated]")
with c2:
    st.markdown("#### 🔸 Abstractive Output (Transformer)")
    st.success(abs_summary if abs_summary else "[Not generated]")

# 2. Reference Summary & ROUGE Scores
st.markdown("---")
st.markdown("### 🎯 ROUGE Quality Evaluation (ROUGE-1, ROUGE-2, ROUGE-L)")

if not ref_summary.strip():
    st.warning(
        "⚠️ **Reference-based ROUGE evaluation is currently unavailable because no reference summary was provided.**\n\n"
        "ROUGE requires a genuine human-written gold-standard reference summary to measure n-gram overlap. "
        "You can provide a reference summary below or in the **Summarize** workspace to compute real ROUGE scores."
    )
    new_ref = st.text_area("Enter Reference Summary to evaluate right now:", value="", height=100)
    if st.button("Evaluate ROUGE Now"):
        st.session_state.reference_summary = new_ref
        st.rerun()
else:
    st.markdown(f"**Ground-Truth Reference:** *{ref_summary}*")

    # Compute genuine ROUGE scores
    rouge_ext = compute_rouge_scores(ext_summary, ref_summary) if ext_summary else None
    rouge_abs = compute_rouge_scores(abs_summary, ref_summary) if abs_summary else None

    # Tabular presentation of Precision, Recall, F1
    table_rows = []
    if rouge_ext and rouge_ext.has_reference:
        table_rows.extend([
            {"Method": "Extractive (TF-IDF)", "Metric": "ROUGE-1 (Unigrams)", "Precision": rouge_ext.rouge1.precision, "Recall": rouge_ext.rouge1.recall, "F1-Score": rouge_ext.rouge1.f1},
            {"Method": "Extractive (TF-IDF)", "Metric": "ROUGE-2 (Bigrams)", "Precision": rouge_ext.rouge2.precision, "Recall": rouge_ext.rouge2.recall, "F1-Score": rouge_ext.rouge2.f1},
            {"Method": "Extractive (TF-IDF)", "Metric": "ROUGE-L (LCS)", "Precision": rouge_ext.rougeL.precision, "Recall": rouge_ext.rougeL.recall, "F1-Score": rouge_ext.rougeL.f1},
        ])
    if rouge_abs and rouge_abs.has_reference:
        table_rows.extend([
            {"Method": "Abstractive (Transformer)", "Metric": "ROUGE-1 (Unigrams)", "Precision": rouge_abs.rouge1.precision, "Recall": rouge_abs.rouge1.recall, "F1-Score": rouge_abs.rouge1.f1},
            {"Method": "Abstractive (Transformer)", "Metric": "ROUGE-2 (Bigrams)", "Precision": rouge_abs.rouge2.precision, "Recall": rouge_abs.rouge2.recall, "F1-Score": rouge_abs.rouge2.f1},
            {"Method": "Abstractive (Transformer)", "Metric": "ROUGE-L (LCS)", "Precision": rouge_abs.rougeL.precision, "Recall": rouge_abs.rougeL.recall, "F1-Score": rouge_abs.rougeL.f1},
        ])

    if table_rows:
        df_rouge = pd.DataFrame(table_rows)
        st.dataframe(
            df_rouge.style.format({"Precision": "{:.2f}%", "Recall": "{:.2f}%", "F1-Score": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )

        # ROUGE F1 Comparison Chart
        st.markdown("#### 📊 ROUGE F1 Comparison Chart")
        if HAS_PLOTLY:
            fig_f1 = px.bar(
                df_rouge,
                x="Metric",
                y="F1-Score",
                color="Method",
                barmode="group",
                text="F1-Score",
                color_discrete_map={"Extractive (TF-IDF)": "#3B82F6", "Abstractive (Transformer)": "#10B981"},
                title="ROUGE F1-Score Comparison (%)",
            )
            fig_f1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_f1.update_layout(yaxis_range=[0, 110], template="plotly_dark", height=400)
            st.plotly_chart(fig_f1, use_container_width=True)
        else:
            pivot_f1 = df_rouge.pivot(index="Metric", columns="Method", values="F1-Score")
            st.bar_chart(pivot_f1, height=350)

        st.caption(
            "ℹ️ **ROUGE Interpretation:** ROUGE scores quantify lexical n-gram and longest common subsequence overlap against the reference. "
            "A higher ROUGE score indicates closer token alignment with the gold standard, though it does not independently assess semantic veracity or grammatical elegance."
        )

# 3. Document Statistics & Efficiency Metrics
st.markdown("---")
st.markdown("### 📈 Efficiency & Compression Analytics")

orig_words = len(doc_text.split()) if doc_text else 1
ext_words = len(ext_summary.split()) if ext_summary else 0
abs_words = len(abs_summary.split()) if abs_summary else 0

ext_comp = compute_compression_ratio(orig_words, ext_words)
abs_comp = compute_compression_ratio(orig_words, abs_words)

ext_lat = ext_meta.get("execution_time", 0.0)
abs_lat = abs_meta.get("execution_time", 0.0)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    # Summary length & compression
    stats_df = pd.DataFrame([
        {"Method": "Original Document", "Words": orig_words, "Compression (%)": 0.0},
        {"Method": "Extractive (TF-IDF)", "Words": ext_words, "Compression (%)": ext_comp},
        {"Method": "Abstractive (Transformer)", "Words": abs_words, "Compression (%)": abs_comp},
    ])
    if HAS_PLOTLY:
        fig_len = px.bar(
            stats_df,
            x="Method",
            y="Words",
            color="Method",
            text="Words",
            title="Summary Word Count vs Source Document",
            color_discrete_sequence=["#64748B", "#3B82F6", "#10B981"],
        )
        fig_len.update_traces(textposition="outside")
        fig_len.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig_len, use_container_width=True)
    else:
        st.markdown("##### Word Count Comparison")
        st.bar_chart(stats_df.set_index("Method")["Words"], height=300)

with col_chart2:
    # Inference Latency Comparison
    latency_df = pd.DataFrame([
        {"Method": "Extractive (TF-IDF)", "Latency (s)": ext_lat},
        {"Method": "Abstractive (Transformer)", "Latency (s)": abs_lat},
    ])
    if HAS_PLOTLY:
        fig_lat = px.bar(
            latency_df,
            x="Method",
            y="Latency (s)",
            color="Method",
            text="Latency (s)",
            title="Processing Latency (Seconds)",
            color_discrete_map={"Extractive (TF-IDF)": "#3B82F6", "Abstractive (Transformer)": "#10B981"},
        )
        fig_lat.update_traces(texttemplate="%{text:.3f}s", textposition="outside")
        fig_lat.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.markdown("##### Latency (Seconds)")
        st.bar_chart(latency_df.set_index("Method")["Latency (s)"], height=300)

# 4. Method Trade-off Synthesis Table
st.markdown("---")
st.markdown("### ⚖️ Architectural Trade-off Synthesis")

st.markdown(
    """
| Dimension | Extractive Summarization (TF-IDF) | Abstractive Summarization (Transformer) |
| :--- | :--- | :--- |
| **Primary Strength** | Fast execution (~0.01s), zero hallucination risk, retains exact author phrasing. | Produces fluent executive summaries, paraphrases verbose discourse naturally. |
| **Primary Limitation** | May assemble disjoint sentences lacking conversational coherence. | Higher computational overhead, sensitive to sequence length truncation. |
| **Preservation of Chronology** | **Yes** — Sentences ranked by salience and restored to original document order. | **Yes** — End-to-end autoregressive sequence generation with beam search. |
| **Ideal Use Case** | Rapid document indexing, legal briefs, medical guidelines, low-power devices. | News digests, executive briefings, general literature overview. |
"""
)
