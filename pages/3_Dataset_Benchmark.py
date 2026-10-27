"""
Page 3: Dataset Benchmark
Benchmarking Extractive and Abstractive Summarization on CNN/DailyMail dataset.
Includes Aggregated ROUGE-1/2/L Comparison, Latency Profiling, and CSV Export.
"""

import streamlit as st
import pandas as pd
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.data.dataset_loader import load_cnn_dailymail_samples
from src.summarization.extractive import ExtractiveSummarizer
from src.summarization.abstractive import AbstractiveSummarizer, SUPPORTED_MODELS
from src.evaluation.benchmark import run_benchmark_on_samples
from src.utils.export import export_benchmark_to_csv

st.set_page_config(
    page_title="Dataset Benchmark | TextSummarize AI",
    page_icon="🧪",
    layout="wide",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .bench-banner {
        background: radial-gradient(circle at 80% 20%, rgba(16, 185, 129, 0.25) 0%, rgba(15, 23, 42, 0) 50%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 18px;
        padding: 2rem 2.4rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .bench-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #34D399 0%, #38BDF8 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .bench-desc {
        color: #94A3B8;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
    <div class="bench-banner">
        <div class="bench-title">CNN/DailyMail Dataset Benchmark</div>
        <div class="bench-desc">
            Empirically evaluate and compare model quality across standard multi-sentence news articles and gold-standard editorial highlights.
            Benchmark runs are strictly bounded (1–10 samples) to ensure fast completion on standard CPU hardware.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Configuration controls
col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

with col_cfg1:
    sample_count = st.slider("Evaluation Samples Count:", min_value=1, max_value=10, value=3)

with col_cfg2:
    dataset_source = st.selectbox(
        "Dataset Source:",
        options=["Curated CNN/DailyMail Split (Fast & Offline-Ready)", "Hugging Face Live Streaming (abisee/cnn_dailymail)"],
        index=0,
    )

with col_cfg3:
    model_choices = {k: v["name"] for k, v in SUPPORTED_MODELS.items()}
    bench_model_key = st.selectbox(
        "Abstractive Model:",
        options=list(model_choices.keys()),
        format_func=lambda k: model_choices[k],
        index=0,
    )

with st.expander("⚙️ Advanced Parameters"):
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        ext_sents = st.slider("Extractive Sentences:", 1, 6, 3)
    with pcol2:
        abs_tokens = st.slider("Abstractive Max Tokens:", 50, 180, 120)

# Caching models
@st.cache_resource(show_spinner=False)
def load_benchmark_models(model_key: str):
    ext_summarizer = ExtractiveSummarizer()
    abs_summarizer = AbstractiveSummarizer(model_key=model_key)
    return ext_summarizer, abs_summarizer

# Benchmark Execution Trigger
if st.button("🚀 Execute Benchmark Run", use_container_width=True):
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(current: int, total: int, msg: str):
        progress_bar.progress(current / total)
        status_text.text(msg)

    # 1. Load samples
    status_text.text("Loading dataset samples...")
    use_hf = "Live Streaming" in dataset_source
    samples = load_cnn_dailymail_samples(
        num_samples=sample_count,
        use_huggingface_streaming=use_hf,
    )

    if not samples:
        st.error("Failed to load dataset samples. Please try using the Curated Local Split.")
    else:
        status_text.text("Initializing summarization engines...")
        ext_engine, abs_engine = load_benchmark_models(bench_model_key)

        status_text.text("Running dual summarization and ROUGE scoring...")
        results = run_benchmark_on_samples(
            samples=samples,
            extractive_summarizer=ext_engine,
            abstractive_summarizer=abs_engine,
            num_extractive_sentences=ext_sents,
            abstractive_max_length=abs_tokens,
            progress_callback=update_progress,
        )

        progress_bar.progress(1.0)
        status_text.text("✅ Benchmark run finished successfully!")

        st.session_state["benchmark_results"] = results
        st.success(f"Successfully evaluated {results['successful_samples']} of {results['total_samples']} samples!")

# Display Benchmark Results
if "benchmark_results" in st.session_state:
    res = st.session_state["benchmark_results"]
    agg = res.get("aggregate", {})
    records = res.get("records", [])

    st.markdown("---")
    st.markdown("### 🏆 Aggregate Benchmark Performance")

    # Aggregate metric cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Evaluated Samples", f"{res['successful_samples']}/{res['total_samples']}")
    mcol2.metric("Extractive Mean ROUGE-1", f"{agg.get('ext_mean_rouge1_f1', 0.0):.1f}%")
    mcol3.metric("Abstractive Mean ROUGE-1", f"{agg.get('abs_mean_rouge1_f1', 0.0):.1f}%")
    mcol4.metric("Avg Latency (Ext vs Abs)", f"{agg.get('ext_avg_latency_sec', 0.0):.3f}s / {agg.get('abs_avg_latency_sec', 0.0):.2f}s")

    # Comparison Plot
    agg_chart_data = pd.DataFrame([
        {"Metric": "ROUGE-1 F1", "Extractive (TF-IDF)": agg.get("ext_mean_rouge1_f1", 0.0), "Abstractive (Transformer)": agg.get("abs_mean_rouge1_f1", 0.0)},
        {"Metric": "ROUGE-2 F1", "Extractive (TF-IDF)": agg.get("ext_mean_rouge2_f1", 0.0), "Abstractive (Transformer)": agg.get("abs_mean_rouge2_f1", 0.0)},
        {"Metric": "ROUGE-L F1", "Extractive (TF-IDF)": agg.get("ext_mean_rougeL_f1", 0.0), "Abstractive (Transformer)": agg.get("abs_mean_rougeL_f1", 0.0)},
    ])
    melted_df = agg_chart_data.melt(id_vars=["Metric"], var_name="Method", value_name="Score (%)")

    if HAS_PLOTLY:
        fig_bench = px.bar(
            melted_df,
            x="Metric",
            y="Score (%)",
            color="Method",
            barmode="group",
            text="Score (%)",
            title="Aggregate Mean ROUGE F1 Comparison across CNN/DailyMail Samples",
            color_discrete_map={"Extractive (TF-IDF)": "#38BDF8", "Abstractive (Transformer)": "#A78BFA"},
        )
        fig_bench.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bench.update_layout(yaxis_range=[0, 100], template="plotly_dark", height=380)
        st.plotly_chart(fig_bench, use_container_width=True)
    else:
        pivot_bench = agg_chart_data.set_index("Metric")
        st.bar_chart(pivot_bench, height=350)

    # Detailed per-sample records table
    st.markdown("### 📋 Sample-by-Sample Details")
    df_records = pd.DataFrame(records)
    display_cols = [
        "sample_id", "original_words",
        "ext_words", "ext_latency_sec", "ext_rouge1_f1", "ext_rouge2_f1", "ext_rougeL_f1",
        "abs_words", "abs_latency_sec", "abs_rouge1_f1", "abs_rouge2_f1", "abs_rougeL_f1",
    ]
    st.dataframe(
        df_records[[c for c in display_cols if c in df_records.columns]],
        use_container_width=True,
        hide_index=True,
    )

    # Inspect individual sample texts
    with st.expander("🔍 Inspect Full Sample Summaries & References"):
        for r in records:
            st.markdown(f"#### Sample: `{r.get('sample_id')}`")
            st.markdown(f"**Ground-Truth Reference:** {r.get('reference_summary', '')}")
            st.markdown(f"**Extractive Output:** {r.get('ext_summary', '')}")
            st.markdown(f"**Abstractive Output:** {r.get('abs_summary', '')}")
            st.markdown("---")

    # CSV Export
    csv_bytes = export_benchmark_to_csv(records)
    st.download_button(
        "📥 Download Benchmark Results (CSV)",
        data=csv_bytes,
        file_name="cnn_dailymail_benchmark_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
