"""
Page 1: Summarize Document
Interactive Summarization Workspace supporting Extractive (TF-IDF) & Abstractive (Transformer).
"""

import streamlit as st
import time

from src.utils.validation import validate_input_text, validate_uploaded_file
from src.utils.text_statistics import compute_text_statistics, compute_compression_ratio
from src.utils.export import export_summary_to_txt, export_summary_to_json
from src.summarization.extractive import ExtractiveSummarizer
from src.summarization.abstractive import AbstractiveSummarizer, SUPPORTED_MODELS
from src.summarization.multilingual import detect_language, get_language_display_name
from src.utils.voice import (
    render_browser_voice_dictation_html,
    render_tts_audio_player_html,
    transcribe_wav_audio_bytes,
    SUPPORTED_VOICE_LANGUAGES,
)
from src.evaluation.rouge_metrics import compute_rouge_scores
from src.data.sample_articles import get_sample_articles

st.set_page_config(
    page_title="Summarize Document | TextSummarize AI",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Summarization Workspace")
st.caption("Generate extractive (TF-IDF) and abstractive (Transformer) summaries from multi-format input across English, Hindi, and Marathi.")

# Session state initialization
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "reference_summary" not in st.session_state:
    st.session_state.reference_summary = ""
if "extractive_summary" not in st.session_state:
    st.session_state.extractive_summary = ""
if "abstractive_summary" not in st.session_state:
    st.session_state.abstractive_summary = ""
if "extractive_metadata" not in st.session_state:
    st.session_state.extractive_metadata = {}
if "abstractive_metadata" not in st.session_state:
    st.session_state.abstractive_metadata = {}
if "rouge_results" not in st.session_state:
    st.session_state.rouge_results = {}
if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "auto"

# Model Cache helper in Streamlit
@st.cache_resource(show_spinner=False)
def get_cached_abstractive_model(model_key: str):
    return AbstractiveSummarizer(model_key=model_key)

@st.cache_resource(show_spinner=False)
def get_cached_extractive_model():
    return ExtractiveSummarizer()

# Sidebar Settings
with st.sidebar:
    st.markdown("### ⚙️ Model & Generation Controls")
    
    model_choices = {k: v["name"] for k, v in SUPPORTED_MODELS.items()}
    selected_model_key = st.selectbox(
        "Abstractive Architecture:",
        options=list(model_choices.keys()),
        format_func=lambda k: model_choices[k],
        index=0,
        help="Select the Transformer backbone architecture for neural abstractive summarization.",
    )
    st.session_state.selected_model_key = selected_model_key

    st.markdown("#### Language Configuration")
    lang_selection = st.selectbox(
        "Target Language:",
        options=["auto", "en", "hi", "mr"],
        format_func=lambda x: {
            "auto": "🌐 Auto-Detect (EN / HI / MR)",
            "en": "English",
            "hi": "Hindi (हिंदी)",
            "mr": "Marathi (मराठी)",
        }[x],
        index=0,
    )
    st.session_state.selected_lang = lang_selection

    st.markdown("#### Extractive Parameters")
    num_sentences = st.slider(
        "Extractive Sentences:",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of most salient sentences to extract and preserve in original chronological order.",
    )

    st.markdown("#### Abstractive Parameters")
    abs_max_tokens = st.slider(
        "Max Output Length (Tokens):",
        min_value=40,
        max_value=250,
        value=130,
        step=10,
    )
    abs_min_tokens = st.slider(
        "Min Output Length (Tokens):",
        min_value=15,
        max_value=100,
        value=35,
        step=5,
    )
    
    run_mode = st.radio(
        "Summarization Execution Mode:",
        options=["Both (Extractive & Abstractive)", "Extractive Only", "Abstractive Only"],
        index=0,
    )

# Document Source selection
col_sample, col_upload = st.columns([1, 1])

with col_sample:
    samples = get_sample_articles()
    sample_options = ["Custom Input (Type, Paste, or Dictate below)"] + [f"{s['title']} ({s['category']})" for s in samples]
    selected_sample = st.selectbox("Load Sample Document (with Reference):", sample_options)

    if selected_sample != "Custom Input (Type, Paste, or Dictate below)":
        idx = sample_options.index(selected_sample) - 1
        st.session_state.document_text = samples[idx]["article"]
        st.session_state.reference_summary = samples[idx]["reference_summary"]

with col_upload:
    uploaded_file = st.file_uploader("Or Upload Document (.txt):", type=["txt"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        is_valid, err, decoded_text = validate_uploaded_file(uploaded_file.name, file_bytes)
        if is_valid:
            st.session_state.document_text = decoded_text
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(decoded_text.split()):,} words)")
        else:
            st.error(err)

# Voice & Audio Input Expander
with st.expander("🎙️ Voice & Audio Summarization (English, Hindi, Marathi)", expanded=False):
    tab_mic, tab_audio_file = st.tabs(["🔴 Live Microphone Dictation", "📁 Audio File Upload / Recording"])
    
    with tab_mic:
        st.caption("Click microphone below to dictate text directly into your browser in English, Hindi, or Marathi:")
        active_code = "en-IN" if st.session_state.selected_lang in ["auto", "en"] else f"{st.session_state.selected_lang}-IN"
        st.components.v1.html(render_browser_voice_dictation_html(active_code), height=130)

    with tab_audio_file:
        voice_rec = st.audio_input("Record Audio from Microphone")
        if voice_rec is not None:
            audio_bytes = voice_rec.read()
            with st.spinner("Transcribing audio to text..."):
                ok, trans_text = transcribe_wav_audio_bytes(audio_bytes, language_code=active_code)
                if ok:
                    st.session_state.document_text = trans_text
                    st.success("Transcribed audio successfully!")
                    st.rerun()
                else:
                    st.warning(f"Audio transcription notice: {trans_text}")

# Text Input Area
input_text = st.text_area(
    "Source Document Text:",
    value=st.session_state.document_text,
    height=230,
    placeholder="Paste, type, or dictate document text here in English, Hindi (हिंदी), or Marathi (मराठी)...",
)
st.session_state.document_text = input_text

# Live Text Stats & Language Detection
if input_text.strip():
    detected_lang_code = detect_language(input_text)
    detected_lang_label = get_language_display_name(detected_lang_code)
    stats = compute_text_statistics(input_text, label="Input")
    
    stat_cols = st.columns(5)
    stat_cols[0].metric("Detected Language", detected_lang_label)
    stat_cols[1].metric("Word Count", f"{stats['word_count']:,}")
    stat_cols[2].metric("Character Count", f"{stats['char_count']:,}")
    stat_cols[3].metric("Sentences", f"{stats['sentence_count']:,}")
    stat_cols[4].metric("Est. Reading Time", f"{stats['reading_time_sec']}s")

# Reference Summary Input (Optional for ROUGE)
with st.expander("🎯 Ground-Truth Reference Summary (Optional - Required for ROUGE evaluation)", expanded=bool(st.session_state.reference_summary)):
    ref_text = st.text_area(
        "Reference Summary:",
        value=st.session_state.reference_summary,
        height=100,
        placeholder="Enter authentic human reference summary if you wish to compute ROUGE-1, ROUGE-2, and ROUGE-L scores...",
    )
    st.session_state.reference_summary = ref_text
    st.caption("ℹ️ Genuine ROUGE evaluation compares generated summaries against this gold-standard text.")

# Summarization Trigger Button
if st.button("⚡ Generate Summaries", use_container_width=True):
    validation = validate_input_text(input_text)
    if not validation.is_valid:
        st.error(f"❌ {validation.error_message}")
    else:
        if validation.warning_message:
            st.warning(f"⚠️ {validation.warning_message}")

        extractive_done = False
        abstractive_done = False
        
        # 1. Extractive Summarization
        if "Extractive" in run_mode:
            with st.spinner("Extracting salient sentences via TF-IDF ranking..."):
                ext_model = get_cached_extractive_model()
                ext_res = ext_model.summarize(validation.cleaned_text, num_sentences=num_sentences)
                st.session_state.extractive_summary = ext_res["summary"]
                st.session_state.extractive_metadata = ext_res
                extractive_done = True

        # 2. Abstractive Summarization
        if "Abstractive" in run_mode:
            with st.spinner(f"Loading {SUPPORTED_MODELS[selected_model_key]['name']} and generating abstractive summary..."):
                try:
                    abs_model = get_cached_abstractive_model(selected_model_key)
                    abs_res = abs_model.summarize(
                        validation.cleaned_text,
                        max_length=abs_max_tokens,
                        min_length=abs_min_tokens,
                    )
                    st.session_state.abstractive_summary = abs_res["summary"]
                    st.session_state.abstractive_metadata = abs_res
                    abstractive_done = True
                except Exception as e:
                    st.error(f"Abstractive generation failed: {str(e)}")

        # 3. ROUGE Evaluation if Reference is Present
        if st.session_state.reference_summary.strip():
            rouge_data = {}
            if st.session_state.extractive_summary:
                rouge_data["Extractive"] = compute_rouge_scores(
                    st.session_state.extractive_summary, st.session_state.reference_summary
                )
            if st.session_state.abstractive_summary:
                rouge_data["Abstractive"] = compute_rouge_scores(
                    st.session_state.abstractive_summary, st.session_state.reference_summary
                )
            st.session_state.rouge_results = rouge_data
        else:
            st.session_state.rouge_results = {}

        st.success("✅ Summarization completed successfully!")

# Display Summarization Results
if st.session_state.extractive_summary or st.session_state.abstractive_summary:
    st.markdown("---")
    st.markdown("### 📋 Generated Summaries")

    res_col1, res_col2 = st.columns(2)
    orig_words = len(input_text.split()) if input_text else 1

    with res_col1:
        st.markdown("#### 🔹 Extractive Summary (TF-IDF Ranking)")
        if st.session_state.extractive_summary:
            ext_words = len(st.session_state.extractive_summary.split())
            ext_comp = compute_compression_ratio(orig_words, ext_words)
            ext_time = st.session_state.extractive_metadata.get("execution_time", 0.0)

            st.text_area(
                "Extractive Output",
                st.session_state.extractive_summary,
                height=220,
                label_visibility="collapsed",
            )
            
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Words", f"{ext_words}")
            mcol2.metric("Compression", f"{ext_comp}%")
            mcol3.metric("Latency", f"{ext_time:.3f}s")

            # Sentence ranking breakdown
            with st.expander("🔍 Salient Sentences Breakdown"):
                for idx, sent, score in st.session_state.extractive_metadata.get("sentence_scores", []):
                    is_chosen = sent in st.session_state.extractive_metadata.get("selected_sentences", [])
                    prefix = "✅ **[Selected]**" if is_chosen else "⚪ [Excluded]"
                    st.markdown(f"{prefix} *Score: {score:.3f}* — {sent}")

            # Audio Text-to-Speech playback
            voice_lang = "en-IN" if detected_lang_code == "en" else f"{detected_lang_code}-IN"
            st.components.v1.html(render_tts_audio_player_html(st.session_state.extractive_summary, voice_lang), height=55)
        else:
            st.info("Extractive summarization not run for this document.")

    with res_col2:
        st.markdown(f"#### 🔸 Abstractive Summary ({SUPPORTED_MODELS[st.session_state.selected_model_key]['name']})")
        if st.session_state.abstractive_summary:
            abs_words = len(st.session_state.abstractive_summary.split())
            abs_comp = compute_compression_ratio(orig_words, abs_words)
            abs_time = st.session_state.abstractive_metadata.get("execution_time", 0.0)
            device_used = st.session_state.abstractive_metadata.get("device", "CPU")

            st.text_area(
                "Abstractive Output",
                st.session_state.abstractive_summary,
                height=220,
                label_visibility="collapsed",
            )

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Words", f"{abs_words}")
            mcol2.metric("Compression", f"{abs_comp}%")
            mcol3.metric("Latency", f"{abs_time:.2f}s ({device_used})")

            if st.session_state.abstractive_metadata.get("truncated"):
                st.info("ℹ️ Input exceeded Transformer context window; first 1024 tokens summarized.")

            # Audio Text-to-Speech playback
            voice_lang = "en-IN" if detected_lang_code == "en" else f"{detected_lang_code}-IN"
            st.components.v1.html(render_tts_audio_player_html(st.session_state.abstractive_summary, voice_lang), height=55)
        else:
            st.info("Abstractive summarization not run for this document.")

    # Export & Navigation Section
    st.markdown("---")
    act_col1, act_col2, act_col3 = st.columns([1, 1, 1])

    with act_col1:
        txt_data = export_summary_to_txt(
            original_text=input_text,
            extractive_summary=st.session_state.extractive_summary,
            abstractive_summary=st.session_state.abstractive_summary,
            model_name=SUPPORTED_MODELS[st.session_state.selected_model_key]["name"],
            metrics={k: v.to_dict() for k, v in st.session_state.rouge_results.items()} if st.session_state.rouge_results else None,
        )
        st.download_button(
            "📥 Download Report (TXT)",
            data=txt_data,
            file_name="summarize_ai_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with act_col2:
        json_data = export_summary_to_json(
            original_text=input_text,
            extractive_summary=st.session_state.extractive_summary,
            abstractive_summary=st.session_state.abstractive_summary,
            model_name=SUPPORTED_MODELS[st.session_state.selected_model_key]["name"],
            reference_summary=st.session_state.reference_summary,
            metrics={k: v.to_dict() for k, v in st.session_state.rouge_results.items()} if st.session_state.rouge_results else None,
            timings={
                "extractive": st.session_state.extractive_metadata.get("execution_time", 0.0),
                "abstractive": st.session_state.abstractive_metadata.get("execution_time", 0.0),
            },
        )
        st.download_button(
            "📥 Download Structured JSON",
            data=json_data,
            file_name="summarize_ai_data.json",
            mime="application/json",
            use_container_width=True,
        )

    with act_col3:
        if st.button("📊 View Full ROUGE Comparison Dashboard", use_container_width=True):
            st.switch_page("pages/2_Compare_Methods.py")
