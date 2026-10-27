"""
Page 4: About & Academic Documentation
Comprehensive technical documentation, mathematical formulations, model limitations catalog, and deployment guides.
"""

import streamlit as st
from src.utils.model_specs import (
    get_model_limitations_dataframe,
    render_model_limitations_cards_html,
)

st.set_page_config(
    page_title="About & Architecture | TextSummarize AI",
    page_icon="ℹ️",
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

    .about-banner {
        background: radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.25) 0%, rgba(15, 23, 42, 0) 50%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 18px;
        padding: 2rem 2.4rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .about-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #A78BFA 0%, #F472B6 50%, #FB7185 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .about-desc {
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
    <div class="about-banner">
        <div class="about-title">Architecture, Methodology & Documentation</div>
        <div class="about-desc">
            Theoretical foundations, mathematical formulations, and engineering architecture behind 
            <strong>TextSummarize AI</strong>. Includes explicit operational constraints, token budgets, and deployment blueprints.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs for structured reading
tab_methods, tab_rouge, tab_limits, tab_deploy, tab_refs = st.tabs([
    "📐 Summarization Engines",
    "📊 ROUGE Evaluation Mathematics",
    "📋 Model Limitations Catalog",
    "🌐 Deployment Blueprint",
    "📚 Academic References",
])

with tab_methods:
    st.markdown(
        """
        ### 1. Extractive Summarization Engine (TF-IDF Ranking)

        Extractive summarization identifies and extracts verbatim sentences from the source document that convey the highest informational salience.

        #### Mathematical Workflow:
        1. **Document Segmentation:** The document $D$ is partitioned into sentences $S = \{s_1, s_2, \dots, s_N\}$ with Devanagari danda (`।`/`॥`) support for Hindi & Marathi.
        2. **TF-IDF Transformation:** Each sentence is mapped into a vector space representation:
           $$\\text{TF}(t, s) = \\frac{f_{t, s}}{\\sum_{t' \\in s} f_{t', s}}$$
           $$\\text{IDF}(t, D) = \\log \\left( \\frac{1 + N}{1 + |\\{s \\in D : t \\in s\\}|} \\right) + 1$$
           $$\\text{TF-IDF}(t, s, D) = \\text{TF}(t, s) \\times \\text{IDF}(t, D)$$
        3. **Sentence Salience Scoring:** Sentence weight is scored based on the mean and volume of its salient non-stopword terms:
           $$\\text{Score}(s) = \\frac{1}{|s_{non-zero}|} \\sum_{t \\in s} \\text{TF-IDF}(t, s) \\times \\log(1 + |s_{non-zero}|)$$
        4. **Chronological Reconstruction:** The top $K$ scoring sentences are identified and **restored to their original sequential position** in the document to preserve logical discourse flow.

        ---

        ### 2. Abstractive Summarization Engine (Transformers)

        Abstractive summarization synthesizes novel textual expressions, paraphrasing source concepts similarly to human writers.

        #### Architecture:
        * **Encoder-Decoder Transformer:** Employs bidirectional self-attention in the encoder and causal masked self-attention with cross-attention in the decoder.
        * **Pretrained Models Supported:**
          * `sshleifer/distilbart-cnn-12-6`: Distilled student model with 306M parameters; operates with high efficiency and low memory footprint.
          * `facebook/bart-large-cnn`: 406M parameter autoregressive seq2seq model fine-tuned on news highlights.
          * `t5-small`: 60M parameter text-to-text unified Transformer.
          * `csebuetnlp/mT5_multilingual_XLSum`: Multilingual seq2seq capable of Hindi and Marathi synthesis.
        * **Deterministic Decoding:** Utilizes beam search ($num\\_beams=4$, $length\\_penalty=2.0$, $no\\_repeat\\_ngram\\_size=3$) without stochastic sampling to guarantee consistent, reproducible results.
        """
    )

with tab_rouge:
    st.markdown(
        """
        ### Standardized ROUGE Metrics Formulation

        ROUGE (Recall-Oriented Understudy for Gesting Evaluation) compares candidate generated summaries ($C$) against human reference summaries ($R$).

        #### 1. ROUGE-1 (Unigram Overlap)
        Measures individual word matching:
        $$\\text{Recall}_{R1} = \\frac{\\sum_{w \\in R} \\min(\\text{Count}(w, C), \\text{Count}(w, R))}{\\sum_{w \\in R} \\text{Count}(w, R)}$$
        $$\\text{Precision}_{R1} = \\frac{\\sum_{w \\in C} \\min(\\text{Count}(w, C), \\text{Count}(w, R))}{\\sum_{w \\in C} \\text{Count}(w, C)}$$
        $$F_1 = \\frac{2 \\times \\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$

        #### 2. ROUGE-2 (Bigram Overlap)
        Measures phrase and consecutive two-word sequence similarity:
        $$\\text{Recall}_{R2} = \\frac{\\sum_{g_2 \\in R} \\min(\\text{Count}(g_2, C), \\text{Count}(g_2, R))}{\\sum_{g_2 \\in R} \\text{Count}(g_2, R)}$$

        #### 3. ROUGE-L (Longest Common Subsequence)
        Captures structural sentence continuity without requiring consecutive matches:
        $$\\text{LCS}(C, R) = \\text{Length of Longest Common Subsequence}$$
        $$\\text{Recall}_{RL} = \\frac{\\text{LCS}(C, R)}{|R|}, \\quad \\text{Precision}_{RL} = \\frac{\\text{LCS}(C, R)}{|C|}$$
        """
    )

with tab_limits:
    st.markdown("### 📋 Model Architecture Specifications & Limitations Catalog")
    st.caption("Standardized catalog delineating paradigm, context bounds, hardware cost, and hallucination risks for every summarization engine.")

    df_limits = get_model_limitations_dataframe()
    st.dataframe(df_limits, use_container_width=True, hide_index=True)

    st.markdown("#### 🔍 Architectural Breakdown Cards")
    st.markdown(render_model_limitations_cards_html(), unsafe_allow_html=True)

with tab_deploy:
    st.markdown(
        """
        ### 🚀 Production Deployment Architecture

        TextSummarize AI is engineered for lightweight, self-contained deployment across cloud environments with 0 external API dependencies.

        #### Option A: Hugging Face Spaces (CPU Basic)
        1. Create a Space on Hugging Face with the **Streamlit** SDK.
        2. Push the repository to the Space git remote:
           ```bash
           git remote add space https://huggingface.co/spaces/YOUR_USERNAME/textsummarize-ai
           git push space main
           ```
        3. The application will initialize and run automatically.

        #### Option B: Streamlit Community Cloud
        1. Connect the GitHub repository at `share.streamlit.io`.
        2. Set entry point to `app.py`.
        3. The platform deploys on public HTTPS with auto-updates on git push.
        """
    )

with tab_refs:
    st.markdown(
        """
        ### 📚 Academic References

        1. **Lewis, M. et al. (2020).** *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL Anthology.
        2. **Lin, C.-Y. (2004).** *ROUGE: A Package for Automatic Evaluation of Summaries.* Association for Computational Linguistics.
        3. **Raffel, C. et al. (2020).** *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* Journal of Machine Learning Research (JMLR).
        4. **See, A. et al. (2017).** *Get To The Point: Summarization with Pointer-Generator Networks.* (CNN/DailyMail Benchmark).
        """
    )
