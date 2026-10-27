"""
Page 4: About & Academic Documentation
Comprehensive technical documentation, mathematical formulations, and free deployment guides.
"""

import streamlit as st

st.set_page_config(
    page_title="About & Architecture | TextSummarize AI",
    page_icon="ℹ️",
    layout="wide",
)

st.title("ℹ️ Architecture, Methodology & Documentation")
st.caption("Theoretical foundations, mathematical formulations, and engineering principles behind TextSummarize AI.")

# Tabs for structured reading
tab_methods, tab_rouge, tab_deploy, tab_refs = st.tabs([
    "📐 Summarization Engines",
    "📊 ROUGE Evaluation Mathematics",
    "🌐 100% Free Deployment Guide",
    "📚 References & Limitations",
])

with tab_methods:
    st.markdown(
        """
        ### 1. Extractive Summarization Engine (TF-IDF Ranking)

        Extractive summarization identifies and extracts verbatim sentences from the source document that convey the highest informational salience.

        #### Mathematical Workflow:
        1. **Document Segmentation:** The document $D$ is partitioned into sentences $S = \{s_1, s_2, \dots, s_N\}$.
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
          * `sshleifer/distilbart-cnn-12-6`: Distilled student model with 306M parameters; operates with 6x lower latency and low memory footprint (ideal for free-tier CPU hosting).
          * `facebook/bart-large-cnn`: 406M parameter autoregressive seq2seq model fine-tuned on news highlights.
          * `t5-small`: 60M parameter text-to-text unified Transformer.
        * **Deterministic Decoding:** Utilizes beam search ($num\\_beams=4$, $length\\_penalty=2.0$) without stochastic sampling to guarantee consistent, reproducible results.
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

with tab_deploy:
    st.markdown(
        """
        ### 🚀 100% Free Hosting & Deployment Guide

        This application has been engineered to deploy completely free with zero subscription costs.

        #### Option A: Hugging Face Spaces (Recommended for ML/NLP - 16GB Free RAM)
        1. Create a free account at [huggingface.co](https://huggingface.co).
        2. Click **New Space** → Name it `textsummarize-ai`.
        3. Select **Streamlit SDK** and **Free CPU Basic (2 vCPU, 16GB RAM)**.
        4. Clone the space repository or connect your GitHub repository.
        5. Push the code:
           ```bash
           git remote add space https://huggingface.co/spaces/YOUR_USERNAME/textsummarize-ai
           git push space main
           ```
        6. Hugging Face automatically installs `requirements.txt` and starts `app.py`!

        #### Option B: Streamlit Community Cloud (1-Click GitHub Deploy)
        1. Push this repository to your GitHub profile.
        2. Navigate to [share.streamlit.io](https://share.streamlit.io).
        3. Click **New App** → Select your repository `textsummarize-ai` → Main file: `app.py`.
        4. Click **Deploy**. Your app is live with a permanent public HTTPS URL at no cost.
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

        ---

        ### ⚠️ System Limitations & Ethical Considerations

        1. **Lexical vs. Semantic Equivalence:** ROUGE evaluates surface n-gram overlap. Paraphrased statements expressing identical facts with alternate synonyms may yield lower ROUGE scores despite being accurate summaries.
        2. **Context Window Bounds:** Standard Transformers possess input limits (1024 tokens for BART). Extended documents are truncated to the initial 1024 tokens.
        3. **Hallucination Risk:** While deterministic beam search mitigates stochastic noise, abstractive models can occasionally synthesize unsupported assertions.
        """
    )
