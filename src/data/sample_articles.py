"""
Curated sample articles with verified ground-truth reference summaries.
Provides instant, offline-ready test data across diverse domains.
"""

from typing import Dict, Any, List, Optional

SAMPLE_ARTICLES: List[Dict[str, Any]] = [
    {
        "id": "sample_ai_breakthrough",
        "title": "Advances in Transformer Neural Architectures for Multimodal Reasoning",
        "category": "Artificial Intelligence & Computing",
        "article": (
            "Over the past decade, sequence-to-sequence Transformer architectures have redefined natural language processing "
            "and computational linguistics. Originally introduced in 2017 with self-attention mechanisms, Transformers replaced "
            "recurrent neural networks by processing entire sequences concurrently rather than sequentially. This architectural "
            "shift eliminated gradient decay issues over long sequences and enabled unprecedented parallelization during training.\n\n"
            "Recent developments have extended Transformers beyond textual data to multimodal domains encompassing vision, "
            "audio synthesis, and sensor processing. Modern encoder-decoder models such as BART and T5 formulate diverse problems "
            "as unified sequence-to-sequence tasks. In summarization benchmarks, these models synthesize complex source materials "
            "into fluid, human-like synopses. However, challenges persist regarding computational complexity, as standard "
            "attention mechanisms scale quadratically with sequence length. Researchers are actively developing linear attention "
            "approximations, sparse attention patterns, and state-space hybrid models to mitigate latency and memory consumption, "
            "thereby broadening real-world deployment on edge computing devices and mobile hardware."
        ),
        "reference_summary": (
            "Transformer architectures revolutionized NLP through parallelized self-attention mechanisms, "
            "replacing recurrent models and expanding into multimodal domains. Models like BART and T5 excel at unified text generation "
            "such as summarization, though researchers continue addressing quadratic scaling costs using sparse attention and hybrid designs."
        ),
    },
    {
        "id": "sample_clean_energy",
        "title": "Next-Generation Perovskite Solar Cells and Grid-Scale Storage",
        "category": "Renewable Energy & Climate Science",
        "article": (
            "Global efforts to accelerate the transition toward renewable energy have ignited substantial innovation in photovoltaic "
            "materials and grid-level electrical storage systems. Traditional silicon-based solar panels have approached their theoretical "
            "efficiency threshold of approximately 29 percent under standard terrestrial conditions. In response, materials scientists "
            "have turned toward tandem perovskite-silicon solar cells, which harvest distinct wavelengths of the electromagnetic spectrum "
            "to achieve laboratory efficiencies exceeding 34 percent.\n\n"
            "Perovskites offer favorable bandgap tunability, lower manufacturing temperatures, and low-cost solution processing. "
            "Nonetheless, operational stability under moisture, ultraviolet radiation, and thermal cycling remains a hurdle for commercialization. "
            "Concurrently, the intermittent nature of solar and wind generation necessitates scalable long-duration energy storage. "
            "Utility companies are deploying sodium-ion and iron-flow batteries alongside existing lithium-ion infrastructure. "
            "These novel chemical architectures leverage abundant, non-toxic materials that reduce supply chain dependencies while providing "
            "twelve or more hours of continuous discharge capacity, stabilizing regional electricity grids during peak demand intervals."
        ),
        "reference_summary": (
            "Tandem perovskite-silicon solar cells exceed traditional silicon efficiency limits by harvesting complementary light spectrums, "
            "though moisture and heat durability challenges remain. In parallel, sodium-ion and iron-flow batteries are being deployed "
            "as abundant, long-duration energy storage solutions to balance grid intermittency."
        ),
    },
    {
        "id": "sample_biomedical",
        "title": "mRNA Platform Technologies in Preventive and Personalized Oncology",
        "category": "Biomedicine & Healthcare",
        "article": (
            "The rapid clinical deployment of messenger RNA (mRNA) vaccines against viral pathogens validated a modular and programmable "
            "paradigm for modern medicine. Following this success, biotechnology laboratories and clinical trial centers have pivoted "
            "the mRNA platform toward targeted oncology and individualized cancer immunotherapies. Unlike traditional chemotherapy, "
            "which indiscriminately damages rapidly dividing cells throughout the human body, individualized mRNA therapies stimulate "
            "the patient's own adaptive immune system to identify tumor-specific neoantigens.\n\n"
            "In this therapeutic workflow, high-throughput sequencing of a patient's biopsied tumor tissue reveals unique somatic mutations. "
            "Computational algorithms then predict which mutated peptides will elicit the strongest cytotoxic T-lymphocyte response. "
            "A customized mRNA sequence encoding these antigens is encapsulated inside lipid nanoparticles and administered to the patient. "
            "Early clinical trials in malignant melanoma and pancreatic ductal adenocarcinoma indicate statistically significant reductions "
            "in recurrence rates, marking a transformative step toward personalized preventative medicine."
        ),
        "reference_summary": (
            "The mRNA vaccine platform is now being leveraged in personalized cancer therapies by sequencing tumor biopsies to identify "
            "patient-specific neoantigens. Custom mRNA packaged in lipid nanoparticles instructs the immune system to target cancer cells, "
            "demonstrating notable reductions in recurrence in early trials."
        ),
    },
    {
        "id": "sample_space_astronomy",
        "title": "Early Universe Observations and Galaxy Formation with JWST",
        "category": "Astrophysics & Space Science",
        "article": (
            "Operating from its halo orbit around the Sun-Earth Lagrange Point 2, the James Webb Space Telescope (JWST) has transformed "
            "observational astrophysics by peering into cosmic epochs previously hidden from human view. Equipped with highly sensitive "
            "infrared spectrographs and an ultra-cooled 6.5-meter beryllium primary mirror, JWST captures ancient infrared light emitted "
            "less than 400 million years after the Big Bang, a period known as the cosmic dawn.\n\n"
            "Early data releases have presented unexpected empirical findings that challenge standard cosmological models of galaxy formation. "
            "Astronomers have observed massive, highly luminous galaxies in the high-redshift universe that appear far more evolved and "
            "chemically enriched than theoretical models predicted. These observations suggest that either stellar mass assembly proceeded "
            "at vastly higher rates during the universe's infancy, or supermassive black holes formed directly from massive primordial gas "
            "clouds without passing through initial stellar-remnant phases. Ongoing spectroscopic surveys are refining these models."
        ),
        "reference_summary": (
            "The James Webb Space Telescope's infrared observations of the early universe have revealed luminous, massive galaxies "
            "existing just 400 million years after the Big Bang. These discoveries challenge standard cosmological models by suggesting "
            "faster early star formation or direct-collapse supermassive black hole origins."
        ),
    },
]


def get_sample_articles() -> List[Dict[str, Any]]:
    """Return all curated sample articles."""
    return SAMPLE_ARTICLES


def get_sample_by_id(sample_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve sample article by its unique identifier."""
    for s in SAMPLE_ARTICLES:
        if s["id"] == sample_id:
            return s
    return None
