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
    {
        "id": "sample_hindi_space",
        "title": "चंद्रयान मिशन और भारत का अंतरिक्ष अनुसंधान (Hindi)",
        "category": "अंतरिक्ष विज्ञान (Hindi)",
        "article": (
            "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने चंद्रयान-3 मिशन के माध्यम से चंद्रमा के दक्षिणी ध्रुव पर सफल सॉफ्ट लैंडिंग "
            "करके विश्व इतिहास में एक नया कीर्तिमान स्थापित किया है। चंद्रमा का दक्षिणी ध्रुवीय क्षेत्र अत्यधिक चुनौतीपूर्ण और "
            "दुर्गम माना जाता है, क्योंकि यहाँ सूर्य का प्रकाश बहुत कम पहुँचता है और तापमान शून्य से नीचे रहता है। इस मिशन का मुख्य "
            "उद्देश्य चंद्रमा की सतह पर सुरक्षित और सुगम लैंडिंग का प्रदर्शन करना तथा रोवर प्रज्ञान के माध्यम से वैज्ञानिक प्रयोग करना था।\n\n"
            "विक्रम लैंडर और प्रज्ञान रोवर में अत्याधुनिक पेलोड शामिल थे, जिन्होंने चंद्रमा की मिट्टी की तापीय चालकता, रासायनिक संरचना "
            "और प्लाज्मा घनत्व का सटीक विश्लेषण किया। इन प्रयोगों के दौरान चंद्रमा की सतह पर सल्फर, एल्यूमीनियम, कैल्शियम, लोहा और टाइटेनियम "
            "की स्पष्ट उपस्थिति दर्ज की गई। यह ऐतिहासिक सफलता न केवल भारत के स्वदेशी अंतरिक्ष तकनीकी कौशल को सिद्ध करती है, बल्कि "
            "भविष्य के मानव अंतरिक्ष अभियानों और अंतरग्रहीय अन्वेषणों के लिए नए द्वार खोलती है। इससे वैश्विक वैज्ञानिक समुदाय को चंद्रमा की "
            "भूवैज्ञानिक संरचना को समझने में अभूतपूर्व सहायता मिली है।"
        ),
        "reference_summary": (
            "इसरो के चंद्रयान-3 मिशन ने चंद्रमा के दक्षिणी ध्रुव पर ऐतिहासिक सफल लैंडिंग की। "
            "विक्रम लैंडर और प्रज्ञान रोवर ने चंद्रमा की सतह पर सल्फर तथा अन्य प्रमुख खनिजों की उपस्थिति की पुष्टि की, "
            "जिससे भारत का तकनीकी सामर्थ्य सिद्ध हुआ और भविष्य के अंतरिक्ष अभियानों को नया मार्ग मिला।"
        ),
    },
    {
        "id": "sample_marathi_agriculture",
        "title": "महाराष्ट्रातील आधुनिक कृषी तंत्रज्ञान आणि जलसंधारण (Marathi)",
        "category": "कृषी आणि पर्यावरण (Marathi)",
        "article": (
            "महाराष्ट्राच्या ग्रामीण अर्थव्यवस्थेत शेती हा अत्यंत महत्त्वाचा घटक असून राज्यातील बहुतांश शेतकरी निसर्गाच्या लहरीपणावर अवलंबून आहेत. "
            "अलिकडच्या वर्षांत बदलते हवामान, अवकाळी पाऊस आणि दुष्काळ यांसारख्या गंभीर समस्यांमुळे शेती उत्पादन धोक्यात आले आहे. "
            "या आव्हानांवर मात करण्यासाठी आधुनिक कृषी तंत्रज्ञान आणि सूक्ष्म सिंचन पद्धतींचा अवलंब मोठ्या प्रमाणावर केला जात आहे. "
            "ठिबक सिंचन आणि तुषार सिंचनामुळे पाण्याची सुमारे ५० ते ६० टक्के बचत होते आणि पिकांची उत्पादकता लक्षणीयरीत्या वाढते.\n\n"
            "याव्यतिरिक्त, ड्रोन तंत्रज्ञानाचा वापर करून पिकांवर कीटकनाशकांची फवारणी, माती परीक्षण आणि उपग्रह आधारित पीक विमा प्रणाली "
            "शेतकऱ्यांसाठी अत्यंत फायदेशीर ठरत आहे. राज्यभरात जलसंधारणाच्या विविध योजनांद्वारे भूजल पातळी वाढवण्यासाठी शेततळी आणि सिमेंट बंधारे "
            "बांधण्यात आले आहेत. सेंद्रिय शेती आणि डिजिटल बाजारपेठेच्या माध्यमातून शेतमालाला थेट योग्य भाव मिळवून देण्याचे प्रयत्न यशस्वी होत आहेत. "
            "कृषी क्षेत्रातील हे तांत्रिक बदल ग्रामीण महाराष्ट्राला स्वावलंबी आणि समृद्ध बनवण्यासाठी अत्यंत महत्त्वपूर्ण ठरत आहेत."
        ),
        "reference_summary": (
            "महाराष्ट्रात दुष्काळ आणि हवामान बदलाच्या आव्हानांवर मात करण्यासाठी ठिबक सिंचन व आधुनिक कृषी तंत्रज्ञानाचा वापर वाढत आहे. "
            "ड्रोन फवारणी, जलसंधारण आणि डिजिटल बाजारपेठेमुळे पाण्याची बचत होऊन शेतकऱ्यांना चांगला नफा मिळत असून शेती शाश्वत होत आहे."
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
