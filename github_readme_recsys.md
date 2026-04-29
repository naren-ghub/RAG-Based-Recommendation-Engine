# 🎬 FilmDB: 3-Stage RAG Recommendation Engine

<a href="YOUR_KAGGLE_NOTEBOOK_URL_HERE" target="_blank">
  <img align="left" alt="Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg" />
</a>
<br><br>

> **FilmDB** is a production-grade Retrieval-Augmented Generation (RAG) movie recommendation engine built over a curated corpus of 95,000+ films. 

Most recommendation systems rely on collaborative filtering ("users who liked X also liked Y"). They fail when a user asks for something nuanced, like: *"Recommend me an Oscar-winning film that is a beautiful musical about jazz and passion"* or *"Recommend me a gritty Indian gangster epic."* 

To solve this, FilmDB abandons keyword matching and utilizes a strict **3-Stage Semantic Pipeline** to retrieve, rerank, and reason over cinematic metadata.

---

## ⚙️ The 3-Stage RAG Pipeline

Our architecture ensures ultra-fast candidate retrieval followed by deep semantic reasoning.

### 1️⃣ Stage 1: Dense Vector Retrieval
* **Technology:** `BAAI/bge-base-en-v1.5` (768-d embeddings) + **Qdrant Cloud**
* **Process:** Every movie is converted into a dense vector representing its title, year, genre, and plot. When a user submits a query, Qdrant performs a lightning-fast Cosine Distance search alongside 14 payload filters to extract the top 50 highly relevant candidates from the 95k corpus.

### 2️⃣ Stage 2: Cross-Encoder Reranking
* **Technology:** `BAAI/bge-reranker-large`
* **Process:** Bi-encoders are fast but lack deep contextual reasoning. The top 50 candidates are fed into a heavy Cross-Encoder model. By evaluating the user's exact query and the movie's plot simultaneously, it generates raw logit scores that perfectly reorder the list, separating true matches from superficial overlaps. We keep the top 10.

### 3️⃣ Stage 3: LLM Curation & Explanation
* **Technology:** `llama-3.3-70b-versatile` (Running via **Groq API**)
* **Process:** The refined top 10 list is passed to LLaMA-70B. Acting as a senior cinematic curator, the LLM enforces a strict quality guard, selects the absolute best 5 films, and generates a personalized, table-formatted explanation of exactly *why* each film fits the user's specific request.

---

## 🛠 Project Lifecycle: Phase by Phase

This repository is divided into sequential Jupyter Notebooks, documenting the entire pipeline from raw data to the interactive UI.

### Phase 1: Data Curation (`01_data_curation.ipynb`)
We started with a massive raw dataset of 347,000+ movies. We applied strict quality filters (deduplication, removing entries with 0 votes, filtering extreme outliers) to curate a high-quality, dense corpus of **95,090 films**. Metadata strings were cleaned and normalized for the embedding pipeline.

### Phase 2: Embedding Generation (`02_embedding_generation.ipynb` & `03_qdrant_indexing.ipynb`)
Each movie's metadata was injected into a highly engineered text template. Using the BGE-base model, we generated 95,090 float32 vector embeddings. These vectors, along with payload metadata (like `imdb_rating`, `director`, and `original_language`), were uploaded to a Qdrant Cloud cluster to establish our searchable Vector DB.

### Phase 3: RAG Evaluation (`05_rag_evaluation.ipynb`)
To validate the pipeline, we designed a **Retrieval Strategy Catalog** consisting of 9 distinct strategies (e.g., *S2_Prestige*, *S6_Director*, *S8_Mood*) and 21 highly specific semantic queries. 
* We executed this heavy evaluation pipeline on a **Kaggle T4 GPU**. 
* The system logged the results of Stage 1 (Dense), Stage 2 (Rerank), and Stage 3 (LLM) for every query.
* The final results were exported as a static JSON artifact (`showcase_results.json`).

### Phase 4: The Interactive Showcase (`06_showcase.ipynb`)
Because the heavy GPU computation was completed in Phase 3, we built an interactive dashboard that runs entirely on **CPU**. By loading the `showcase_results.json` dataset, the Showcase notebook uses `ipywidgets` and `plotly` to render an interactive dropdown UI. Users can explore Head-to-Head pipeline comparisons, latency metrics, and reranker quality lift charts without needing an API key or GPU.

---

## 🚀 Explore the Showcase
Want to see how the engine handles queries like *"Recommend me a quiet, slice-of-life story about a family in Tokyo"*?

Explore the interactive evaluation pipeline live on Kaggle. **(No GPU or API keys required!)**

🔗 **[Launch the Interactive Kaggle Showcase](YOUR_KAGGLE_NOTEBOOK_URL_HERE)**
