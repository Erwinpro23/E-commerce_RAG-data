# 🛒 E-Commerce RAG Analytics

An AI-powered **Retrieval-Augmented Generation (RAG)** system for e-commerce analytics. Ask natural-language questions about your sales data and get comprehensive AI-generated insights backed by Google Gemini.

---

## ✨ Features

- **Semantic Search** – Finds the most relevant sales records using FAISS vector similarity
- **Fairness-Aware Retrieval** – Reranks results to ensure diversity across regions, categories, and segments
- **Google Gemini Integration** – Generates detailed analytical answers grounded in your data
- **Streamlit Chat UI** – Clean, modern chat interface with a dark glassmorphism design

---

## 📁 Project Structure

```
RAG(e-commerce)/
├── src/                        # Core source modules
│   ├── retriever.py            # FAISS retrieval + fairness-aware reranking
│   └── RAG_model.py              # Gemini LLM integration & prompt engineering
├── data/
│   ├── raw/
│   │   └── SampleSuperstore.csv    # Original dataset
│   └── processed/              # Generated files (gitignored — see below)
│       ├── faiss_index.bin
│       └── metadata.pkl
├── notebooks/
│   └── model.ipynb             # Data ingestion → embeddings → FAISS index
├── tests/
│   └── test_env.py             # Environment / API key sanity check
├── app.py                      # Streamlit entry point
├── documents.json              # Pre-processed document corpus
├── requirements.txt
├── .env                        # Secret keys (gitignored)
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/RAG-e-commerce.git
cd RAG-e-commerce
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
# Create .env and add your GOOGLE_API_KEY
```

### 3. Generate the FAISS index

Open and run **`notebooks/model.ipynb`** top-to-bottom. This reads `data/raw/SampleSuperstore.csv`, builds embeddings, and writes `data/processed/faiss_index.bin` + `data/processed/metadata.pkl`.

### 4. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and start asking questions! 🎉

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API key for Gemini |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector store | FAISS |
| LLM | Google Gemini 2.5 Flash |
| Frontend | Streamlit |
| Data | Superstore Sales Dataset |

---

## 📄 License

MIT — see [LICENSE](LICENSE).
