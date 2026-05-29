# Social Media Monitoring Agent

Ένα end-to-end σύστημα ανάλυσης κοινωνικών δικτύων που συνδυάζει **ανάκτηση πληροφορίας**, **ανάλυση συναισθήματος**, **topic clustering** και **αυτόματη σύνοψη** σε ένα interactive dashboard.

---

## Τεχνολογίες & Pipeline

```
Query Input
    ↓
[1] Data Ingestion       → SemEval-2018 Task 1 E-c (11 emotions, ~11k tweets)
    ↓
[2] Preprocessing        → TweetTokenizer · emoji demojize · deduplication
    ↓
[3] Retrieval            → BM25 (rank-bm25) · Semantic (FAISS + MiniLM) · Hybrid fusion
    ↓
[4] Classification       → Logistic Regression · Linear SVM · Twitter-RoBERTa
    ↓
[5] Aggregation          → K-Means · UMAP · aspect-oriented grouping
    ↓
[6] Report               → Extractive (centroid-nearest) · Abstractive (DistilBART)
    ↓
    Custom Web Dashboard (FastAPI + vanilla JS/CSS)
```

| Κατηγορία | Εργαλεία |
|---|---|
| Dataset | SemEval-2018 Task 1 E-c (`datasets` / HuggingFace) |
| Preprocessing | NLTK TweetTokenizer, `emoji`, `tweet-preprocessor` |
| Retrieval | `rank-bm25`, `sentence-transformers` (all-MiniLM-L6-v2), `faiss-cpu` |
| Classification | `scikit-learn` (LR, LinearSVC), `cardiffnlp/twitter-roberta-base-emotion-multilabel-latest` |
| Clustering | K-Means (`scikit-learn`), UMAP (`umap-learn`) |
| Summarisation | `sshleifer/distilbart-cnn-12-6` (offline, ~600 MB) |
| Dashboard | `fastapi` + `uvicorn` (backend) · custom HTML/CSS/JS + Apache ECharts (frontend) — χωρίς έτοιμα templates/themes ή paid frameworks |
| Deep Learning | `transformers`, `torch` (CUDA-compatible) |

---

## Εκκίνηση της Εφαρμογής

### Προαπαιτούμενα

- Python 3.11
- *(Προαιρετικό)* NVIDIA GPU με CUDA για ταχύτερο inference

### 1. Κλωνοποίηση & εγκατάσταση

```bash
git clone <repo-url>
cd SocialMediaAgent

python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Δημιουργία των data artefacts (ΑΠΑΡΑΙΤΗΤΟ πριν το app)

> ⚠️ Ο φάκελος `data/` **δεν** ανεβαίνει στο git (είναι στο `.gitignore`). Σε ένα
> φρέσκο `git clone` τα αρχεία δεδομένων **δεν υπάρχουν**, οπότε το app θα
> επιστρέψει σφάλμα *«Required data files are missing…»* μέχρι να τα παράξεις.

Πριν τρέξεις το dashboard, εκτέλεσε **σειριακά** τα notebooks `01` → `06`
(βλ. ενότητα [Εκτέλεση των Notebooks](#εκτέλεση-των-notebooks) παρακάτω). Το app
χρειάζεται κατ’ ελάχιστον τα εξής, που παράγονται από αυτά:

```
data/processed/tweets_processed.parquet   (notebook 02)
data/index/bm25.pkl                        (notebook 03)
data/index/embeddings.npy                  (notebook 03)
data/index/faiss.index                     (notebook 03)
```

> Εναλλακτικά, αν τα έχεις ήδη παράξει σε άλλο μηχάνημα, αντίγραψε ολόκληρο τον
> φάκελο `data/` εκεί.

### 3. Εκκίνηση του Dashboard

```bash
uvicorn app.server:app --reload
# ή απλά:
python -m app.server
```

Άνοιξε στον browser το **http://127.0.0.1:8000**
(το `--reload` είναι προαιρετικό, για development). Το API documentation
είναι διαθέσιμο στο **/api/docs**.

### 4. Χρήση

1. Πληκτρολογήστε ένα query στο sidebar (π.χ. `vaccines`, `bitcoin`, `education`)
2. Επιλέξτε retriever: **Hybrid** (προτεινόμενο), BM25, ή Semantic
3. Πατήστε **▶ Run**
4. Εξερευνήστε τα αποτελέσματα στα 5 tabs:
   - **Sentiment Distribution** — κατανομή polarities & emotion heatmap
   - **Retrieved Tweets** — filterable πίνακας με emotion labels
   - **Topic Clusters** — UMAP scatter plot των retrieved tweets
   - **Aspect Analysis** — radar chart & grouped bars ανά θεματική
   - **Summary Report** — extractive σύνοψη + abstractive (DistilBART, on-demand)

> **Σημείωση για το abstractive summary**: Η πρώτη φορά που πατάτε «Generate Abstractive Summaries» φορτώνει το μοντέλο DistilBART (~600 MB). Απαιτεί λίγη ώρα· το μοντέλο παραμένει στη μνήμη για τις επόμενες εκτελέσεις.

---

## Εκτέλεση των Notebooks

Τα notebooks εκτελούνται **σειριακά** (κάθε ένα παράγει αρχεία που χρησιμοποιεί το επόμενο):

```bash
source venv/bin/activate
jupyter notebook
```

| # | Notebook | Περιγραφή |
|---|---|---|
| 01 | `01_data_ingestion.ipynb` | Φόρτωση SemEval-2018, EDA, αποθήκευση parquet |
| 02 | `02_preprocessing.ipynb` | Καθαρισμός κειμένου, deduplication |
| 03 | `03_retrieval.ipynb` | BM25 + FAISS index, αξιολόγηση retrieval |
| 04 | `04_classification.ipynb` | Εκπαίδευση LR/SVM, inference RoBERTa |
| 05 | `05_aggregation.ipynb` | Clustering, aspect analysis, DistilBART |
| 06 | `06_report_dashboard.ipynb` | Extractive + abstractive report per polarity |

---

## Δομή Αρχείων

```
SocialMediaAgent/
├── data/
│   ├── semeval2018/        ← raw parquet (train/val/test)
│   ├── processed/          ← καθαρισμένα tweets (text_clean, text_light)
│   ├── index/              ← BM25 index, FAISS index, embeddings.npy
│   ├── models/             ← LR & SVM classifiers, TF-IDF vectorizers
│   ├── aggregation/        ← retrieved + clustered tweets
│   └── report/             ← extractive/abstractive summaries, HTML report
├── notebooks/              ← 01–06 Jupyter notebooks
├── app/
│   ├── server.py           ← FastAPI backend (JSON API + serves frontend)
│   ├── engine.py           ← inference layer (retrieval/sentiment/clustering/summary)
│   └── static/             ← custom frontend (index.html · styles.css · app.js)
└── requirements.txt
```
# SocialMediaAgent
