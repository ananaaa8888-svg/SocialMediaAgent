# Social Media Monitoring Agent

Ολοκληρωμένο σύστημα ανάλυσης κοινωνικών δικτύων το οποίο συνδυάζει **ανάκτηση πληροφορίας (Information Retrieval)**, **ανάλυση συναισθήματος πολλαπλών ετικετών (Multi-label Emotion Classification)**, **θεματική ομαδοποίηση (Topic Clustering)** και **αυτόματη παραγωγή συνόψεων (Automatic Summarization)**, με προβολή των αποτελεσμάτων σε ένα διαδραστικό web dashboard.

Η εργασία υλοποιήθηκε ως σύνολο έξι Jupyter notebooks (που αντιστοιχούν στα στάδια του pipeline) και μιας web εφαρμογής (FastAPI backend + χειρόγραφο static frontend), η οποία αξιοποιεί τα προεπεξεργασμένα δεδομένα και τα ευρετήρια που παράγουν τα notebooks.

---

## 1. Επισκόπηση του Pipeline

Το σύστημα ακολουθεί έξι διαδοχικά στάδια, καθένα από τα οποία παράγει artefacts που χρησιμοποιούνται από το επόμενο:

```
                  ┌────────────────────────────────────────────────┐
                  │                Query από τον χρήστη            │
                  └───────────────────────┬────────────────────────┘
                                          │
[1] Data Ingestion          → SemEval-2018 Task 1 E-c (11 emotions, 10.983 tweets)
                                          │
[2] Preprocessing           → TweetTokenizer · emoji demojize · MD5 deduplication
                                          │                       (text_clean / text_light)
[3] Retrieval               → BM25 (rank-bm25) · Semantic (FAISS + MiniLM) · Hybrid fusion
                                          │
[4] Classification          → Logistic Regression · Linear SVM · Twitter-RoBERTa
                                          │
[5] Aggregation             → K-Means · UMAP 2D · aspect-oriented grouping
                                          │
[6] Report Generation       → Extractive (centroid-nearest) · Abstractive (DistilBART)
                                          │
                  ┌───────────────────────▼────────────────────────┐
                  │ Web Dashboard (FastAPI + ECharts + vanilla JS) │
                  └────────────────────────────────────────────────┘
```

Η ίδια αρχιτεκτονική επαναχρησιμοποιείται στο `app/engine.py`, με αποτέλεσμα ό,τι παρουσιάζεται στα notebooks (ως πειραματικό περιβάλλον) να εκτελείται σε πραγματικό χρόνο και μέσω του dashboard.

---

## 2. Στοίβα Τεχνολογιών

Παρατίθενται όλες οι βιβλιοθήκες, τα μοντέλα και τα εργαλεία που χρησιμοποιούνται στο έργο:

### 2.1 Δεδομένα και Διαχείριση

| Συνιστώσα | Χρήση |
|---|---|
| `datasets` (HuggingFace) | Φόρτωση του SemEval-2018 Task 1 E-c (`subtask5.english`) |
| `pandas` 2.2.3 | Επεξεργασία πινάκων, in-memory analytics |
| `pyarrow` 18.1.0 | Σειριοποίηση σε στήλεα μορφή Parquet |
| `numpy` | Αριθμητικοί τανυστές, διανυσματικές πράξεις |

### 2.2 Προεπεξεργασία Κειμένου

| Συνιστώσα | Χρήση |
|---|---|
| `nltk` 3.9.1 (TweetTokenizer, stopwords) | Tokenization προσαρμοσμένο σε tweets |
| `emoji` 2.14.1 | Μετατροπή emojis σε λεκτική περιγραφή (`demojize`) |
| `tweet-preprocessor` 0.6.0 | Βοηθητικός καθαρισμός social media κειμένου |
| `hashlib` (MD5) | Εντοπισμός ακριβών διπλότυπων |

### 2.3 Ανάκτηση Πληροφορίας

| Συνιστώσα | Χρήση |
|---|---|
| `rank-bm25` | Υλοποίηση `BM25Okapi` (sparse keyword retrieval) |
| `sentence-transformers` 3.3.1 | Μοντέλο `all-MiniLM-L6-v2` (384-διάστατα embeddings) |
| `faiss-cpu` 1.9.0.post1 | `IndexFlatIP` για brute-force inner-product search |

### 2.4 Ταξινόμηση Συναισθημάτων

| Συνιστώσα | Χρήση |
|---|---|
| `scikit-learn` ≥ 1.4 | `TfidfVectorizer`, `OneVsRestClassifier`, `LogisticRegression`, `LinearSVC` |
| `transformers` 4.47.1 | Φόρτωση/inference του `cardiffnlp/twitter-roberta-base-emotion-multilabel-latest` |
| `torch` 2.5.1 | Tensor backend (υποστήριξη CUDA) |

### 2.5 Ομαδοποίηση και Μείωση Διαστάσεων

| Συνιστώσα | Χρήση |
|---|---|
| `scikit-learn` (KMeans, silhouette_score) | Clustering και επιλογή optimal k |
| `umap-learn` ≥ 0.5.6 | Προβολή 384D → 2D για οπτικοποίηση |
| `hdbscan` ≥ 0.8.38 | Εναλλακτικό density-based clustering (διαθέσιμο) |

### 2.6 Παραγωγή Συνόψεων

| Συνιστώσα | Χρήση |
|---|---|
| `sklearn.metrics.pairwise.cosine_similarity` | Centroid-nearest extractive summarisation |
| `sshleifer/distilbart-cnn-12-6` (≈ 600 MB) | Abstractive summarisation (offline, χωρίς API) |

### 2.7 Web Εφαρμογή

| Συνιστώσα | Χρήση |
|---|---|
| `fastapi` ≥ 0.115 | REST API (`/api/search`, `/api/summary`, `/api/health`) |
| `uvicorn[standard]` ≥ 0.32 | ASGI server |
| `pydantic` | Validation των request schemas |
| Χειρόγραφο HTML/CSS/JS | Frontend χωρίς frameworks (React/Vue/Streamlit) |
| Apache **ECharts** 5.5.1 | Διαδραστικά διαγράμματα (pie, bar, heatmap, scatter, radar) |
| Inter & JetBrains Mono | Τυπογραφία |

### 2.8 Οπτικοποίηση στα Notebooks

| Συνιστώσα | Χρήση |
|---|---|
| `matplotlib` ≥ 3.9 | Στατικά διαγράμματα (`.png`) |
| `seaborn` ≥ 0.13 | Heatmaps, palettes |
| `plotly` ≥ 5.24 | Διαδραστικά διαγράμματα εντός notebooks |

### 2.9 Περιβάλλον Εκτέλεσης

| Συνιστώσα | Χρήση |
|---|---|
| `jupyter` 1.1.1, `ipykernel` 6.29.5, `ipywidgets` ≥ 8.1 | Notebook environment |
| Python **3.11** | Γλώσσα ανάπτυξης |

---

## 3. Δομή του Repository

```
SocialMediaAgent/
├── data/                                ← παράγεται από τα notebooks (δεν είναι σε git)
│   ├── semeval2018/                     ← raw parquet (train/val/test/all)
│   ├── processed/                       ← καθαρισμένα tweets (text_clean, text_light) + plots
│   ├── index/                           ← bm25.pkl, embeddings.npy, faiss.index
│   ├── models/                          ← lr_clf.pkl, svm_clf.pkl, tfidf_*.pkl
│   ├── aggregation/                     ← retrieved_aggregated.parquet + cluster plots
│   └── report/                          ← extractive_summaries.parquet, report.html, JSON
│
├── notebooks/                           ← Πειραματικό pipeline (Jupyter)
│   ├── 01_data_ingestion.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_retrieval.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_aggregation.ipynb
│   └── 06_report_dashboard.ipynb
│
├── app/                                 ← Production web εφαρμογή
│   ├── server.py                        ← FastAPI backend (REST API)
│   ├── engine.py                        ← Inference layer (retrieval/sentiment/clustering)
│   └── static/                          ← Custom frontend
│       ├── index.html
│       ├── styles.css
│       └── app.js
│
├── REPORT.md                            ← Ακαδημαϊκή αναφορά του έργου
├── README.md                            ← Το παρόν αρχείο
└── requirements.txt                     ← Εξαρτήσεις
```

---

## 4. Οδηγός Εγκατάστασης και Εκτέλεσης

Η εκτέλεση του έργου χωρίζεται σε **δύο φάσεις** που πρέπει να εκτελεστούν με τη σωστή σειρά:

> **Φάση Α — Παραγωγή των δεδομένων και των ευρετηρίων** μέσω των notebooks (γίνεται μία φορά).
>
> **Φάση Β — Εκκίνηση του web dashboard** για διαδραστική χρήση.

### 4.1 Προαπαιτούμενα

Παρακαλούμε εξασφαλίστε τα ακόλουθα πριν προχωρήσετε:

- **Python 3.11** (η ίδια έκδοση που χρησιμοποιήθηκε για την ανάπτυξη)
- **Σύνδεση στο διαδίκτυο** κατά την πρώτη εκτέλεση (για κατέβασμα των μοντέλων από το HuggingFace Hub)
- **8 GB ελεύθερης μνήμης RAM** για την παράλληλη φόρτωση όλων των πόρων
- **(Προαιρετικά)** NVIDIA GPU με υποστήριξη CUDA για ταχύτερη εκτέλεση των transformers

### 4.2 Βήμα 1 — Κλωνοποίηση και Δημιουργία Virtual Environment

```bash
git clone <repo-url>
cd SocialMediaAgent

# Δημιουργία απομονωμένου περιβάλλοντος
python3.11 -m venv venv

# Ενεργοποίηση (Linux / macOS)
source venv/bin/activate

# Ενεργοποίηση (Windows)
venv\Scripts\activate
```

### 4.3 Βήμα 2 — Εγκατάσταση των Εξαρτήσεων

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Η εγκατάσταση περιλαμβάνει βαρέα πακέτα (PyTorch, FAISS, Transformers) και μπορεί να διαρκέσει μερικά λεπτά.

### 4.4 Βήμα 3 — Εκτέλεση των Notebooks (Φάση Α)

> Σημείωση: Ο φάκελος `data/` **δεν** συμπεριλαμβάνεται στο git. Σε ένα φρέσκο `git clone`, τα αρχεία δεδομένων δεν υπάρχουν, οπότε η web εφαρμογή θα επιστρέψει σφάλμα *«Required data files are missing…»* μέχρι να εκτελέσετε τα notebooks (ή να αντιγράψετε έναν προ-υπάρχοντα φάκελο `data/`).

Εκκινήστε το Jupyter:

```bash
jupyter notebook
```

και εκτελέστε τα notebooks **σειριακά** (το καθένα προϋποθέτει τα artefacts του προηγουμένου):

| Σειρά | Notebook | Παραγόμενα Artefacts | Σκοπός |
|:-:|---|---|---|
| 1 | `01_data_ingestion.ipynb` | `data/semeval2018/{train,validation,test,all}.parquet` και τρία διαγράμματα EDA στο `data/processed/` | Κατέβασμα του SemEval-2018 Task 1 E-c, εξερευνητική ανάλυση, αποθήκευση σε Parquet |
| 2 | `02_preprocessing.ipynb` | `data/processed/tweets_processed.parquet`, `{train,validation,test}_processed.parquet`, `token_length_dist.png`, `top_tokens.png` | Δύο παραλλαγές καθαρισμού (`text_clean` για TF-IDF/BM25, `text_light` για RoBERTa/MiniLM), MD5 deduplication |
| 3 | `03_retrieval.ipynb` | `data/index/bm25.pkl`, `embeddings.npy` (≈ 16 MB), `faiss.index` (≈ 16 MB), `emotion_dist_retrieved.png` | Κατασκευή ευρετηρίων BM25 και FAISS, υλοποίηση τριών στρατηγικών αναζήτησης, αξιολόγηση με Precision@k, Recall@k, nDCG@k |
| 4 | `04_classification.ipynb` | `data/models/{lr_clf,svm_clf,tfidf_word,tfidf_char}.pkl`, `per_emotion_f1_comparison.png` | Εκπαίδευση LR & SVM σε TF-IDF (word 1-2 + char 3-5), zero-shot inference με Twitter-RoBERTa, συγκριτική αξιολόγηση |
| 5 | `05_aggregation.ipynb` | `data/aggregation/retrieved_aggregated.parquet` και έξι διαγράμματα (`elbow_silhouette.png`, `umap_clusters.png`, `emotion_per_cluster.png`, `emotion_per_aspect_*`, `umap_cluster_vs_aspect.png`) | K-Means με επιλογή optimal k μέσω silhouette, UMAP 2D προβολή, αυτόματη εξαγωγή aspects με top TF-IDF terms |
| 6 | `06_report_dashboard.ipynb` | `data/report/{polarity_distribution,polarity_emotion_heatmap}.png`, `extractive_summaries.parquet`, `report_summary.json`, `report.html` | Αντιστοίχιση συναισθημάτων σε πολικότητα, extractive σύνοψη (centroid-nearest), abstractive σύνοψη (DistilBART), τελικός αναφορά HTML |

> **Κρίσιμη ελάχιστη προϋπόθεση για το app**: αρκούν τα notebooks **02 και 03**, καθώς το app χρειάζεται ελάχιστα τα παρακάτω αρχεία:
>
> - `data/processed/tweets_processed.parquet` (από το NB 02)
> - `data/index/bm25.pkl`, `data/index/embeddings.npy`, `data/index/faiss.index` (από το NB 03)

### 4.5 Βήμα 4 — Εκκίνηση του Web Dashboard (Φάση Β)

Αφού ολοκληρωθεί η Φάση Α, εκκινήστε τον FastAPI server:

```bash
# Μέθοδος 1 — με auto-reload για ανάπτυξη
uvicorn app.server:app --reload

# Μέθοδος 2 — απευθείας εκτέλεση
python -m app.server
```

Στη συνέχεια, ανοίξτε στον browser τη διεύθυνση:

- **Dashboard**: <http://127.0.0.1:8000>
- **Διαδραστικό API documentation (Swagger UI)**: <http://127.0.0.1:8000/api/docs>

---

## 5. Χρήση του Dashboard

Το dashboard διαρθρώνεται σε **sidebar** (παράμετροι αναζήτησης) και **main area** (πέντε tabs αποτελεσμάτων):

### 5.1 Παράμετροι Αναζήτησης (Sidebar)

| Πεδίο | Περιγραφή |
|---|---|
| **Query** | Ελεύθερο κείμενο (π.χ. `vaccines`, `bitcoin`, `education technology ai`) |
| **Retriever** | `Hybrid` (συνιστώμενο), `BM25` (sparse keywords), `Semantic` (FAISS + MiniLM) |
| **Hybrid α** | Συντελεστής βαρύτητας του BM25 (μόνο σε hybrid mode), προεπιλογή `0.5` |
| **Top-K results** | Πλήθος επιστρεφόμενων tweets (50–500), προεπιλογή `200` |

Πατώντας **▶ Run analysis** στέλνεται αίτημα `POST /api/search` με τις παραπάνω παραμέτρους.

### 5.2 Tabs Αποτελεσμάτων

1. **Sentiment Distribution** — pie chart και bar chart πολικότητας, heatmap (polarity × emotion), bar chart με τη συνολική ένταση συναισθημάτων.
2. **Retrieved Tweets** — διαδραστικός πίνακας με δυνατότητα φιλτραρίσματος ανά πολικότητα και συναίσθημα.
3. **Topic Clusters** — UMAP scatter plot των retrieved tweets, λίστα με τους κορυφαίους όρους ανά cluster, εναλλαγή χρωματισμού (cluster / polarity).
4. **Aspect Analysis** — bar chart πολικοτήτων ανά aspect, radar chart συναισθημάτων, heatmap, αντιπροσωπευτικά tweets.
5. **Summary Report** — extractive σύνοψη ανά πολικότητα (πάντα διαθέσιμη) και κουμπί παραγωγής abstractive συνόψεων με DistilBART κατ' απαίτηση.

> **Σημείωση για την abstractive σύνοψη**: Την πρώτη φορά που πατάτε «Generate Abstractive Summaries», το μοντέλο DistilBART (≈ 600 MB) κατεβαίνει και φορτώνεται στη μνήμη. Η διαδικασία απαιτεί λίγη ώρα, αλλά το μοντέλο παραμένει σε memory cache για τις επόμενες κλήσεις.

---

## 6. REST API

Ο FastAPI server εκθέτει τέσσερα endpoints:

| Μέθοδος | Διαδρομή | Περιγραφή |
|:-:|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/search` | Εκτέλεση retrieval + sentiment + clustering και επιστροφή JSON payload |
| `POST` | `/api/summary` | Παραγωγή abstractive συνόψεων για ένα προηγούμενο `result_id` (LRU cache 8 entries) |
| `GET` | `/` | Σερβίρισμα του `index.html` |

Παραδείγματα body για το `/api/search`:

```json
{
  "query": "education technology ai",
  "mode": "hybrid",
  "alpha": 0.5,
  "top_k": 200
}
```

---

## 7. Σημαντικές Σχεδιαστικές Επιλογές

Παρακαλούμε λάβετε υπόψη τα ακόλουθα κατά την αξιολόγηση:

- **Decoupling πειραματικού και παραγωγικού περιβάλλοντος**: Τα notebooks αποτελούν αυτοτελή ερευνητικά κείμενα, ενώ το `app/engine.py` αναπαράγει την ίδια λογική σε καθαρή, thread-safe μορφή κατάλληλη για web inference.
- **Lazy loading**: Όλοι οι βαριοί πόροι (corpus, BM25, FAISS, MiniLM, DistilBART) φορτώνονται **την πρώτη φορά που απαιτηθούν**, μέσω singleton pattern με `threading.Lock`.
- **LRU result cache**: Τα αποτελέσματα αναζήτησης κρατούνται σε cache 8 εγγραφών (`uuid` ως κλειδί), ώστε το `/api/summary` να μην επαναλαμβάνει retrieval.
- **Διαχωρισμός `text_clean` / `text_light`**: Διαφορετικά μοντέλα έχουν διαφορετικές ανάγκες — τα κλασικά ML μοντέλα και ο BM25 απαιτούν επιθετικό καθαρισμό, ενώ τα transformer-based μοντέλα ωφελούνται από τη διατήρηση hashtags και emojis.
- **Πρόληψη data leakage**: Στο notebook 04, ο `TfidfVectorizer` εκπαιδεύεται **αποκλειστικά** στο train split.
- **Κατασκευή frontend από την αρχή**: Δεν χρησιμοποιείται React, Vue, Streamlit, Tailwind ή κάποιο έτοιμο template. Το UI υλοποιήθηκε από την αρχή σε vanilla JavaScript με Apache ECharts για τα διαγράμματα.

---

## 8. Αναφορές για Περαιτέρω Μελέτη

Στο repository θα βρείτε ένα συμπληρωματικό κείμενο:

- **`REPORT.md`** — Ακαδημαϊκή αναφορά (Abstract, Related Work, Δεδομένα, Μεθοδολογία, Αποτελέσματα, Συζήτηση, Βιβλιογραφία).

---

