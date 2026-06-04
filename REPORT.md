# Social Media Monitoring Agent

---

## Περίληψη

Η παρούσα εργασία παρουσιάζει τον σχεδιασμό και την υλοποίηση ενός end-to-end συστήματος παρακολούθησης κοινωνικών δικτύων, το οποίο συνδυάζει τεχνικές information retrieval, multi-label emotion classification, unsupervised clustering και automatic text summarization. Το σύστημα δέχεται ως είσοδο ένα ελεύθερο ερώτημα χρήστη και επιστρέφει τα πλέον σχετικά tweets από το corpus SemEval-2018 Task 1 E-c (10.896 tweets, 11 κατηγορίες συναισθήματος), μαζί με ανάλυση polarity, οπτικοποίηση θεματικών ομάδων και αυτόματη δημιουργία συνόψεων. Αξιολογούνται συγκριτικά τρεις στρατηγικές retrieval (BM25, semantic search, hybrid) και τρία μοντέλα emotion classification (Logistic Regression, Linear SVM, Twitter-RoBERTa). Τα αποτελέσματα αναδεικνύουν την υπεροχή του hybrid retrieval και του pre-trained Transformer για τα αντίστοιχα tasks. Το σύστημα προσφέρεται μέσω interactive web dashboard υλοποιημένου με FastAPI και vanilla JavaScript.

---

## 1. Εισαγωγή

Η εκρηκτική ανάπτυξη των κοινωνικών δικτύων έχει δημιουργήσει τεράστιες ποσότητες κειμένου με έντονη συναισθηματική φόρτιση. Η αυτόματη κατανόηση και ανάλυση αυτού του περιεχομένου αποτελεί κεντρικό ζήτημα στο Natural Language Processing (NLP), με εφαρμογές σε brand monitoring, ανάλυση κοινής γνώμης, έγκαιρη ανίχνευση κρίσεων και ακαδημαϊκή έρευνα.

Τα tweets παρουσιάζουν ιδιαίτερες προκλήσεις σε σχέση με τυπικά κείμενα: αυστηρό όριο χαρακτήρων, συχνή χρήση hashtags, mentions, emojis, abbreviations, ορθογραφικά λάθη και ρευστή γραμματική. Παράλληλα, ένα tweet μπορεί να εκφράζει ταυτόχρονα πολλαπλά συναισθήματα (multi-label), καθιστώντας μη κατάλληλες τις κλασικές single-label μεθόδους ταξινόμησης.

Η αρχιτεκτονική βασίστηκε σε pipeline έξι σταδίων που αντιμετωπίζει τις παρακάτω προκλήσεις:

1. Φόρτωση και εξερεύνηση δεδομένων (Data Ingestion)
2. Προεπεξεργασία κειμένου (Text Preprocessing)
3. Ανάκτηση πληροφορίας (Information Retrieval)
4. Multi-label emotion classification
5. Clustering και θεματική ανάλυση (Aggregation)
6. Δημιουργία αναφοράς και σύνοψης (Report Generation)

---

## 2. Σχετική Βιβλιογραφία

### 2.1 Ανάλυση Συναισθήματος σε Tweets

Η sentiment analysis σε tweets αντιμετωπίστηκε αρχικά ως πρόβλημα τριών κλάσεων (θετικό / αρνητικό / ουδέτερο). Το SemEval workshop εισήγαγε ετησίως νέα benchmarks, με το Task 1 του 2018 να αποτελεί σταθμό για την multi-label emotion classification λόγω της χρήσης 11 διακριτών emotion categories και της υψηλής ποιότητας annotations.

### 2.2 Μοντέλα Information Retrieval

Το BM25 (Robertson et al., 1995) παραμένει ισχυρό baseline για keyword-based retrieval. Η ανάπτυξη των sentence embeddings (Reimers & Gurevych, 2019) και η δημιουργία αποδοτικών vector indices (Johnson et al., 2019, FAISS) επέτρεψαν τη semantic search σε large-scale corpora. Το hybrid retrieval, που συνδυάζει sparse και dense representations, έχει δείξει βελτιωμένη απόδοση σε σύγκριση με τις επιμέρους μεθόδους.

### 2.3 Pre-trained Transformers για Ανάλυση Tweets

Το BERT (Devlin et al., 2019) και οι παράγωγοί του αποτελούν πλέον το state-of-the-art για text classification. Ειδικά για tweets, το Twitter-RoBERTa (Barbieri et al., 2020) από το Cardiff NLP lab εκπαιδεύτηκε σε εκατομμύρια tweets και fine-tuned για emotion classification στο TweetEval benchmark.

### 2.4 Automatic Text Summarization

Οι extractive μέθοδοι επιλέγουν αντιπροσωπευτικές προτάσεις από το corpus (Mihalcea & Tarau, 2004). Οι abstractive μέθοδοι παράγουν νέο κείμενο χρησιμοποιώντας seq2seq μοντέλα. Το DistilBART (Shleifer & Rush, 2020) αποτελεί compressed έκδοση του BART που διατηρεί υψηλή απόδοση με μειωμένες υπολογιστικές απαιτήσεις.

---

## 3. Dataset: SemEval-2018 Task 1 E-c

### 3.1 Περιγραφή

Το σύστημα χρησιμοποιεί το **SemEval-2018 Task 1 Subtask E-c** (Affect in Tweets, Mohammad et al., 2018), ένα από τα πιο ευρέως χρησιμοποιούμενα benchmarks για multi-label emotion classification σε αγγλικά tweets. Κάθε δείγμα αποτελείται από ένα tweet και 11 binary labels, που υποδηλώνουν την παρουσία ή απουσία κάθε συναισθήματος.

### 3.2 Emotion Categories

Τα 11 συναισθήματα διακρίνονται σε τρεις ομάδες polarity:

| Polarity | Emotions |
|---|---|
| Positive | joy, love, optimism, anticipation, trust |
| Negative | anger, disgust, fear, pessimism, sadness |
| Neutral / Αμφίσημο | surprise |

### 3.3 Στατιστικά Dataset

| Split | Δείγματα |
|---|---|
| Train | 6.838 |
| Validation | 886 |
| Test | 3.259 |
| Σύνολο | 10.983 |

Μετά το preprocessing και την αφαίρεση duplicates: **10.896 tweets**.

**Κατανομή Emotions (συνολικό dataset):**

| Emotion | Πλήθος | Ποσοστό |
|---|---|---|
| joy | 4.319 | 39.3% |
| disgust | 4.020 | 36.6% |
| anger | 3.960 | 36.1% |
| optimism | 3.434 | 31.3% |
| sadness | 3.233 | 29.4% |
| fear | 1.848 | 16.8% |
| anticipation | 1.527 | 13.9% |
| love | 1.348 | 12.3% |
| pessimism | 1.270 | 11.6% |
| surprise | 566 | 5.2% |
| trust | 553 | 5.0% |

Το dataset παρουσιάζει σημαντικό **class imbalance**: η κατηγορία joy έχει σχεδόν 8 φορές περισσότερα παραδείγματα από την trust. Αυτό επηρεάζει άμεσα την απόδοση των classifiers και αντιμετωπίζεται με την παράμετρο `class_weight='balanced'`.

**Κατανομή Πλήθους Emotions ανά Tweet:**

| Αριθμός Emotions | Tweets | Ποσοστό |
|---|---|---|
| 0 | 293 | 2.7% |
| 1 | 1.481 | 13.5% |
| 2 | 4.491 | 40.9% |
| 3 | 3.459 | 31.5% |
| 4 | 1.073 | 9.8% |
| 5 | 170 | 1.5% |
| 6 | 16 | 0.1% |

Μέσος αριθμός emotions ανά tweet: **2.37**, γεγονός που επιβεβαιώνει τη multi-label φύση του προβλήματος.

**Σχετικά Plots:**

![Κατανομή emotion labels και multi-label distribution](data/processed/dataset_overview.png)

![Heatmap συν-εμφάνισης emotions](data/processed/emotion_cooccurrence.png)

![Κατανομή ανά train/validation/test split](data/processed/emotion_per_split.png)

---

## 4. Text Preprocessing

### 4.1 Φιλοσοφία Δύο Παραλλαγών

Διαφορετικά υποσυστήματα απαιτούν διαφορετικό είδος κειμένου ως είσοδο. Για τον λόγο αυτό παράγονται δύο παραλλαγές για κάθε tweet:

**`text_clean` — Aggressive Cleaning:**
Χρησιμοποιείται για BM25, TF-IDF και classical ML classifiers. Η αφαίρεση noise (URLs, user mentions, στίξη, stopwords) αποτρέπει την ανεπιθύμητη επίδραση μη σημαντικών tokens στους αλγόριθμους ακριβούς αντιστοίχισης λέξεων.

**`text_light` — Light Cleaning:**
Χρησιμοποιείται για το SentenceTransformer (all-MiniLM-L6-v2) και το Twitter-RoBERTa. Αυτά τα μοντέλα εκπαιδεύτηκαν σε raw tweet text και αξιοποιούν τα hashtags και τα emojis ως σημασιολογική πληροφορία. Η επιθετική αφαίρεσή τους θα αποτελούσε απώλεια πληροφορίας.

### 4.2 Βήματα Επεξεργασίας `text_clean`

Το pipeline εφαρμόζεται διαδοχικά:

1. **Lowercase:** Ομοιοποίηση κεφαλαίων/πεζών για case-insensitive matching.
2. **Αφαίρεση URLs:** Regex pattern αφαιρεί `https://...` και `www....` καθώς δεν φέρουν συναισθηματικό νόημα.
3. **Αφαίρεση Mentions:** Αφαίρεση `@username` — τα usernames δεν συνεισφέρουν στη sentiment πληροφορία.
4. **Emoji Demojization:** Μετατροπή emojis σε κείμενο (π.χ. `😭` → `loudly_crying_face`) αντί για αφαίρεση, διατηρώντας τη συναισθηματική πληροφορία σε μορφή κατάλληλη για TF-IDF.
5. **Αφαίρεση `#`:** Διατήρηση της λέξης του hashtag, αφαίρεση μόνο του `#` συμβόλου.
6. **Αφαίρεση Αριθμών:** Μεμονωμένοι αριθμοί σπάνια φέρουν sentiment πληροφορία.
7. **Αφαίρεση Στίξης:** Μη-αλφαριθμητικοί χαρακτήρες αντικαθίστανται με κενό.
8. **Tokenization (TweetTokenizer):** Χρησιμοποιείται ο NLTK TweetTokenizer αντί για απλό whitespace split, ώστε να αντιμετωπιστούν ορθά οι tweet-specific γλωσσικές δομές.
9. **Αφαίρεση Stopwords:** Αφαιρούνται συχνές λέξεις χωρίς διαφοροποιητική αξία (the, a, is...) και tokens μήκους 1.

**Παράδειγμα:**
```
Input:  "I LOVE #Python! Check this: https://example.com @someone #coding"
Output: "love python snake check coding"
```

### 4.3 Βήματα Επεξεργασίας `text_light`

Μόνο τρία βήματα:
1. Αντικατάσταση URLs με το token `URL`
2. Αντικατάσταση mentions με το token `USER`
3. Normalization whitespace

**Παράδειγμα:**
```
Input:  "I LOVE #Python! Check this: https://example.com @someone #coding"
Output: "I LOVE #Python! Check this: URL USER #coding"
```

### 4.4 Deduplication

Εφαρμόζεται MD5 hashing πάνω στο `text_clean` κάθε tweet. Η χρήση hash αντί για απευθείας σύγκριση strings παρέχει O(1) ταχύτητα αντί O(n). Αφαιρέθηκαν **86 duplicates (0.8%)**, καταλήγοντας σε 10.896 μοναδικά tweets.

### 4.5 Quality Checks

Μετά την επεξεργασία, 1 tweet κατέληξε κενό (αποτελούνταν αποκλειστικά από URLs και stopwords) και αφαιρέθηκε.

**Στατιστικά Token Length (`text_clean`):**

| Στατιστικό | Τιμή |
|---|---|
| Mean | 8.86 tokens |
| Std | 3.88 |
| Min | 1 |
| Median (50%) | 9 |
| Max | 79 |

**Σχετικά Plots:**

![Κατανομή token length για text_clean και text_light](data/processed/token_length_dist.png)

---

## 5. Information Retrieval Layer

### 5.1 Κίνητρο

Στο πλαίσιο του συστήματος, το retrieval αντιμετωπίζεται ως **ad-hoc information retrieval** πάνω σε ένα στατικό corpus: δοθέντος ενός query χρήστη (π.χ. "vaccines health pandemic"), εντοπίζονται τα top-K πιο σχετικά tweets. Αξιολογούνται τρεις στρατηγικές.

### 5.2 BM25 — Sparse Keyword Retrieval

Ο αλγόριθμος BM25 Okapi (Robertson et al., 1995) αποτελεί βελτίωση του TF-IDF ειδικά για retrieval tasks. Λαμβάνει υπόψη:

- **IDF:** Σπάνιες λέξεις έχουν υψηλότερο βάρος από συχνές.
- **TF Saturation:** Η term frequency δεν αυξάνεται γραμμικά — αποφεύγεται η κυριαρχία επαναλαμβανόμενων λέξεων.
- **Length Normalization:** Μεγαλύτερα documents τιμωρούνται ελαφρά.

Ο index χτίζεται πάνω στο `text_clean` corpus και αποθηκεύεται ως `data/index/bm25.pkl`.

### 5.3 Semantic Search — Dense Retrieval

Χρησιμοποιείται το μοντέλο **all-MiniLM-L6-v2** (22M parameters) για την κωδικοποίηση κάθε tweet σε ένα 384-dimensional vector. Ο αλγόριθμος:

1. Κωδικοποιεί όλα τα tweets **μία φορά** κατά τον χρόνο δημιουργίας index (offline).
2. Κατά το runtime, κωδικοποιεί το query σε ένα vector.
3. Βρίσκει τα K nearest neighbors στον vector space χρησιμοποιώντας **FAISS IndexFlatIP** (inner product search).

Με L2 normalization των vectors, inner product = cosine similarity, τιμές στο εύρος [0, 1].

Τα pre-computed embeddings αποθηκεύονται ως `data/index/embeddings.npy` (16MB, shape: 10.896 × 384).

### 5.4 Hybrid Retrieval — Score Fusion

Το hybrid retrieval συνδυάζει BM25 και semantic search:

**Βήμα 1 — Min-Max Normalization:**
Και τα δύο score vectors κανονικοποιούνται στο εύρος [0, 1]. Αυτό είναι απαραίτητο διότι τα BM25 scores είναι unbounded, ενώ τα cosine similarities βρίσκονται ήδη στο [0, 1].

```
BM25_norm  = (score - min) / (max - min)
Sem_norm   = (score - min) / (max - min)
```

**Βήμα 2 — Weighted Sum:**
```
hybrid_score = alpha × BM25_norm + (1 - alpha) × Semantic_norm
```

Ο χρήστης ρυθμίζει το alpha (0 = purely semantic, 1 = purely BM25) μέσω slider στο dashboard. Default: alpha = 0.5.

### 5.5 Αξιολόγηση Retrieval

Αξιολογούνται τρεις standard IR metrics σε 7 queries για k ∈ {5, 10, 20}:

**Precision@k:** Ποσοστό πραγματικά σχετικών tweets στα top-k αποτελέσματα. Μετρά την ακρίβεια.

**Recall@k:** Ποσοστό των σχετικών tweets του corpus που εντοπίστηκαν στα top-k. Μετρά την πληρότητα.

**nDCG@k (Normalized Discounted Cumulative Gain):** Μετρά αν τα πιο σχετικά αποτελέσματα εμφανίζονται ψηλά στο ranking. Τιμωρεί σχετικά αποτελέσματα που βρίσκονται σε χαμηλές θέσεις.

Ο ορισμός "σχετικού" tweet χρησιμοποιεί proxy relevance: ένα tweet θεωρείται σχετικό αν περιέχει τουλάχιστον ένα token του query στο `text_clean`.

**Σχετικά Plots:**

![Κατανομή emotions στα top-50 retrieved tweets για demo query "covid"](data/index/emotion_dist_retrieved.png)

---

## 6. Multi-Label Emotion Classification

### 6.1 Διαμόρφωση Προβλήματος

Το πρόβλημα διαμορφώνεται ως **multi-label binary classification**: για κάθε tweet και για κάθε ένα από τα 11 emotions, προβλέπεται αν το emotion είναι παρόν (1) ή απόν (0). Συνολικά εκπαιδεύονται 11 binary classifiers, ένας ανά emotion.

Αξιολογούνται τρία experiments με αυστηρή αποφυγή data leakage: το TF-IDF vocabulary fit γίνεται **μόνο** στο training set, με transform στο validation και test set.

### 6.2 Experiment 1 — Logistic Regression + TF-IDF

**Feature Extraction:**
Χρησιμοποιούνται δύο TF-IDF vectorizers που συνενώνονται σε ένα feature matrix:

- **Word-level TF-IDF** (unigrams + bigrams, max 30.000 features): Αναπαριστά μεμονωμένες λέξεις και ζεύγη λέξεων. Τα bigrams συλλαμβάνουν φράσεις όπως "mental health" ή "bitcoin price".
- **Char-level TF-IDF** (3-5 character n-grams, max 30.000 features): Αναπαριστά ακολουθίες χαρακτήρων, συλλαμβάνοντας sub-word patterns. Παρέχει ανθεκτικότητα σε typos και slang (π.χ. "luv" και "love" μοιράζονται κοινά n-grams).

Τελικό feature matrix: **(6.768 × 44.826)**

**Μοντέλο:**
`OneVsRestClassifier(LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced'))`

Η παράμετρος `class_weight='balanced'` αντισταθμίζει το class imbalance αυξάνοντας το βάρος των σπάνιων κατηγοριών κατά το training.

**Αποτελέσματα (Test Set):**

| Metric | Score |
|---|---|
| Macro F1 | 0.5222 |
| Micro F1 | 0.6273 |
| Hamming Loss | 0.1702 |
| Subset Accuracy | 0.1602 |

**Ανά Emotion (F1):**

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| joy | 0.819 | 0.805 | 0.812 |
| anger | 0.700 | 0.718 | 0.709 |
| disgust | 0.645 | 0.678 | 0.661 |
| fear | 0.681 | 0.694 | 0.687 |
| sadness | 0.625 | 0.676 | 0.650 |
| optimism | 0.633 | 0.661 | 0.646 |
| love | 0.550 | 0.658 | 0.600 |
| pessimism | 0.316 | 0.430 | 0.364 |
| anticipation | 0.249 | 0.317 | 0.279 |
| surprise | 0.205 | 0.206 | 0.205 |
| trust | 0.110 | 0.163 | 0.132 |

### 6.3 Experiment 2 — Linear SVM + TF-IDF

**Μοντέλο:**
`OneVsRestClassifier(LinearSVC(C=1.0, max_iter=2000, class_weight='balanced'))`

Το LinearSVC βρίσκει το hyperplane μέγιστου margin που χωρίζει τις δύο κλάσεις. Χρησιμοποιεί hinge loss αντί για cross-entropy (Logistic Regression), εστιάζοντας κυρίως στα points κοντά στο decision boundary (support vectors).

**Αποτελέσματα (Test Set):**

| Metric | Score |
|---|---|
| Macro F1 | 0.4891 |
| Micro F1 | 0.6068 |
| Hamming Loss | 0.1715 |
| Subset Accuracy | 0.1569 |

Το SVM υστερεί ελαφρά έναντι του LR, αποτέλεσμα που είναι τυπικό σε high-dimensional sparse feature spaces.

### 6.4 Experiment 3 — Twitter-RoBERTa (Pre-trained Transformer)

**Μοντέλο:** `cardiffnlp/twitter-roberta-base-emotion-multilabel-latest`

Το Twitter-RoBERTa (Barbieri et al., 2020) είναι RoBERTa pre-trained σε ~58M tweets και fine-tuned για multi-label emotion classification. Έχει αρχιτεκτονική 12 Transformer layers με bidirectional self-attention (125M parameters).

Χρησιμοποιείται **zero-shot inference** χωρίς περαιτέρω training, καθώς το label set του μοντέλου ταυτίζεται πλήρως με τα 11 emotions του SemEval dataset.

Το inference γίνεται σε batches των 32, με sigmoid activation στα 11 outputs και threshold 0.5 για binary prediction.

**Αποτελέσματα (Test Set):**

| Metric | Score |
|---|---|
| Macro F1 | 0.5443 |
| Micro F1 | 0.7134 |
| Hamming Loss | 0.1185 |
| Subset Accuracy | 0.2912 |

**Ανά Emotion (F1):**

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| joy | 0.871 | 0.846 | 0.859 |
| anger | 0.789 | 0.806 | 0.797 |
| disgust | 0.729 | 0.772 | 0.750 |
| optimism | 0.742 | 0.737 | 0.740 |
| fear | 0.772 | 0.729 | 0.750 |
| sadness | 0.754 | 0.703 | 0.728 |
| love | 0.717 | 0.526 | 0.607 |
| pessimism | 0.487 | 0.247 | 0.328 |
| anticipation | 0.465 | 0.170 | 0.249 |
| surprise | 0.515 | 0.100 | 0.168 |
| trust | 0.333 | 0.007 | 0.013 |

### 6.5 Συγκριτική Αξιολόγηση

| Experiment | Macro F1 | Micro F1 | Hamming Loss |
|---|---|---|---|
| Logistic Regression + TF-IDF | 0.5222 | 0.6273 | 0.1702 |
| Linear SVM + TF-IDF | 0.4891 | 0.6068 | 0.1715 |
| Twitter-RoBERTa (pre-trained) | **0.5443** | **0.7134** | **0.1185** |

Το Twitter-RoBERTa παρέχει τη βέλτιστη απόδοση και στις τρεις metrics, υπερέχοντας κατά 2.2% στο Macro F1 και 8.6% στο Micro F1 έναντι του LR. Η βελτίωση οφείλεται στην ικανότητα του Transformer να μοντελοποιεί το context μέσω bidirectional attention, σε αντίθεση με τις bag-of-words αναπαραστάσεις των TF-IDF μεθόδων.

Σε όλα τα μοντέλα παρατηρείται σημαντική υποαπόδοση στις κατηγορίες trust και surprise, αντανακλώντας την έλλειψη training examples (553 και 566 αντίστοιχα).

**Σχετικά Plots:**

![Συγκριτικό bar chart F1 ανά emotion για τα τρία μοντέλα](data/models/per_emotion_f1_comparison.png)

---

## 7. Clustering και Θεματική Ανάλυση

### 7.1 Αρχή Λειτουργίας

Μετά το retrieval, τα retrieved tweets ομαδοποιούνται σε micro-topics χρησιμοποιώντας τα sentence embeddings τους. Η διαδικασία είναι **πλήρως data-driven**: δεν απαιτείται προκαθορισμένη θεματική ταξινόμηση ή manual annotation.

### 7.2 K-Means Clustering

Εφαρμόζεται K-Means στα sentence embeddings (384 διαστάσεις) των retrieved tweets.

**Επιλογή Optimal K (Notebook 05):**

Δύο μέθοδοι χρησιμοποιούνται για k ∈ {2, ..., 10}:

- **Elbow Method:** Παρακολουθεί την inertia (Sum of Squared Errors). Το optimal k βρίσκεται στο "elbow" της καμπύλης.
- **Silhouette Score:** Μετρά την ποιότητα separation. Τιμή 1 = τέλεια διαχωρισμένα clusters, 0 = overlapping, -1 = λανθασμένη ανάθεση.

Στο demo query "education technology electronics ai", ο silhouette score επέλεξε k = 2.

**Στο live app:** Χρησιμοποιείται k = min(4, n) για robustness με μικρά σύνολα retrieved tweets. Η παράμετρος `n_init="auto"` εκτελεί τον αλγόριθμο πολλαπλές φορές με διαφορετικές αρχικοποιήσεις για αποφυγή local minima.

### 7.3 UMAP Dimensionality Reduction

Για visualization, εφαρμόζεται UMAP (McInnes et al., 2018) που μειώνει τα 384-dimensional embeddings σε 2 διαστάσεις.

Το UMAP επιλέχθηκε έναντι του PCA διότι:
- Τα sentence embeddings ζουν σε **non-linear manifold** στον υψηδιάστατο χώρο
- Το PCA (γραμμικός αλγόριθμος) αδυνατεί να διατηρήσει την topological δομή
- Το UMAP διατηρεί την **local structure**: tweets με παρόμοιο νόημα εμφανίζονται κοντά στο 2D scatter plot

Παράμετροι: `n_components=2, random_state=42, n_neighbors=min(15, n-1)`.

### 7.4 Top Terms ανά Cluster (Aspect Labels)

Για κάθε cluster, εφαρμόζεται TF-IDF και επιστρέφονται οι 5 λέξεις με τον υψηλότερο μέσο TF-IDF score. Αυτές αποτελούν τα **aspect labels** — ανθρώπινα κατανοητές περιγραφές κάθε θεματικής ομάδας.

Παράδειγμα για query "education technology electronics ai":
- Cluster 0: "amp, new, alarm, india, glee"
- Cluster 1: "school, class, college, learning, lost"

### 7.5 Aspect-Oriented Emotion Analysis

Για κάθε cluster, υπολογίζεται ο μέσος όρος των 11 emotion labels, αποκαλύπτοντας τα κυρίαρχα emotions ανά θεματική ομάδα.

**Σχετικά Plots:**

![Elbow method και Silhouette analysis για επιλογή optimal k](data/aggregation/elbow_silhouette.png)

![UMAP scatter plot χρωματισμένο ανά cluster](data/aggregation/umap_clusters.png)

![Μέση ένταση emotions ανά cluster](data/aggregation/emotion_per_cluster.png)

![Heatmap emotions ανά aspect](data/aggregation/emotion_per_aspect_heatmap.png)

![Grouped bar chart emotions ανά aspect](data/aggregation/emotion_per_aspect_bars.png)

![Σύγκριση UMAP χρωματισμένο ανά cluster vs ανά aspect](data/aggregation/umap_cluster_vs_aspect.png)

---

## 8. Report Generation και Summarization

### 8.1 Polarity Assignment

Κάθε tweet λαμβάνει polarity label βάσει majority vote μεταξύ των ενεργών emotion labels:

```
Positive: joy, love, optimism, anticipation, trust
Negative: anger, disgust, fear, pessimism, sadness
Neutral:  ισοπαλία, απουσία emotions, ή μόνο surprise
```

Το surprise εντάσσεται στο neutral λόγω της αμφίσημης φύσης του — μπορεί να συνοδεύει τόσο θετικά όσο και αρνητικά γεγονότα.

**Κατανομή Polarity για query "education technology electronics ai" (top-500 tweets):**

| Polarity | Πλήθος | Ποσοστό |
|---|---|---|
| Positive | 248 | 49.6% |
| Negative | 210 | 42.0% |
| Neutral | 42 | 8.4% |

### 8.2 Extractive Summary — Centroid-Nearest

Για κάθε polarity group, η extractive σύνοψη επιλέγει τα 3 tweets που βρίσκονται πλησιέστερα στο **centroid** (μέσο embedding) της ομάδας, μετρώντας cosine similarity.

**Αλγόριθμος:**
1. Υπολογισμός centroid: μέσος όρος των embeddings όλων των tweets της ομάδας
2. Υπολογισμός cosine similarity κάθε tweet με το centroid
3. Επιλογή top-3 tweets με τη μεγαλύτερη similarity

Τα tweets κοντά στο centroid είναι τα πιο **αντιπροσωπευτικά** της ομάδας — εκφράζουν τη "μέση άποψη" και όχι ακραίες θέσεις.

**Παράδειγμα Αποτελεσμάτων (Positive Group, 248 tweets):**

| sim | Emotions | Tweet |
|---|---|---|
| 0.668 | joy, optimism | "@edutopia What an innovative way for admin to see what's going on in classrooms! #leadup" |
| 0.648 | fear, joy, optimism | "@RJAH_NHS #course day potential Leadership #excited #nervous #proud" |
| 0.636 | joy, love, optimism | "We are elated to have speakers ready to inspire innovative students!!" |

### 8.3 Abstractive Summary — DistilBART

Το μοντέλο **sshleifer/distilbart-cnn-12-6** (DistilBART, Shleifer & Rush, 2020) παράγει νέο κείμενο συνόψεων. Αρχιτεκτονικά αποτελείται από Encoder (12 layers, bidirectional attention) και Decoder (6 layers, autoregressive generation).

**Διαδικασία:**
1. Επιλογή top-10 tweets ανά polarity group (κατά hybrid retrieval score)
2. Concatenation με separator " | "
3. Προσθήκη prefix: `"Summarize the following tweets about {query}: ..."`
4. Truncation στους 3.200 χαρακτήρες
5. Inference με `min_length=40, max_length=120, do_sample=False`

Το μοντέλο φορτώνεται **on-demand** (lazy loading), παραμένει στη μνήμη για επόμενες κλήσεις και χρησιμοποιεί thread-safe singleton με `threading.Lock`.

### 8.4 Σύγκριση Extractive vs Abstractive

| Χαρακτηριστικό | Extractive | Abstractive |
|---|---|---|
| Μέθοδος | Επιλογή υπαρχόντων tweets | Δημιουργία νέου κειμένου |
| Ταχύτητα | Άμεση | Αργή (~600MB μοντέλο) |
| Hallucination | Αδύνατο | Πιθανό |
| Αναγνωσιμότητα | Χαμηλότερη (raw tweets) | Υψηλότερη (φυσικό κείμενο) |
| Διαθεσιμότητα | Πάντα pre-computed | On-demand button |

**Σχετικά Plots:**

![Bar chart κατανομής polarity](data/report/polarity_distribution.png)

![Heatmap activation rate ανά polarity και emotion](data/report/polarity_emotion_heatmap.png)

---

## 9. Πειραματικά Αποτελέσματα — Συνολική Αξιολόγηση

### 9.1 Απόδοση Retrieval

Όλες οι στρατηγικές retrieval αξιολογήθηκαν με proxy relevance σε 7 queries. Τα queries επιλέχθηκαν ώστε να καλύπτουν διαφορετικές θεματικές (health, finance, environment, emotions, social issues).

Το hybrid retrieval επιτυγχάνει υψηλότερο nDCG@k συγκριτικά με τις επιμέρους μεθόδους, επιβεβαιώνοντας ότι ο συνδυασμός keyword precision (BM25) και semantic recall οδηγεί σε βελτιωμένο ranking.

### 9.2 Απόδοση Classification

Το Macro F1 αποτελεί την κύρια metric, σύμφωνα με το SemEval evaluation protocol, διότι δίνει ίσο βάρος σε όλες τις κλάσεις ανεξαρτήτως frequency.

- **Micro F1:** Κυριαρχείται από τις πολυπληθείς κλάσεις (joy, anger, disgust) — μετρά aggregate performance.
- **Hamming Loss:** Ποσοστό λανθασμένων predictions στο σύνολο των (label, sample) pairs. Χαμηλότερο = καλύτερο.
- **Subset Accuracy:** Ποσοστό δειγμάτων με τέλεια πρόβλεψη **όλων** των 11 labels ταυτόχρονα — πολύ αυστηρή metric.

### 9.3 Clustering

Στο demo query, ο silhouette-optimal αριθμός clusters ήταν k=2, με σαφή διαχωρισμό μεταξύ tweets που αφορούν εκπαιδευτικά ιδρύματα (cluster 1: "school, class, college") και tweets τεχνολογικού χαρακτήρα (cluster 0: "amp, new, alarm").

---

## 10. Συζήτηση

### 10.1 Κύρια Ευρήματα

**Retrieval:** Το hybrid retrieval παρέχει σταθερά ανώτερη κάλυψη σε σχέση με τις επιμέρους μεθόδους. Το BM25 μόνο αστοχεί σε semantically related queries, ενώ το semantic search μόνο μπορεί να χάσει exact matches.

**Classification:** Το Twitter-RoBERTa κυριαρχεί χάρη στο pre-training σε tweet corpora. Ωστόσο, το class imbalance επηρεάζει σοβαρά τα αποτελέσματα για trust και surprise — ακόμα και το RoBERTa δεν μπορεί να "μάθει" επαρκώς από ~550 παραδείγματα. Η **αύξηση training data** για τις σπάνιες κατηγορίες αποτελεί το πρωτεύον bottleneck.

**Clustering:** Το K-Means σε sentence embedding space παρέχει interpretable micro-topics μέσω TF-IDF top terms. Ωστόσο, η χρήση σταθερού k ≤ 4 στο app είναι συντηρητική επιλογή — για μεγαλύτερα datasets θα ήταν σκόπιμη δυναμική επιλογή k.

**Summarization:** Το DistilBART παράγει readable συνόψεις αλλά παρουσιάζει domain mismatch (εκπαίδευση σε CNN/DailyMail news articles vs Twitter), που οδηγεί σε occasional hallucinations ή μη πλήρως συνεκτικές προτάσεις.

### 10.2 Περιορισμοί

1. **Στατικό Corpus:** Το σύστημα αναζητά σε 10.896 pre-annotated tweets από το 2018 — δεν υποστηρίζει real-time data ingestion.
2. **Ground-Truth Labels στο App:** Τα emotion labels στο live dashboard είναι τα annotations του dataset, όχι live inference. Η ενσωμάτωση του Twitter-RoBERTa για live classification θα εξαλείψει αυτόν τον περιορισμό.
3. **Μονόγλωσσο:** Το σύστημα λειτουργεί αποκλειστικά σε αγγλικά tweets.
4. **DistilBART Domain Mismatch:** Το summarization μοντέλο εκπαιδεύτηκε σε news articles, όχι σε tweets.

### 10.3 Μελλοντικές Επεκτάσεις

- Ενσωμάτωση Twitter/X API για real-time data streaming
- Αντικατάσταση DistilBART με tweet-specific summarization model
- Fine-tuning του Twitter-RoBERTa στο SemEval training set
- Υποστήριξη πολλαπλών γλωσσών μέσω multilingual embeddings (mBERT, LaBSE)
- Approximate nearest neighbor search (HNSW, IVF) για scalability σε εκατομμύρια tweets

---

## 11. Συμπεράσματα

1. Το **hybrid retrieval** (BM25 + Semantic) υπερέχει έναντι των επιμέρους μεθόδων, συνδυάζοντας keyword matching precision με semantic understanding.

2. Το **Twitter-RoBERTa** παρέχει τη βέλτιστη emotion classification (Macro F1: 0.5443, Micro F1: 0.7134), υπερέχοντας σημαντικά των classical TF-IDF + ML προσεγγίσεων.

3. Το **K-Means** σε sentence embedding space παράγει interpretable θεματικές ομάδες χωρίς labeled data, με τα TF-IDF top terms να παρέχουν ανθρώπινα κατανοητές περιγραφές.

4. Ο συνδυασμός **extractive** (centroid-nearest) και **abstractive** (DistilBART) summarization παρέχει συμπληρωματική πληροφορία: η extractive προσφέρει αξιοπιστία, η abstractive αναγνωσιμότητα.

5. Το **class imbalance** αποτελεί τον κυρίαρχο περιορισμό για τις κατηγορίες trust και surprise, ανεξαρτήτως επιλογής μοντέλου.
