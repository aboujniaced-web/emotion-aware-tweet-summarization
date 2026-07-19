
============================================================

IMPORTS

============================================================

import pandas as pdimport numpy as npimport reimport warnings

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

import umap

from sklearn.cluster import KMeansfrom sklearn.metrics import silhouette_scorefrom sklearn.feature_extraction.text import CountVectorizerfrom sklearn.preprocessing import normalize

import faiss

============================================================

1. LOAD DATA

============================================================

DATA_PATH = "/kaggle/input/datasets/khaoulaomnassim/corona/covid19_tweets.csv"

df = pd.read_csv(DATA_PATH)

============================================================

AUTOMATIC TEXT COLUMN DETECTION

============================================================

possible_cols = ["text","tweet","content","sentence"]

TEXT_COL = None

for c in possible_cols:

if c in df.columns:

    TEXT_COL = c
    break

if TEXT_COL is None:

raise ValueError("No text column found")

print("Using text column:", TEXT_COL)

============================================================

AUTOMATIC DATE COLUMN DETECTION

============================================================

possible_date_cols = ["date","created_at","timestamp"]

DATE_COL = None

for c in possible_date_cols:

if c in df.columns:

    DATE_COL = c
    break

print("Date column:", DATE_COL)

============================================================

REMOVE NULL TEXTS

============================================================

df = df.dropna(subset=[TEXT_COL])

print("Initial shape:", df.shape)

============================================================

2. PREPROCESSING

============================================================

print("\nPreprocessing tweets...")

------------------------------------------------------------

KEEP RAW TEXT

------------------------------------------------------------

df["raw_text"] = df[TEXT_COL].astype(str)

------------------------------------------------------------

CLEAN FOR SEMANTIC CLUSTERING

remove emojis/noise

------------------------------------------------------------

def clean_semantic(t):

t = str(t).lower()

t = re.sub(r"http\S+", "", t)
t = re.sub(r"www\S+", "", t)

t = re.sub(r"@\S+", "", t)

t = re.sub(r"#", "", t)

t = re.sub(r"[^a-z\s]", " ", t)

t = re.sub(r"\s+", " ", t).strip()

return t

------------------------------------------------------------

CLEAN FOR EMOTION DETECTION

KEEP emojis

------------------------------------------------------------

def clean_emotion(t):

t = str(t).lower()

t = re.sub(r"http\S+", "", t)
t = re.sub(r"www\S+", "", t)

t = re.sub(r"@\S+", "", t)

t = re.sub(r"#", "", t)

t = re.sub(
    r"[^a-zA-Z0-9\s"
    r"\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF]+",
    " ",
    t
)

t = re.sub(r"\s+", " ", t).strip()

return t

============================================================

CREATE COLUMNS

============================================================

df["clean_text"] = df["raw_text"].apply(clean_semantic)

df["emotion_text"] = df["raw_text"].apply(clean_emotion)

============================================================

FILTER SHORT TEXTS

============================================================

df = df[df["clean_text"].str.len() > 10]

============================================================

REMOVE DUPLICATES

============================================================

df = df.drop_duplicates("clean_text")

df = df.reset_index(drop=True)

print("After preprocessing:", df.shape)

============================================================

3. TIME WINDOWS

============================================================

if DATE_COL is not None:

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)

df = df.dropna(subset=[DATE_COL])

df["time_window"] = (
    df[DATE_COL]
    .dt.to_period("W")
    .astype(str)
)

else:

df["time_window"] = "global"

print("Time windows:",df["time_window"].nunique())

============================================================

4. SBERT EMBEDDINGS

============================================================

print("\nGenerating embeddings...")

model = SentenceTransformer("all-mpnet-base-v2")

embeddings = model.encode(

df["clean_text"].tolist(),

batch_size=64,

show_progress_bar=True,

convert_to_numpy=True

)

print("Embeddings shape:",embeddings.shape)

============================================================

5. UMAP REDUCTION

============================================================

print("\nRunning UMAP...")

umap_model = umap.UMAP(

n_neighbors=30,

n_components=10,

min_dist=0.0,

metric="cosine",

random_state=42

)

X = umap_model.fit_transform(embeddings)

print("Reduced shape:",X.shape)

============================================================

6. INITIAL CLUSTERING

============================================================

print("\nRunning KMeans...")

N_CLUSTERS = 120

kmeans = KMeans(

n_clusters=N_CLUSTERS,

random_state=42,

n_init="auto"

)

labels = kmeans.fit_predict(X)

df["event_id"] = labels

print("Initial clusters:",df["event_id"].nunique())

============================================================

7. SILHOUETTE SCORE

============================================================

try:

sil = silhouette_score(
    X,
    labels
)

print(
    f"\n🔥 Silhouette Score: {sil:.4f}"
)

except Exception as e:

print("Silhouette failed:", e)

============================================================

8. DYNAMIC FILTERING

============================================================

print("\nFiltering small clusters...")

cluster_sizes = (df["event_id"].value_counts())

MIN_SIZE = max(10,int(len(df) * 0.001))

print("Dynamic min cluster size:",MIN_SIZE)

valid_clusters = cluster_sizes[cluster_sizes >= MIN_SIZE].index

============================================================

IMPORTANT FIX

FILTER DF + EMBEDDINGS + X TOGETHER

============================================================

mask = df["event_id"].isin(valid_clusters)

df = df[mask].reset_index(drop=True)

embeddings = embeddings[mask.values]

X = X[mask.values]

print("Clusters after filtering:",df["event_id"].nunique())

============================================================

REMAP IDS

============================================================

df["event_id"] = (df["event_id"].astype("category").cat.codes)

============================================================

9. BUILD CLUSTER CENTROIDS

============================================================

print("\nBuilding cluster centroids...")

cluster_vectors = []

unique_clusters = sorted(df["event_id"].unique())

for cid in unique_clusters:

mask_cluster = (
    df["event_id"] == cid
).values

vec = embeddings[
    mask_cluster
].mean(axis=0)

cluster_vectors.append(vec)

cluster_vectors = np.vstack(cluster_vectors).astype("float32")

cluster_vectors = normalize(cluster_vectors)

print("Cluster vectors:",cluster_vectors.shape)

============================================================

10. FAISS MERGING

============================================================

print("\nRunning FAISS merging...")

index = faiss.IndexFlatIP(cluster_vectors.shape[1])

index.add(cluster_vectors)

D, I = index.search(cluster_vectors,k=5)

============================================================

11. ADAPTIVE THRESHOLD

============================================================

sim_values = []

for i in range(len(cluster_vectors)):

for k, j in enumerate(I[i]):

    if i != j:

        sim_values.append(
            D[i][k]
        )

sim_values = np.array(sim_values)

THRESHOLD = np.quantile(sim_values,0.90)

print(f"🔥 Adaptive Threshold: "f"{THRESHOLD:.4f}")

============================================================

12. UNION FIND MERGING

============================================================

parent = {i: i for i in range(len(cluster_vectors))}

def find(x):

while parent[x] != x:

    parent[x] = parent[
        parent[x]
    ]

    x = parent[x]

return x

def union(a, b):

ra = find(a)
rb = find(b)

if ra != rb:

    parent[rb] = ra

for i in range(len(cluster_vectors)):

for k, j in enumerate(I[i]):

    if (
        i != j and
        D[i][k] > THRESHOLD
    ):

        union(i, j)

============================================================

13. FINAL EVENT IDS

============================================================

mapping = {}

new_id = 0

for i in range(len(cluster_vectors)):

root = find(i)

if root not in mapping:

    mapping[root] = new_id

    new_id += 1

cid_map = {}

for i, old in enumerate(unique_clusters):

cid_map[old] = mapping[
    find(i)
]

df["event_id"] = (df["event_id"].map(cid_map))

print("\nFinal events:",df["event_id"].nunique())

============================================================

14. TOPIC COHERENCE

============================================================

print("\nComputing topic coherence...")

def topic_coherence(texts):

try:

    vectorizer = CountVectorizer(

        stop_words="english",

        max_features=3000
    )

    X_text = (
        vectorizer.fit_transform(
            texts
        )
    )

    X_bin = (
        X_text > 0
    ).astype(int)

    co_occur = (
        X_bin.T @ X_bin
    ).toarray()

    scores = co_occur / (

        co_occur.sum(
            axis=1,
            keepdims=True
        ) + 1e-9
    )

    return np.mean(scores)

except:

    return 0

coherences = []

for eid in df["event_id"].unique():

texts = df[
    df["event_id"] == eid
]["clean_text"].tolist()

if len(texts) < 10:
    continue

coherences.append(
    topic_coherence(texts)
)

print(f"🔥 Avg Topic Coherence: "f"{np.mean(coherences):.4f}")

============================================================

15. EVENT LABELING

============================================================

print("\nGenerating event labels...")

def get_top_words(texts, n=3):

try:

    vectorizer = CountVectorizer(

        stop_words="english",

        max_features=5000
    )

    X_text = (
        vectorizer.fit_transform(
            texts
        )
    )

    words = (
        vectorizer
        .get_feature_names_out()
    )

    scores = np.asarray(
        X_text.sum(axis=0)
    ).ravel()

    top = words[
        scores.argsort()[::-1][:n]
    ]

    return " ".join(top).title()

except:

    return "Generic Event"

event_titles = {}

for eid in df["event_id"].unique():

texts = df[
    df["event_id"] == eid
]["clean_text"].tolist()

if len(texts) < 10:

    event_titles[eid] = (
        "Small Event"
    )

else:

    event_titles[eid] = (
        get_top_words(texts)
    )

df["event_title"] = (df["event_id"].map(event_titles))

============================================================

16. FINAL STATISTICS

============================================================

stats = (df["event_id"].value_counts().reset_index())

stats.columns = ["event_id","num_tweets"]

print("\n==============================")print("FINAL RESULTS")print("==============================")

print("\nTotal events:",df["event_id"].nunique())

print("Total tweets:",len(df))

print("\nTop events:\n")

print(stats.head(10))

============================================================

17. EXPORT FINAL DATASET

============================================================

OUTPUT_FILE = ("/kaggle/working/""events_emotion_aware_final.csv")

df[["raw_text","clean_text","emotion_text","event_id","event_title","time_window"]].to_csv(OUTPUT_FILE,index=False)

print("\nSaved:", OUTPUT_FILE)