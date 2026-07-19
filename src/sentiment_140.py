# ============================================================
# SENTIMENT140 — FULL CLEAN + EVENT DETECTION PIPELINE
# ============================================================

!pip install -q sentence-transformers umap-learn scikit-learn faiss-cpu

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import faiss

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "/kaggle/input/datasets/khaoulaomnassim/sentiment140/training.1600000.processed.noemoticon.csv"

cols = ["sentiment", "tweet_id", "date", "query", "user", "text"]

df = pd.read_csv(DATA_PATH, encoding="latin-1", names=cols)

print("Initial shape:", df.shape)

# ============================================================
# 2. KEEP IMPORTANT COLUMNS (IMPORTANT)
# ============================================================

# KEEP RAW TEXT + SENTIMENT
df = df[["text", "date", "sentiment"]].copy()

# ============================================================
# 3. CLEAN TEXT (BUT KEEP RAW)
# ============================================================

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"www\S+", "", t)
    t = re.sub(r"@\S+", "", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["clean_text"] = df["text"].apply(clean_text)

df = df[df["clean_text"].str.len() > 10]
df = df.drop_duplicates("clean_text").reset_index(drop=True)

print("After cleaning:", df.shape)

# ============================================================
# 4. TIME WINDOW
# ============================================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["time_window"] = df["date"].dt.to_period("W").astype(str)

print("Weeks:", df["time_window"].nunique())

# ============================================================
# 5. OPTIONAL SAMPLING (VERY IMPORTANT KAGGLE)
# ============================================================

MAX_SAMPLES = 200000

if len(df) > MAX_SAMPLES:
    df = df.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)

print("Final shape:", df.shape)

# ============================================================
# 6. SBERT EMBEDDINGS
# ============================================================

print("Generating embeddings...")

model = SentenceTransformer("all-mpnet-base-v2")

embeddings = model.encode(
    df["clean_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings shape:", embeddings.shape)

# ============================================================
# 7. UMAP REDUCTION
# ============================================================

print("Running UMAP...")

umap_model = umap.UMAP(
    n_neighbors=30,
    n_components=10,
    metric="cosine",
    random_state=42
)

X = umap_model.fit_transform(embeddings)

# ============================================================
# 8. KMEANS CLUSTERING
# ============================================================

N_CLUSTERS = 200

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init="auto"
)

labels = kmeans.fit_predict(X)

df["event_id"] = labels

# ============================================================
# 9. SILHOUETTE SCORE
# ============================================================

sil = silhouette_score(X, labels)
print("Silhouette:", sil)

# ============================================================
# 10. FILTER SMALL CLUSTERS
# ============================================================

cluster_sizes = df["event_id"].value_counts()

MIN_SIZE = max(10, int(len(df) * 0.001))

valid_clusters = cluster_sizes[cluster_sizes >= MIN_SIZE].index

mask = df["event_id"].isin(valid_clusters)

df = df[mask].reset_index(drop=True)
embeddings = embeddings[mask.values]
X = X[mask.values]

# remap
df["event_id"] = df["event_id"].astype("category").cat.codes

print("Clusters after filtering:", df["event_id"].nunique())

# ============================================================
# 11. CLUSTER CENTROIDS
# ============================================================

cluster_vectors = []
unique_clusters = sorted(df["event_id"].unique())

for cid in unique_clusters:
    vec = embeddings[df["event_id"] == cid].mean(axis=0)
    cluster_vectors.append(vec)

cluster_vectors = np.vstack(cluster_vectors).astype("float32")
cluster_vectors = normalize(cluster_vectors)

# ============================================================
# 12. FAISS SIMILARITY MERGING
# ============================================================

index = faiss.IndexFlatIP(cluster_vectors.shape[1])
index.add(cluster_vectors)

D, I = index.search(cluster_vectors, k=5)

# ============================================================
# 13. ADAPTIVE THRESHOLD
# ============================================================

sim = []

for i in range(len(cluster_vectors)):
    for k, j in enumerate(I[i]):
        if i != j:
            sim.append(D[i][k])

THRESHOLD = np.quantile(sim, 0.90)

print("Adaptive threshold:", THRESHOLD)

# ============================================================
# 14. UNION FIND MERGING
# ============================================================

parent = {i: i for i in range(len(cluster_vectors))}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for i in range(len(cluster_vectors)):
    for k, j in enumerate(I[i]):
        if i != j and D[i][k] > THRESHOLD:
            union(i, j)

# ============================================================
# 15. FINAL REMAPPING
# ============================================================

mapping = {}
new_id = 0

for i in range(len(cluster_vectors)):
    r = find(i)
    if r not in mapping:
        mapping[r] = new_id
        new_id += 1

cid_map = {old: mapping[find(i)] for i, old in enumerate(unique_clusters)}

df["event_id"] = df["event_id"].map(cid_map)

print("Final events:", df["event_id"].nunique())

# ============================================================
# 16. EVENT LABELS
# ============================================================

def label(texts):
    vec = CountVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    words = vec.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).ravel()
    return " ".join(words[scores.argsort()[::-1][:3]]).title()

event_titles = {}

for eid in df["event_id"].unique():
    texts = df[df["event_id"] == eid]["clean_text"].tolist()
    if len(texts) < 10:
        event_titles[eid] = "Small Event"
    else:
        event_titles[eid] = label(texts)

df["event_title"] = df["event_id"].map(event_titles)

# ============================================================
# 17. FINAL EXPORT (IMPORTANT)
# ============================================================

OUTPUT = "/kaggle/working/sentiment140_events_final.csv"

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)