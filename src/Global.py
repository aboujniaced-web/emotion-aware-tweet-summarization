 # ============================================================
# GLOBAL MERGE — ROBUST TEXT EXTRACTION FIXED
# ============================================================

import pandas as pd

# ============================================================
# LOAD DATASETS
# ============================================================

crisis = pd.read_csv("/kaggle/input/datasets/khaoulaomnassim/datadatada/crisisnlp_events.csv")
corona = pd.read_csv("/kaggle/input/datasets/khaoulaomnassim/datadatada/events_emotion_aware_final.csv")
sentiment = pd.read_csv("/kaggle/input/datasets/khaoulaomnassim/datadatada/sentiment140_events_final.csv")

# ============================================================
# RESET INDEX
# ============================================================

crisis = crisis.reset_index(drop=True)
corona = corona.reset_index(drop=True)
sentiment = sentiment.reset_index(drop=True)

# ============================================================
# ROBUST EXTRACT FUNCTION (FIX)
# ============================================================

def extract(df):

    # ALL POSSIBLE TEXT COLUMNS
    possible_text_cols = [
        "raw_text",
        "text",
        "raw_tweet",
        "clean_text"
    ]

    text_col = None
    for c in possible_text_cols:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        raise ValueError(f"No text column found. Available: {df.columns}")

    out = pd.DataFrame()

    # RAW TEXT
    out["raw_text"] = df[text_col]

    # EVENT ID
    if "event_id" in df.columns:
        out["event_id"] = df["event_id"]
    else:
        out["event_id"] = -1

    # EVENT TITLE
    if "event_title" in df.columns:
        out["event_title"] = df["event_title"]
    else:
        out["event_title"] = "Unknown Event"

    return out

# ============================================================
# APPLY
# ============================================================

crisis = extract(crisis)
corona = extract(corona)
sentiment = extract(sentiment)

# ============================================================
# DATASET NAME
# ============================================================

crisis["dataset_name"] = "crisisnlp"
corona["dataset_name"] = "corona"
sentiment["dataset_name"] = "sentiment140"

# ============================================================
# GLOBAL EVENT ID
# ============================================================

def add_global_id(df):
    df["global_event_id"] = (
        df["dataset_name"].astype(str)
        + "_"
        + df["event_id"].astype(str)
    )
    return df

crisis = add_global_id(crisis)
corona = add_global_id(corona)
sentiment = add_global_id(sentiment)

# ============================================================
# CONCAT SAFE
# ============================================================

df_global = pd.concat(
    [crisis, corona, sentiment],
    ignore_index=True
)

# ============================================================
# CLEAN
# ============================================================

df_global = df_global.dropna(subset=["raw_text"])
df_global = df_global.drop_duplicates(subset=["raw_text"])
df_global = df_global.reset_index(drop=True)

# ============================================================
# CHECK
# ============================================================

print("\nShape:", df_global.shape)
print("Events:", df_global["global_event_id"].nunique())

print(df_global.head())

# ============================================================
# SAVE
# ============================================================

df_global.to_csv(
    "/kaggle/working/global_event_dataset_final.csv",
    index=False
)

print("Saved OK")