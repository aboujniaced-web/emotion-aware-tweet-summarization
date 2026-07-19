 # ============================================================
# BUILD CRISISNLP EVENT DATASET
# ============================================================

import os
import glob
import pandas as pd
import re

# ============================================================
# ROOT PATH
# ============================================================

ROOT = "/kaggle/input/datasets/khaoulaomnassim/crisis/CrisisNLP_labeled_data_crowdflower"

# ============================================================
# FIND ALL TSV FILES
# ============================================================

tsv_files = glob.glob(
    os.path.join(ROOT, "**", "*.tsv"),
    recursive=True
)

print("TSV files found:", len(tsv_files))

# ============================================================
# CLEAN FUNCTION
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

# ============================================================
# STORAGE
# ============================================================

all_rows = []

event_mapping = {}

event_counter = 0

# ============================================================
# PROCESS EACH EVENT FILE
# ============================================================

for file in tsv_files:

    try:

        # ----------------------------------------------------
        # EVENT NAME = FOLDER NAME
        # ----------------------------------------------------

        event_name = os.path.basename(
            os.path.dirname(file)
        )

        # ----------------------------------------------------
        # EVENT ID
        # ----------------------------------------------------

        if event_name not in event_mapping:

            event_mapping[event_name] = event_counter
            event_counter += 1

        event_id = event_mapping[event_name]

        # ----------------------------------------------------
        # LOAD TSV
        # ----------------------------------------------------

        df = pd.read_csv(
            file,
            sep="\t",
            encoding="latin-1",
            on_bad_lines="skip"
        )

        print(f"\nProcessing: {event_name}")
        print("Shape:", df.shape)

        # ----------------------------------------------------
        # FIND TEXT COLUMN
        # ----------------------------------------------------

        possible_cols = [
            "tweet",
            "text",
            "Tweet",
            "tweet_text",
            "message"
        ]

        text_col = None

        for c in possible_cols:

            if c in df.columns:

                text_col = c
                break

        if text_col is None:

            print("No tweet column found")
            continue

        # ----------------------------------------------------
        # KEEP TWEETS
        # ----------------------------------------------------

        df = df[[text_col]].copy()

        df.columns = ["raw_tweet"]

        df = df.dropna(subset=["raw_tweet"])

        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        df["clean_text"] = df["raw_tweet"].apply(
            clean_text
        )

        df = df[
            df["clean_text"].str.len() > 10
        ]

        df = df.drop_duplicates(
            "clean_text"
        )

        # ----------------------------------------------------
        # EVENT INFO
        # ----------------------------------------------------

        df["event_id"] = event_id

        df["event_title"] = (
            event_name
            .replace("_", " ")
        )

        df["source"] = "crisisnlp"

        # ----------------------------------------------------
        # APPEND
        # ----------------------------------------------------

        all_rows.append(df)

    except Exception as e:

        print("ERROR:", file)
        print(e)

# ============================================================
# CONCAT ALL EVENTS
# ============================================================

final_df = pd.concat(
    all_rows,
    ignore_index=True
)

# ============================================================
# FINAL CLEAN
# ============================================================

final_df = final_df.drop_duplicates(
    "clean_text"
)

final_df = final_df.reset_index(drop=True)

# ============================================================
# STATS
# ============================================================

print("\n==============================")
print("FINAL CRISISNLP DATASET")
print("==============================")

print("Shape:", final_df.shape)

print("Events:",
      final_df["event_id"].nunique())

print("\nTop events:\n")

print(
    final_df["event_title"]
    .value_counts()
    .head(10)
)

# ============================================================
# SAVE
# ============================================================

OUTPUT = "/kaggle/working/crisisnlp_events.csv"

final_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:", OUTPUT)