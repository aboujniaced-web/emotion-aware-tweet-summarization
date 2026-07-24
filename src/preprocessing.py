# ============================================================
# EMOTION-PRESERVING PREPROCESSING
# ============================================================

import pandas as pd
import re

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "/kaggle/input/datasets/khaoulaomnassim/global/global_event_dataset_final.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # --------------------------------------------------------
    # Remove user mentions
    # --------------------------------------------------------
    text = re.sub(r'@\w+', '', text)

    # --------------------------------------------------------
    # Remove RT only if it is at the beginning
    # --------------------------------------------------------
    text = re.sub(r'^RT\s+', '', text)

    # --------------------------------------------------------
    # Keep hashtag word
    # Example:
    # #earthquake -> earthquake
    # --------------------------------------------------------
    text = re.sub(r'#(\w+)', r'\1', text)

    # --------------------------------------------------------
    # Limit repeated exclamation marks
    # !!!!!!!!!! -> !!!
    # --------------------------------------------------------
    text = re.sub(r'!{4,}', '!!!', text)

    # --------------------------------------------------------
    # Limit repeated question marks
    # ????????? -> ???
    # --------------------------------------------------------
    text = re.sub(r'\?{4,}', '???', text)

    # --------------------------------------------------------
    # Limit mixed punctuation
    # !!!!!?????? -> !!!???
    # --------------------------------------------------------
    text = re.sub(r'!{3,}\?{3,}', '!!!???', text)

    # --------------------------------------------------------
    # Remove tabs/new lines
    # --------------------------------------------------------
    text = re.sub(r'[\r\n\t]', ' ', text)

    # --------------------------------------------------------
    # Remove multiple spaces
    # --------------------------------------------------------
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ============================================================
# CREATE CLEAN TEXT
# ============================================================

df["clean_text"] = df["raw_text"].apply(preprocess)

# ============================================================
# EMOTION FEATURES
# ============================================================

# Number of emojis
emoji_pattern = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "]+",
    flags=re.UNICODE
)

df["emoji_count"] = df["clean_text"].apply(
    lambda x: len(emoji_pattern.findall(str(x)))
)

# Number of !
df["exclamation_count"] = df["clean_text"].str.count("!")

# Number of ?
df["question_count"] = df["clean_text"].str.count(r"\?")

# Number of hashtags remaining
df["hashtag_count"] = df["raw_text"].str.count(r"#")

# Number of uppercase words
df["uppercase_word_count"] = df["raw_text"].apply(
    lambda x: len(re.findall(r"\b[A-Z]{2,}\b", str(x)))
)

# Tweet length
df["tweet_length"] = df["clean_text"].str.len()

# Number of words
df["word_count"] = df["clean_text"].str.split().str.len()

# ============================================================
# REMOVE EMPTY TWEETS
# ============================================================

before = len(df)

df = df[df["clean_text"].str.strip() != ""]

after = len(df)

print(f"Removed {before-after} empty tweets.")

# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\nExamples\n")

for i in range(min(5, len(df))):

    print("="*70)

    print("RAW:")
    print(df.iloc[i]["raw_text"])

    print()

    print("CLEAN:")
    print(df.iloc[i]["clean_text"])

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset shape:", df.shape)

print("\nColumns:")

print(df.columns.tolist())

# ============================================================
# SAVE
# ============================================================

SAVE_PATH = "/kaggle/working/global_event_dataset_preprocessed.csv"

df.to_csv(
    SAVE_PATH,
    index=False
)

print("\nSaved successfully!")

print(SAVE_PATH)