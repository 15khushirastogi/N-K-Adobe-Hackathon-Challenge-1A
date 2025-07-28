import pandas as pd
import re
from pathlib import Path

def load_dataset(csv_folder):
    all_data = []
    for csv_file in Path(csv_folder).glob("*.csv"):
        df = pd.read_csv(csv_file)
        print(f"Loaded {csv_file.name}: {len(df)} rows")
        df['source_file'] = csv_file.name
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()

def encode_label(label):
    label_map = {"title": 0, "h1": 1, "h2": 2, "h3": 3, "o": 4}
    return label_map.get(str(label).strip().lower(), 4)

def preprocess_data(df):
    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'] != ""]  
    df['label_enc'] = df['label'].apply(encode_label)
    return df
