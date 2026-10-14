"""Muat dan bersihkan review PRDECT-ID, lalu simpan ke DuckDB."""
import re
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd

RAW_PATH = Path("data/raw/prdect_id_dataset.csv")
DB_PATH = Path("data/reviews.duckdb")
CLEAN_CSV = Path("data/processed/reviews_clean.csv")
NUMERIC_COLS = ["price", "overall_rating", "number_sold", "total_review", "customer_rating"]


def load_reviews(path=RAW_PATH) -> pd.DataFrame:
    """Satu-satunya tempat yang tahu sumber data. Di perusahaan, isinya diganti query ke warehouse."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return df


def clean_location(x):
    if pd.isna(x):
        return x
    x = unicodedata.normalize("NFKC", str(x))
    x = re.sub(r"[\u200b-\u200d\ufeff]", "", x)
    x = re.sub(r"\s+", " ", x).strip().title()
    return re.sub(r"\bKab\.?\s+", "Kabupaten ", x)


def clean_text(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.lower()
            .str.replace(r"http\S+|www\.\S+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())


def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    log = {"awal": len(df)}

    for col in ["sentiment", "emotion", "category"]:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["sentiment"] = df["sentiment"].str.title()
    df["emotion"] = df["emotion"].str.title()
    df["location"] = df["location"].apply(clean_location)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["review_clean"] = clean_text(df["customer_review"])
    df = df[df["review_clean"].str.len() > 0]
    log["setelah buang teks kosong"] = len(df)

    n_label = df.groupby("review_clean")["sentiment"].transform("nunique")
    df = df[n_label == 1]
    log["setelah buang teks berlabel ganda"] = len(df)

    df = df.drop_duplicates(subset="review_clean", keep="first").reset_index(drop=True)
    log["setelah buang duplikat"] = len(df)

    df.insert(0, "review_id", range(1, len(df) + 1))
    return df, log


def save_to_duckdb(raw: pd.DataFrame, clean: pd.DataFrame, db_path=DB_PATH) -> None:
    con = duckdb.connect(str(db_path))
    con.register("raw_df", raw)
    con.register("clean_df", clean)
    con.execute("CREATE OR REPLACE TABLE reviews_raw AS SELECT * FROM raw_df")
    con.execute("CREATE OR REPLACE TABLE reviews_clean AS SELECT * FROM clean_df")
    con.close()


if __name__ == "__main__":
    raw = load_reviews()
    clean, log = clean_reviews(raw)
    save_to_duckdb(raw, clean)
    clean.to_csv(CLEAN_CSV, index=False)
    print(pd.Series(log))