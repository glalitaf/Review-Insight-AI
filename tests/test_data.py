import pandas as pd
from src.data import clean_reviews


def make_df(reviews, sentiments, locations=None):
    n = len(reviews)
    return pd.DataFrame({
        "category": ["A"] * n,
        "location": locations or ["Jakarta"] * n,
        "customer_review": reviews,
        "sentiment": sentiments,
        "emotion": ["Happy"] * n,
    })


def test_duplikat_dibuang_setelah_dirapikan():
    df, _ = clean_reviews(make_df(["Bagus banget", "bagus   banget "], ["Positive", "Positive"]))
    assert len(df) == 1


def test_teks_sama_label_beda_dibuang():
    df, _ = clean_reviews(make_df(["oke", "oke"], ["Positive", "Negative"]))
    assert len(df) == 0


def test_teks_kosong_dan_hanya_url_dibuang():
    df, _ = clean_reviews(make_df(["", "http://contoh.com"], ["Positive", "Negative"]))
    assert len(df) == 0


def test_kab_jadi_kabupaten():
    df, _ = clean_reviews(make_df(["mantap"], ["Positive"], ["Kab. Bogor"]))
    assert df.loc[0, "location"] == "Kabupaten Bogor"


def test_hasil_selalu_sama_kalau_dijalankan_ulang():
    data = make_df(["bagus", "jelek", "bagus"], ["Positive", "Negative", "Positive"])
    a, _ = clean_reviews(data)
    b, _ = clean_reviews(data)
    pd.testing.assert_frame_equal(a, b)