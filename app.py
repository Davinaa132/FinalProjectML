# -*- coding: utf-8 -*-
"""app.py

Aplikasi Streamlit untuk deteksi hoaks dengan peningkatan akurasi dan logika klasifikasi yang diperbaiki.
"""

import streamlit as st
import pickle
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

# ------------------------------
# Daftar sumber berita resmi
# ------------------------------
sumber_resmi = [
    "cnnindonesia.com", "kompas.com", "tempo.co", "antaranews.com",
    "detik.com", "liputan6.com", "beritasatu.com",
    "bbc.com", "cnbcindonesia.com", "republika.co.id"
]

def is_sumber_resmi(url):
    for domain in sumber_resmi:
        if domain in url:
            return True
    return False

# ------------------------------
# Fungsi simpan laporan ke CSV
# ------------------------------
def simpan_laporan(judul, url, isi, prediksi_awal, label_benar):
    data = {
        "timestamp": datetime.now().isoformat(),
        "judul": judul,
        "url": url,
        "isi": isi,
        "prediksi_awal": prediksi_awal,
        "label_benar": label_benar,
        "sumber_resmi": is_sumber_resmi(url)
    }
    df = pd.DataFrame([data])
    if os.path.exists("laporan_kesalahan.csv"):
        df.to_csv("laporan_kesalahan.csv", mode='a', index=False, header=False)
    else:
        df.to_csv("laporan_kesalahan.csv", index=False)

# ------------------------------
# Bersihkan isi teks
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    return text.strip()

# ------------------------------
# Ambil isi artikel dari URL
# ------------------------------
def extract_article_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        isi = ' '.join([p.get_text() for p in paragraphs])
        return clean_text(isi)
    except Exception as e:
        return f"[Gagal mengambil isi dari URL: {e}]"

# ------------------------------
# Load model & vectorizer
# ------------------------------
model_path = 'multinomial_nb_modelUMPOH.pkl'
vectorizer_path = 'tfidf_vectorizerUMPOH.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model atau vectorizer tidak ditemukan.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Deteksi Hoaks Berita", layout="centered")
st.title("📰 Deteksi Hoaks dari Judul dan URL Berita")
st.markdown("Masukkan **judul** dan **tautan URL** berita. Sistem akan mendeteksi apakah berita tersebut hoaks atau valid.")

judul = st.text_input("📝 Judul Berita")
url = st.text_input("🔗 URL Berita")

if st.button("🔍 Deteksi"):
    if not judul or not url:
        st.warning("⚠️ Silakan isi judul dan URL terlebih dahulu.")
    else:
        st.info("📡 Mengambil isi berita dari URL...")
        isi = extract_article_from_url(url)

        if isi.startswith("[Gagal"):
            st.error(isi)
        else:
            full_text = judul + " " + isi
            X_input = vectorizer.transform([full_text])
            proba_array = model.predict_proba(X_input)[0]
            prob_valid, prob_hoax = proba_array

            # Tambahkan bobot ke valid jika sumber resmi
            if is_sumber_resmi(url):
                prob_valid += 0.15
                prob_valid = min(prob_valid, 1.0)
                prob_hoax = 1.0 - prob_valid

            # Logika klasifikasi
            if prob_valid >= 0.60:
                status = "VALID"
                st.success(f"✅ Deteksi: **VALID** (Probabilitas: {prob_valid:.2f})")
            elif prob_hoax >= 0.60:
                status = "HOAKS"
                st.error(f"🚨 Deteksi: **HOAKS** (Probabilitas: {prob_hoax:.2f})")
            else:
                status = "TIDAK YAKIN"
                st.warning(f"🤔 Deteksi: **TIDAK YAKIN** (Valid: {prob_valid:.2f} | Hoaks: {prob_hoax:.2f})")
                if is_sumber_resmi(url):
                    st.info("⚠️ *Model tidak yakin, tapi sumber berasal dari media resmi.*")

            # Probabilitas detail
            st.markdown("### 📊 Probabilitas Klasifikasi:")
            st.markdown(f"- **Valid:** {prob_valid:.2f}")
            st.markdown(f"- **Hoaks:** {prob_hoax:.2f}")

            # Form pelaporan kesalahan
            with st.expander("🔁 Apakah hasil ini salah?"):
                label_benar = st.radio("Menurut Anda, berita ini sebenarnya:", ["Valid", "Hoaks"])
                if st.button("📩 Laporkan Kesalahan Deteksi"):
                    simpan_laporan(judul, url, isi, status, label_benar)
                    st.success("✅ Laporan Anda telah disimpan. Terima kasih!")
