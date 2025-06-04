# -*- coding: utf-8 -*-
"""app.py"""
import streamlit as st
import pickle
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

# --------------------------
# Daftar sumber berita resmi
# --------------------------
sumber_resmi = [
    "cnnindonesia.com", "kompas.com", "tempo.co", "antaranews.com",
    "detik.com", "liputan6.com", "beritasatu.com",
    "bbc.com", "cnbcindonesia.com", "republika.co.id"
]

def is_sumber_resmi(url):
    return any(domain in url for domain in sumber_resmi)

# --------------------------
# Simpan laporan ke CSV
# --------------------------
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

# --------------------------
# Preprocessing text
# --------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    return text.strip()

def extract_article_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        isi = ' '.join([p.get_text() for p in paragraphs])
        return clean_text(isi)
    except Exception as e:
        return f"[Gagal mengambil isi dari URL: {e}]"

# --------------------------
# Load model & vectorizer
# --------------------------
model_path = 'multinomial_nb_modelUMPOH.pkl'
vectorizer_path = 'tfidf_vectorizerUMPOH.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model atau vectorizer tidak ditemukan.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Deteksi Hoaks Berita", layout="centered")
st.title("📰 Deteksi Hoaks Berita")
st.markdown("Masukkan **judul**, lalu pilih antara mengisi **URL berita** atau menulis **isi berita** secara manual.")

judul = st.text_input("📝 Judul Berita")
url = st.text_input("🔗 URL Berita")
isi_manual = st.text_area("📄 Isi Berita (jika tidak ada URL)", height=200)

if st.button("🔍 Deteksi"):
    if not judul or (not url and not isi_manual):
        st.warning("⚠️ Harap isi judul dan salah satu dari URL atau isi berita.")
    else:
        if url:
            st.info("📡 Mengambil isi berita dari URL...")
            isi = extract_article_from_url(url)
        else:
            isi = clean_text(isi_manual)

        if isi.startswith("[Gagal"):
            st.error(isi)
        else:
            full_text = judul + " " + isi
            X_input = vectorizer.transform([full_text])

            proba_array = model.predict_proba(X_input)[0]
            prob_valid = prob_hoax = 0.0

            # Mapping berdasarkan label
            for i, cls in enumerate(model.classes_):
                if cls == 0:
                    prob_valid = proba_array[i]
                elif cls == 1:
                    prob_hoax = proba_array[i]

            # Tambahkan bobot valid jika dari sumber resmi
            if url and is_sumber_resmi(url):
                prob_valid += 0.15
                prob_valid = min(prob_valid, 1.0)
                prob_hoax = 1.0 - prob_valid

            # Penentuan status pasti
            if prob_valid > prob_hoax:
                status = "VALID"
                st.success(f"✅ Deteksi: **VALID** (Probabilitas: {prob_valid:.2f})")
            else:
                status = "HOAKS"
                st.error(f"🚨 Deteksi: **HOAKS** (Probabilitas: {prob_hoax:.2f})")

            # Tampilkan detail probabilitas
            st.markdown("### 📊 Probabilitas Klasifikasi:")
            st.markdown(f"- **Valid:** {prob_valid:.2f}")
            st.markdown(f"- **Hoaks:** {prob_hoax:.2f}")

            # Form pelaporan kesalahan
            with st.expander("🔁 Apakah hasil ini salah?"):
                label_benar = st.radio("Menurut Anda, berita ini sebenarnya:", ["Valid", "Hoaks"])
                if st.button("📩 Laporkan Kesalahan Deteksi"):
                    simpan_laporan(judul, url or "-", isi, status, label_benar)
                    st.success("✅ Laporan Anda telah disimpan. Terima kasih!")
