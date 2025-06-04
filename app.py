# -*- coding: utf-8 -*-
"""app.py

Aplikasi Streamlit untuk deteksi hoaks berdasarkan judul dan URL berita,
dengan fitur pelaporan kesalahan dan pembelajaran dari laporan.
"""

import streamlit as st
import pickle
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Daftar sumber berita resmi
sumber_resmi = [
    "cnnindonesia.com", "kompas.com", "tempo.co", "antaranews.com",
    "detik.com", "liputan6.com", "beritasatu.com",
    "bbc.com", "cnbcindonesia.com", "republika.co.id"
]

# Fungsi cek apakah URL berasal dari sumber resmi
def is_sumber_resmi(url):
    for domain in sumber_resmi:
        if domain in url:
            return True
    return False

# Fungsi simpan laporan ke file CSV
def simpan_laporan(judul, url, isi, prediksi_awal, label_benar):
    data = {
        "timestamp": datetime.now().isoformat(),
        "judul": judul,
        "url": url,
        "isi": isi,
        "prediksi_awal": prediksi_awal,
        "label_benar": label_benar
    }
    df = pd.DataFrame([data])
    if os.path.exists("laporan_kesalahan.csv"):
        df.to_csv("laporan_kesalahan.csv", mode='a', index=False, header=False)
    else:
        df.to_csv("laporan_kesalahan.csv", index=False)

# Load model & vectorizer
model_path = 'multinomial_nb_modelUMPOH.pkl'
vectorizer_path = 'tfidf_vectorizerUMPOH.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model atau vectorizer tidak ditemukan.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Ambil isi dari URL
def extract_article_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs]).strip()
    except Exception as e:
        return f"[Gagal mengambil isi dari URL: {e}]"

# Streamlit UI
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
            prediction = model.predict(X_input)[0]

            # Probabilitas klasifikasi
            try:
                proba_array = model.predict_proba(X_input)[0]
                prob_valid = proba_array[0]
                prob_hoax = proba_array[1]
            except Exception:
                prob_valid = 0.0
                prob_hoax = 0.0

            # Threshold valid minimum 0.40
            threshold_valid = 0.40
            if prob_valid >= threshold_valid:
                st.success(f"✅ Deteksi: **VALID** (Probabilitas: {prob_valid:.2f})")
                hasil_prediksi = "Valid"
            else:
                st.error(f"🚨 Deteksi: **HOAKS** (Probabilitas: {prob_hoax:.2f})")
                hasil_prediksi = "Hoaks"
                if is_sumber_resmi(url):
                    st.info("⚠️ *Hasil mungkin tidak akurat karena sumber berita berasal dari media resmi.*")

            # Tampilkan detail probabilitas
            st.markdown("### 📊 Probabilitas Klasifikasi:")
            st.markdown(f"- **Valid:** {prob_valid:.2f}")
            st.markdown(f"- **Hoaks:** {prob_hoax:.2f}")

            # Tombol pelaporan kesalahan
            with st.expander("🔁 Apakah hasil ini salah?"):
                label_benar = st.radio("Menurut Anda, berita ini sebenarnya:", ["Valid", "Hoaks"])
                if st.button("📩 Laporkan Kesalahan Deteksi"):
                    simpan_laporan(judul, url, isi, hasil_prediksi, label_benar)
                    st.success("✅ Laporan Anda telah disimpan. Terima kasih!")
