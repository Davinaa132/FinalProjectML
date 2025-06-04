# -*- coding: utf-8 -*-
"""app.py

Script untuk deteksi berita hoaks berbasis Streamlit.
"""

import streamlit as st
import pickle
import os
import requests
from bs4 import BeautifulSoup

# Ganti path ini jika file disimpan di folder lain
model_path = 'multinomial_nb_modelUMPOH.pkl'
vectorizer_path = 'tfidf_vectorizerUMPOH.pkl'

# Cek keberadaan file
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model atau vectorizer tidak ditemukan. Jalankan train_model.py dahulu.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

def extract_article_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Ambil semua tag <p>
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        return text.strip()
    except Exception as e:
        return f"[Gagal mengambil isi dari URL: {e}]"

st.set_page_config(page_title="Deteksi Hoaks Berita", layout="centered")
st.title("📰 Deteksi Hoaks dari Judul dan URL Berita")
st.markdown("Masukkan **judul** dan **tautan URL** berita. Sistem akan mendeteksi apakah berita tersebut hoaks atau valid.")

judul = st.text_input("📝 Judul Berita")
url = st.text_input("🔗 URL Berita")

if st.button("🔍 Deteksi"):
    if not judul or not url:
        st.warning("⚠️ Silakan isi judul dan URL berita terlebih dahulu.")
    else:
        st.info("📡 Mengambil isi berita dari URL...")
        isi = extract_article_from_url(url)

        if isi.startswith("[Gagal"):
            st.error(isi)
        else:
            full_text = judul + " " + isi
            X_input = vectorizer.transform([full_text])
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][prediction]

            if prediction == 1:
                st.error(f"🚨 Deteksi: **HOAKS** (Probabilitas: {proba:.2f})")
            else:
                st.success(f"✅ Deteksi: **VALID** (Probabilitas: {proba:.2f})")
