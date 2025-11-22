import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="GadgetMatch.AI - Skripsi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom (SUDAH DIPERBAIKI)
st.markdown("""
<style>
    /* Latar Belakang Header */
    .main-header {
        font-size: 40px;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Desain Kartu Hasil */
    .review-card {
        background-color: #ffffff;
        color: #333333 !important; /* PERBAIKAN: Memaksa teks jadi hitam */
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .review-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Memaksa paragraf di dalam kartu berwarna gelap */
    .review-card p {
        color: #333333 !important;
        margin: 0;
    }
    
    /* Badge Skor */
    .score-badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & MODEL (BACKEND)
# ==========================================
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('gadget_clean.csv')
        df['clean_text'] = df['clean_text'].fillna('')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def build_model(df):
    stopwords_indo = [
        'dan', 'yang', 'di', 'itu', 'ini', 'ke', 'dari', 'ada', 'buat', 'yg', 'mau',
        'ga', 'gak', 'aku', 'sama', 'kalo', 'lagi', 'bisa', 'karena', 'jadi', 'apa',
        'tapi', 'suka', 'udah', 'banget', 'ya', 'dia', 'kita', 'untuk', 'dengan',
        'pada', 'atau', 'adalah', 'saya', 'mereka', 'kan', 'juga', 'aja', 'kalo',
        'kalau', 'langsung', 'banyak', 'tp', 'dr', 'bgt', 'sdh', 'udh', 'nih', 'sih',
        'kok', 'deh', 'masih', 'biar', 'tetap', 'pun', 'doang', 'nya'
    ]
    tfidf = TfidfVectorizer(stop_words=stopwords_indo)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim, tfidf_matrix

# Load Process
df = load_data()

# ==========================================
# 3. SIDEBAR (PROFIL AUTHOR)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>👨‍🎓 Author</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='100'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Mahasiswa SI</h3>", unsafe_allow_html=True)
    
    st.write("---")
    st.success("✅ **Status Sistem:** Online")
    st.info("📅 **Last Update:** Nov 2025")
    
    st.write("---")
    st.caption("Skripsi Sistem Informasi\nUniversitas Amikom Yogyakarta")

# ==========================================
# 4. HALAMAN UTAMA (DASHBOARD)
# ==========================================

st.markdown('<div class="main-header">📱 GadgetMatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Rekomendasi Gadget Cerdas Berbasis Content-Based Filtering</div>', unsafe_allow_html=True)

if df is None:
    st.error("⚠️ Data tidak ditemukan! Harap upload 'gadget_clean.csv'.")
    st.stop()

# Metrics Bar
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Total Review", f"{len(df)} Data")
with col_m2:
    st.metric("Bahasa", "Indonesia (ID)")
with col_m3:
    st.metric("Metode", "TF-IDF + Cosine")

st.write("---")

# Load Model
df, cosine_sim, tfidf_matrix = build_model(df)

# Kolom Pencarian
col_search, col_btn = st.columns([4, 1])
with col_search:
    keyword = st.text_input("", placeholder="🔍 Ketik kebutuhanmu... (Contoh: HP gaming murah, Kamera jernih)", label_visibility="collapsed")
with col_btn:
    tombol_cari = st.button("Cari Rekomendasi", type="primary", use_container_width=True)

# ==========================================
# 5. LOGIKA HASIL & UI CARD
# ==========================================
if tombol_cari and keyword:
    keyword_lower = keyword.lower()
    
    with st.spinner('🤖 Menguraikan kata kunci & mencari kemiripan...'):
        time.sleep(0.5)
        
        hasil = df[df['clean_text'].str.contains(keyword_lower)]
        
        if len(hasil) == 0:
            st.warning(f"🤔 Hmm, tidak menemukan review tentang **'{keyword}'**. Coba kata kunci yang lebih umum.")
        else:
            # Ambil Patokan
            idx = hasil.index[0]
            tweet_patokan = df.iloc[idx]['full_text']
            
            # Tampilkan Patokan
            st.markdown(f"""
            <div style="background-color: #e8f5e9; color: #000; padding: 15px; border-radius: 10px; border: 1px solid #c8e6c9; margin-bottom: 20px;">
                <strong>🎯 Basis Pencarian:</strong><br>"{tweet_patokan}"
            </div>
            """, unsafe_allow_html=True)

            # Hitung Skor
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_5 = sim_scores[1:6]

            st.subheader("✨ Top 5 Rekomendasi Paling Relevan")
            
            # Tampilkan Kartu
            for i, skor in top_5:
                tweet_mirip = df.iloc[i]['full_text']
                persen = skor * 100
                
                # Warna Badge
                if persen > 30:
                    warna_badge = "#d1e7dd"
                    teks_warna = "#0f5132"
                    label = "Sangat Mirip"
                elif persen > 15:
                    warna_badge = "#fff3cd"
                    teks_warna = "#856404"
                    label = "Cukup Mirip"
                else:
                    warna_badge = "#f8d7da"
                    teks_warna = "#721c24"
                    label = "Agak Mirip"

                # HTML Injection (Fixed Text Color)
                st.markdown(f"""
                <div class="review-card">
                    <span style="background-color: {warna_badge}; color: {teks_warna}; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;">
                        {label} ({persen:.1f}%)
                    </span>
                    <p style="margin-top: 10px; font-size: 16px; line-height: 1.6; color: #333333;">"{tweet_mirip}"</p>
                </div>
                """, unsafe_allow_html=True)

st.write("")
st.markdown("<center style='color: #999; font-size: 12px;'>Made with ❤️ using Streamlit for Undergraduate Thesis</center>", unsafe_allow_html=True)
