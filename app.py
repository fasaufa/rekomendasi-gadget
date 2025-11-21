import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. KONFIGURASI HALAMAN (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Sistem Rekomendasi Gadget",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
<style>
    .main-header {font-size: 30px; font-weight: bold; color: #2e86c1;}
    .sub-header {font-size: 20px; color: #555;}
    .card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2e86c1;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOAD DATA & MODEL
# ==========================================
@st.cache_resource
def load_data_and_model():
    try:
        # Load Data Bersih
        # Pastikan file 'gadget_clean.csv' ada di satu folder dengan file ini
        df = pd.read_csv('gadget_clean.csv')
        df['clean_text'] = df['clean_text'].fillna('')
        
        # Definisi Stopwords (Sama seperti di Colab)
        stopwords_indo = [
            'dan', 'yang', 'di', 'itu', 'ini', 'ke', 'dari', 'ada', 'buat', 'yg', 'mau',
            'ga', 'gak', 'aku', 'sama', 'kalo', 'lagi', 'bisa', 'karena', 'jadi', 'apa',
            'tapi', 'suka', 'udah', 'banget', 'ya', 'dia', 'kita', 'untuk', 'dengan',
            'pada', 'atau', 'adalah', 'saya', 'mereka', 'kan', 'juga', 'aja', 'kalo',
            'kalau', 'langsung', 'banyak', 'tp', 'dr', 'bgt', 'sdh', 'udh', 'nih', 'sih',
            'kok', 'deh', 'masih', 'biar', 'tetap', 'pun', 'doang', 'nya'
        ]

        # Melatih Model TF-IDF
        tfidf = TfidfVectorizer(stop_words=stopwords_indo)
        tfidf_matrix = tfidf.fit_transform(df['clean_text'])
        
        # Menghitung Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return df, cosine_sim
    except FileNotFoundError:
        return None, None

# Memanggil fungsi load data
df, cosine_sim = load_data_and_model()

# ==========================================
# 3. TAMPILAN SIDEBAR
# ==========================================
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.info(
        """
        Aplikasi ini adalah implementasi **Sistem Rekomendasi** menggunakan metode 
        **Content-Based Filtering**.
        
        Sistem menganalisis kemiripan teks antar review gadget di Twitter.
        """
    )
    st.write("---")
    st.write("**Teknologi:**")
    st.code("Python\nStreamlit\nScikit-Learn\nPandas")
    st.write("---")
    st.caption("Dibuat untuk Skripsi Mahasiswa Sistem Informasi")

# ==========================================
# 4. HALAMAN UTAMA
# ==========================================
st.markdown('<p class="main-header">📱 Pencari Review Gadget Cerdas</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Temukan gadget yang cocok berdasarkan review jujur netizen.</p>', unsafe_allow_html=True)
st.divider()

# Cek apakah data ada
if df is None:
    st.error("⚠️ File 'gadget_clean.csv' tidak ditemukan!")
    st.warning("Silakan upload file 'gadget_clean.csv' ke folder yang sama dengan file app.py ini.")
    st.stop()

# Input Pencarian
col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input("Ketik kebutuhanmu (Misal: Samsung kamera jernih, iPhone baterai awet)", "")
with col2:
    st.write("") # Spasi kosong
    st.write("") 
    cari_btn = st.button("🔍 Cari Rekomendasi", use_container_width=True)

# Logika Pencarian
if cari_btn:
    if not keyword:
        st.toast("⚠️ Masukkan kata kunci dulu dong!")
    else:
        # Proses pencarian
        keyword_lower = keyword.lower()
        hasil = df[df['clean_text'].str.contains(keyword_lower)]
        
        if len(hasil) == 0:
            st.error(f"Maaf, tidak ada review tentang '{keyword}'. Coba kata lain.")
        else:
            # Ambil patokan pertama
            idx = hasil.index[0]
            tweet_patokan = df.iloc[idx]['full_text']
            
            st.success("✅ Review ditemukan! Berikut rekomendasinya:")
            
            # Tampilkan Patokan
            with st.expander("Lihat Review Basis Pencarian", expanded=True):
                st.write(f"**Review Asal:** \"{tweet_patokan}\"")

            # Hitung skor kemiripan
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_5 = sim_scores[1:6]

            # Tampilkan Hasil 5 Teratas
            st.subheader("⭐ 5 Review Gadget Paling Mirip:")
            
            for i, skor in top_5:
                tweet_mirip = df.iloc[i]['full_text']
                persen = skor * 100
                
                # Tampilan Kartu
                st.markdown(f"""
                <div class="card">
                    <h4>Kemiripan: <span style="color:green">{persen:.1f}%</span></h4>
                    <p>"{tweet_mirip}"</p>
                </div>
                """, unsafe_allow_html=True)