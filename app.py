# ============================================================
# WEBSITE ANALISIS PDKT COWOK DAN CEWEK
# Random Forest (Machine Learning) vs LSTM (Deep Learning)
# Pendekatan CRISP-DM | Streamlit
# Dataset simulasi 1.200 pasangan
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------- KONFIGURASI -------------------------
st.set_page_config(
    page_title="Analisis PDKT | ML & Deep Learning",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid")
np.random.seed(42)
tf.random.set_seed(42)

FITUR = [
    "Frekuensi_Chat",
    "Inisiatif",
    "Perhatian",
    "Kenyamanan",
    "Respons_Positif",
    "Saling_Menghargai",
    "Kecepatan_Balasan",
    "Konsistensi_Komunikasi",
    "Kecocokan_Hobi"
]

FITUR_URUTAN = [
    "Frekuensi Chat", "Inisiatif", "Perhatian", "Kenyamanan",
    "Respons Positif", "Saling Menghargai", "Kecepatan Balasan"
]

# ------------------------- DATA & MODEL -------------------------
@st.cache_resource(show_spinner="Membuat dataset dan melatih Random Forest serta LSTM...")
def bangun_model():
    N = 1200
    rng = np.random.default_rng(42)

    gender_a = rng.choice(["Cowok", "Cewek"], N)
    gender_b = np.where(gender_a == "Cowok", "Cewek", "Cowok")

    frekuensi_chat = rng.uniform(1, 10, (N, 10))
    inisiatif = rng.uniform(1, 10, (N, 10))
    perhatian = rng.uniform(1, 10, (N, 10))
    kenyamanan = rng.uniform(1, 10, (N, 10))
    respons_positif = rng.uniform(1, 10, (N, 10))
    saling_menghargai = rng.uniform(1, 10, (N, 10))
    kecepatan_balasan = rng.uniform(1, 24, (N, 10))

    df = pd.DataFrame({
        "Gender_A": gender_a,
        "Gender_B": gender_b,
        "Frekuensi_Chat": frekuensi_chat.mean(axis=1),
        "Inisiatif": inisiatif.mean(axis=1),
        "Perhatian": perhatian.mean(axis=1),
        "Kenyamanan": kenyamanan.mean(axis=1),
        "Respons_Positif": respons_positif.mean(axis=1),
        "Saling_Menghargai": saling_menghargai.mean(axis=1),
        "Kecepatan_Balasan": kecepatan_balasan.mean(axis=1),
        "Konsistensi_Komunikasi": rng.uniform(1, 10, N),
        "Kecocokan_Hobi": rng.uniform(1, 10, N)
    })

    skor = (
        0.12 * df["Frekuensi_Chat"]
        + 0.13 * df["Inisiatif"]
        + 0.13 * df["Perhatian"]
        + 0.14 * df["Kenyamanan"]
        + 0.15 * df["Respons_Positif"]
        + 0.12 * df["Saling_Menghargai"]
        + 0.10 * df["Konsistensi_Komunikasi"]
        + 0.08 * df["Kecocokan_Hobi"]
        + 0.03 * (10 - df["Kecepatan_Balasan"])
    )
    skor += rng.normal(0, 0.9, N)

    df["Hasil_PDKT"] = (skor >= 5.0).astype(int)
    df["Status_PDKT"] = df["Hasil_PDKT"].map({0: "Gagal", 1: "Berhasil"})

    X = df[FITUR].copy()
    y = df["Hasil_PDKT"].copy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=250, max_depth=8, min_samples_split=5,
        random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    fitur_urutan = [
        frekuensi_chat, inisiatif, perhatian, kenyamanan,
        respons_positif, saling_menghargai, kecepatan_balasan
    ]
    X_sequence = np.stack(fitur_urutan, axis=2).astype("float32")
    X_seq_train = X_sequence[idx_train]
    X_seq_test = X_sequence[idx_test]

    scaler = StandardScaler()
    n_train = X_seq_train.shape[0]
    n_test = X_seq_test.shape[0]

    X_seq_train_scaled = scaler.fit_transform(
        X_seq_train.reshape(-1, 7)
    ).reshape(n_train, 10, 7)
    X_seq_test_scaled = scaler.transform(
        X_seq_test.reshape(-1, 7)
    ).reshape(n_test, 10, 7)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    lstm = Sequential([
        Input(shape=(10, 7)),
        LSTM(64, return_sequences=True),
        Dropout(0.25),
        LSTM(32),
        Dropout(0.20),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    lstm.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    history = lstm.fit(
        X_seq_train_scaled, y_train.values,
        epochs=40, batch_size=32, validation_split=0.20,
        callbacks=[early_stop], verbose=0
    )

    prob_lstm = lstm.predict(X_seq_test_scaled, verbose=0).flatten()
    pred_lstm = (prob_lstm >= 0.5).astype(int)

    def metrik(nama, aktual, prediksi):
        return {
            "Metode": nama,
            "Accuracy": accuracy_score(aktual, prediksi),
            "Precision": precision_score(aktual, prediksi, zero_division=0),
            "Recall": recall_score(aktual, prediksi, zero_division=0),
            "F1-Score": f1_score(aktual, prediksi, zero_division=0)
        }

    evaluasi = pd.DataFrame([
        metrik("Random Forest", y_test, pred_rf),
        metrik("LSTM", y_test, pred_lstm)
    ])

    importance = pd.DataFrame({
        "Faktor": FITUR,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "df": df, "rf": rf, "lstm": lstm, "scaler": scaler,
        "X_test": X_test, "y_test": y_test,
        "pred_rf": pred_rf, "pred_lstm": pred_lstm,
        "history": history.history, "evaluasi": evaluasi,
        "importance": importance, "analisis_gender": df.groupby("Gender_A")["Hasil_PDKT"]
            .agg(["count", "mean", "sum"]).reset_index(),
        "faktor_rata": df.groupby("Status_PDKT")[FITUR].mean().T
    }

hasil = bangun_model()
df = hasil["df"]
evaluasi = hasil["evaluasi"]

# ------------------------- STYLE -------------------------
st.markdown("""
<style>
    .main {background-color: #f7f8fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.5rem 1.8rem; border-radius: 18px;
        background: linear-gradient(115deg, #252b64, #6457c8);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin-bottom: .25rem;}
    .hero p {color: #eeeeff; margin-bottom: 0;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e8eaf2;
        padding: 15px; border-radius: 14px;
    }
    .note {
        padding: 12px 15px; border-radius: 10px;
        background: #fff7e6; border-left: 4px solid #e6a23c;
        color: #594214;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------- SIDEBAR -------------------------
with st.sidebar:
    st.markdown("## 💬 PDKT Analytics")
    st.caption("Machine Learning • Deep Learning")
    menu = st.radio(
        "NAVIGASI WEBSITE",
        [
            "Dashboard", "Data PDKT", "Evaluasi Model",
            "Analisis Faktor", "Analisis Cowok & Cewek",
            "Prediksi PDKT", "Tahapan CRISP-DM"
        ]
    )
    st.divider()
    st.markdown("**Dataset:** 1.200 data simulasi")
    st.markdown("**Model:** Random Forest & LSTM")
    st.markdown("**Pembagian:** 80% training / 20% testing")

# ------------------------- HEADER -------------------------
st.markdown("""
<div class="hero">
    <h1>💬 Analisis PDKT Cowok dan Cewek</h1>
    <p>Analisis PDKT berhasil dan gagal menggunakan Random Forest dan LSTM
    dengan pendekatan CRISP-DM.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="note"><b>Catatan akademik:</b> aplikasi ini memakai data simulasi, '
    'bukan survei nyata. Nilai prediksi hanya keluaran model dari pola buatan dan '
    'tidak dapat memastikan perasaan, ketertarikan, atau masa depan hubungan.</div>',
    unsafe_allow_html=True
)
st.write("")

# ------------------------- DASHBOARD -------------------------
if menu == "Dashboard":
    st.subheader("Ringkasan Dashboard")
    total = len(df)
    berhasil = int((df["Hasil_PDKT"] == 1).sum())
    gagal = int((df["Hasil_PDKT"] == 0).sum())
    persen = berhasil / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{total:,}")
    c2.metric("PDKT Berhasil", f"{berhasil:,}")
    c3.metric("PDKT Gagal", f"{gagal:,}")
    c4.metric("Proporsi Berhasil", f"{persen:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Komposisi Hasil PDKT")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["Status_PDKT"].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               startangle=90, wedgeprops={"width": 0.42})
        ax.set_title("PDKT Berhasil vs Gagal")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Perbandingan Performa Model")
        fig, ax = plt.subplots(figsize=(5, 4))
        evaluasi.set_index("Metode")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(
            kind="bar", ax=ax, ylim=(0, 1)
        )
        ax.set_ylabel("Skor")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(loc="lower right", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Cuplikan Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    st.download_button(
        "⬇️ Download Dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        "dataset_analisis_pdkt.csv",
        "text/csv"
    )

# ------------------------- DATA PDKT -------------------------
elif menu == "Data PDKT":
    st.subheader("Data Understanding")
    st.write("Dataset simulasi berisi 1.200 pasangan dengan nilai interaksi pada skala 1–10. "
             "Kecepatan balasan diukur dalam jam.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris", f"{df.shape[0]:,}")
    c2.metric("Jumlah Kolom", df.shape[1])
    c3.metric("Data Kosong", int(df.isnull().sum().sum()))

    status = st.multiselect(
        "Filter Status PDKT",
        options=["Berhasil", "Gagal"],
        default=["Berhasil", "Gagal"]
    )
    tampil = df[df["Status_PDKT"].isin(status)]
    st.dataframe(tampil, use_container_width=True, height=420)
    st.caption(f"Menampilkan {len(tampil):,} dari {len(df):,} baris.")
    st.download_button(
        "⬇️ Download data yang ditampilkan",
        tampil.to_csv(index=False).encode("utf-8"),
        "data_pdkt_filter.csv",
        "text/csv"
    )
    with st.expander("Statistik deskriptif"):
        st.dataframe(df[FITUR].describe().T, use_container_width=True)

# ------------------------- EVALUASI MODEL -------------------------
elif menu == "Evaluasi Model":
    st.subheader("Evaluasi Machine Learning dan Deep Learning")
    st.dataframe(
        evaluasi.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1-Score": "{:.4f}"
        }),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    evaluasi.set_index("Metode")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(
        kind="bar", ax=ax, ylim=(0, 1)
    )
    ax.set_title("Perbandingan Metrik Model")
    ax.set_ylabel("Nilai")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    for col, nama, aktual, prediksi in [
        (col1, "Random Forest", hasil["y_test"], hasil["pred_rf"]),
        (col2, "LSTM", hasil["y_test"], hasil["pred_lstm"])
    ]:
        with col:
            st.markdown(f"#### Confusion Matrix {nama}")
            cm = confusion_matrix(aktual, prediksi)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Gagal", "Berhasil"],
                yticklabels=["Gagal", "Berhasil"], ax=ax
            )
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("#### Classification Report")
    tab1, tab2 = st.tabs(["Random Forest", "LSTM"])
    with tab1:
        st.code(classification_report(
            hasil["y_test"], hasil["pred_rf"],
            target_names=["Gagal", "Berhasil"], zero_division=0
        ))
    with tab2:
        st.code(classification_report(
            hasil["y_test"], hasil["pred_lstm"],
            target_names=["Gagal", "Berhasil"], zero_division=0
        ))

    st.markdown("#### Grafik Training LSTM")
    hist = hasil["history"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist["accuracy"], label="Training Accuracy")
    ax.plot(hist["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Riwayat Training LSTM")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "⬇️ Download Hasil Evaluasi CSV",
        evaluasi.to_csv(index=False).encode("utf-8"),
        "hasil_perbandingan_model.csv",
        "text/csv"
    )

# ------------------------- ANALISIS FAKTOR -------------------------
elif menu == "Analisis Faktor":
    st.subheader("Faktor yang Berhubungan dengan Hasil PDKT")
    importance = hasil["importance"]
    st.dataframe(importance, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importance, x="Importance", y="Faktor", ax=ax)
    ax.set_title("Feature Importance Random Forest")
    ax.set_xlabel("Tingkat kepentingan relatif")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Rata-rata Faktor: Berhasil vs Gagal")
    faktor_rata = hasil["faktor_rata"]
    fig, ax = plt.subplots(figsize=(11, 5))
    faktor_rata.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rata-rata nilai")
    ax.set_xlabel("Faktor")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Status")
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "⬇️ Download Feature Importance",
        importance.to_csv(index=False).encode("utf-8"),
        "faktor_keberhasilan_pdkt.csv",
        "text/csv"
    )

# ------------------------- ANALISIS GENDER -------------------------
elif menu == "Analisis Cowok & Cewek":
    st.subheader("Analisis Berdasarkan Gender Pemulai PDKT")
    ag = hasil["analisis_gender"].copy()
    ag.columns = ["Gender Pemulai", "Jumlah PDKT", "Rata-rata Hasil", "Jumlah Berhasil"]
    ag["Persentase Berhasil (%)"] = ag["Rata-rata Hasil"] * 100
    st.dataframe(
        ag[["Gender Pemulai", "Jumlah PDKT", "Jumlah Berhasil", "Persentase Berhasil (%)"]]
        .style.format({"Persentase Berhasil (%)": "{:.2f}%"}),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ag, x="Gender Pemulai", y="Persentase Berhasil (%)", ax=ax)
    ax.set_title("Proporsi Label Berhasil Menurut Gender Pemulai")
    ax.set_ylim(0, 100)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Gender pemulai hanya digunakan untuk analisis deskriptif, bukan sebagai fitur "
        "prediksi. Perbedaan pada data simulasi ini tidak membuktikan bahwa salah satu "
        "gender lebih berhasil dalam PDKT."
    )

# ------------------------- PREDIKSI -------------------------
elif menu == "Prediksi PDKT":
    st.subheader("Simulasi Prediksi Interaksi PDKT")
    st.write("Masukkan nilai contoh interaksi. Skala faktor 1–10, sedangkan kecepatan balasan dalam jam.")
    with st.form("form_prediksi"):
        col1, col2, col3 = st.columns(3)
        with col1:
            chat = st.slider("Frekuensi Chat", 1.0, 10.0, 8.0, 0.5)
            inisiatif = st.slider("Inisiatif", 1.0, 10.0, 8.0, 0.5)
            perhatian = st.slider("Perhatian", 1.0, 10.0, 9.0, 0.5)
        with col2:
            nyaman = st.slider("Kenyamanan", 1.0, 10.0, 9.0, 0.5)
            respons = st.slider("Respons Positif", 1.0, 10.0, 8.0, 0.5)
            hormat = st.slider("Saling Menghargai", 1.0, 10.0, 9.0, 0.5)
        with col3:
            balasan = st.slider("Kecepatan Balasan (jam)", 1.0, 24.0, 2.0, 0.5)
            konsistensi = st.slider("Konsistensi Komunikasi", 1.0, 10.0, 8.0, 0.5)
            hobi = st.slider("Kecocokan Hobi", 1.0, 10.0, 8.0, 0.5)
        submit = st.form_submit_button("🔍 Analisis PDKT", use_container_width=True)

    if submit:
        data_baru = pd.DataFrame([[
            chat, inisiatif, perhatian, nyaman, respons, hormat,
            balasan, konsistensi, hobi
        ]], columns=FITUR)

        prob_rf = hasil["rf"].predict_proba(data_baru)[0, 1] * 100

        urutan = np.array([[
            chat, inisiatif, perhatian, nyaman, respons, hormat, balasan
        ]] * 10, dtype="float32")
        urutan_scaled = hasil["scaler"].transform(urutan).reshape(1, 10, 7)
        prob_lstm = float(hasil["lstm"].predict(urutan_scaled, verbose=0)[0, 0] * 100)

        a, b = st.columns(2)
        a.metric("Skor keluaran Random Forest", f"{prob_rf:.2f}%")
        b.metric("Skor keluaran LSTM", f"{prob_lstm:.2f}%")

        st.markdown("#### Kategori keluaran model")
        c1, c2 = st.columns(2)
        c1.write("**Random Forest:** " + ("Berhasil" if prob_rf >= 50 else "Gagal"))
        c2.write("**LSTM:** " + ("Berhasil" if prob_lstm >= 50 else "Gagal"))
        st.caption(
            "Persentase tersebut adalah probabilitas keluaran model pada data simulasi, "
            "bukan persentase peluang nyata atau penilaian perasaan seseorang."
        )

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(["Random Forest", "LSTM"], [prob_rf, prob_lstm])
        ax.axhline(50, linestyle="--", label="Ambang kategori 50%")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Skor keluaran model (%)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Hubungan yang sehat membutuhkan komunikasi dua arah, persetujuan, "
            "kenyamanan, dan rasa saling menghargai. Jangan gunakan skor ini untuk "
            "menekan atau menyimpulkan perasaan orang lain."
        )

# ------------------------- CRISP-DM -------------------------
elif menu == "Tahapan CRISP-DM":
    st.subheader("Penerapan Tahapan CRISP-DM")
    tahapan = [
        ("1. Business Understanding",
         "Menentukan tujuan analisis, yaitu mempelajari pola interaksi PDKT dan "
         "membandingkan hasil model Random Forest dengan LSTM."),
        ("2. Data Understanding",
         "Memahami dataset simulasi sebanyak 1.200 pasangan, 10 tahap interaksi, "
         "fitur komunikasi, dan label PDKT berhasil atau gagal."),
        ("3. Data Preparation",
         "Memeriksa data kosong dan duplikat, memilih sembilan fitur, membagi data "
         "menjadi 80% training dan 20% testing, serta melakukan standardisasi data LSTM."),
        ("4. Modeling",
         "Melatih Random Forest sebagai Machine Learning dan LSTM sebagai Deep Learning "
         "untuk klasifikasi dua kelas."),
        ("5. Evaluation",
         "Mengevaluasi model menggunakan Accuracy, Precision, Recall, F1-Score, "
         "classification report, dan confusion matrix."),
        ("6. Deployment",
         "Menyajikan hasil analisis melalui website Streamlit dan menyediakan "
         "fitur simulasi prediksi serta unduhan dataset.")
    ]
    for judul, uraian in tahapan:
        with st.expander(judul, expanded=True):
            st.write(uraian)

    st.markdown("#### Ringkasan Pembagian Data")
    x1, x2 = st.columns(2)
    x1.metric("Training", "960 data (80%)")
    x2.metric("Testing", "240 data (20%)")

# ------------------------- FOOTER -------------------------
st.divider()
st.caption(
    "Proyek pembelajaran Data Mining | CRISP-DM | Random Forest & LSTM | "
    "Dataset simulasi untuk demonstrasi akademik"
)
