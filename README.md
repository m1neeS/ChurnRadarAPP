# ChurnRadarAPP 📡
by M1neeS

Prediksi customer churn untuk perusahaan telekomunikasi. Model ini membantu mengidentifikasi pelanggan yang berisiko berhenti berlangganan sehingga tim retention bisa mengambil tindakan lebih awal.

## Kenapa Proyek Ini?

Kehilangan pelanggan itu mahal. Biaya akuisisi pelanggan baru bisa 5-7x lebih besar dibanding mempertahankan yang sudah ada. Dengan model prediksi ini, kita bisa:
- Identifikasi pelanggan berisiko tinggi sebelum mereka pergi
- Prioritaskan resource untuk retention campaign
- Pahami faktor-faktor yang mendorong churn

## Dataset

Menggunakan Telco Customer Churn dataset:
- 7,043 pelanggan
- 21 fitur (demografi, layanan, kontrak, billing)
- Churn rate sekitar 26.5%

## Hasil

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| Random Forest | **0.838** | 75.7% |
| XGBoost | 0.834 | 76.2% |
| Logistic Regression | 0.831 | 76.3% |
| LightGBM | 0.831 | 76.3% |
| Decision Tree | 0.763 | 74.2% |

Random Forest dipilih sebagai model final karena ROC-AUC tertinggi.

## Struktur Folder

```
├── data/
│   ├── raw/                 # Dataset asli
│   └── processed/           # Data setelah preprocessing
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── models/                  # Model & scaler yang sudah di-train
├── streamlit_app/           # Web app untuk prediksi
└── outputs/                 # Visualisasi & report
```

## Cara Pakai

### Setup
```bash
pip install -r requirements.txt
```

### Jalankan Notebook
Buka notebook secara berurutan dari 01 sampai 04 untuk melihat keseluruhan proses.

### Jalankan Web App
```bash
cd streamlit_app
streamlit run app.py
```

## Insight Utama

Beberapa faktor yang paling berpengaruh terhadap churn:
- **Kontrak month-to-month** → churn rate 3x lebih tinggi
- **Tenure rendah** (< 12 bulan) → periode kritis
- **Pembayaran electronic check** → korelasi dengan churn lebih tinggi
- **Fiber optic tanpa add-on services** → cenderung churn

## Tech Stack

- Python 3.x
- scikit-learn, XGBoost, LightGBM
- imbalanced-learn (SMOTE)
- SHAP untuk interpretability
- Streamlit untuk web app
- Plotly untuk visualisasi interaktif

## Screenshot

Web app menampilkan:
- Risk score dengan gauge chart
- Rekomendasi retention berdasarkan level risiko
- Analisis faktor risiko per pelanggan

## Kesimpulan

### Performance
Model Random Forest berhasil memprediksi churn dengan ROC-AUC **0.838**. Dari 374 pelanggan yang benar-benar churn di test set, model berhasil menangkap 283 (recall 75.7%). Precision 52.9% artinya dari semua prediksi churn, sekitar setengahnya benar.

### Confusion Matrix
| | Predicted No | Predicted Yes |
|---|---|---|
| Actual No | 783 (TN) | 252 (FP) |
| Actual Yes | 91 (FN) | 283 (TP) |

### Business Impact
Dengan threshold 70%:
- **288 pelanggan** teridentifikasi high-risk
- **56 pelanggan** diperkirakan bisa di-retain
- **Net benefit: $27,600**
- **ROI: 95.8%**

### Rekomendasi
1. Fokus retention pada pelanggan dengan churn probability > 70%
2. Monitor false negatives untuk improve recall
3. Adjust threshold sesuai kebutuhan bisnis - kalau mau lebih agresif, turunkan threshold

---

Feel free to fork dan improve. Kalau ada pertanyaan atau saran, buka issue aja.
