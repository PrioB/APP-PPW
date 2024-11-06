import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Judul aplikasi
st.title("Aplikasi Klasifikasi Berita")

# Deskripsi aplikasi
st.write("""
Aplikasi ini melakukan klasifikasi berita menggunakan Logistic Regression.
Data yang digunakan telah melalui preprocessing -> menggunakan fitur TF-IDF -> Reduksi Dimensi menggunakan SVD.
""")

# Load data
df = pd.read_csv("preprocessing-kompas.csv")

# Handling missing values
df['stopword_removal'] = df['stopword_removal'].fillna('')

# Tampilkan data mentah
if st.checkbox("Tampilkan Data"):
    st.write(df.head(10))

# Text vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['stopword_removal'])

# SVD untuk pengurangan dimensi
svd = TruncatedSVD(n_components=100, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

# Target dan split data
y = df['kategori']
X_train, X_test, y_train, y_test = train_test_split(svd_matrix, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)
accuracy = (y_pred == y_test).mean()

# # Tampilkan hasil evaluasi
# st.write("Akurasi Model: {:.2f}".format(accuracy))
# st.text("Classification Report")
# st.text(class_report)

# # Visualisasi confusion matrix
# st.subheader("Confusion Matrix")
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=model.classes_, yticklabels=model.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# st.pyplot(fig)

# Fungsi untuk klasifikasi berita baru
def classify_news(text):
    text_tfidf = vectorizer.transform([text])
    text_svd = svd.transform(text_tfidf)
    prediction = model.predict(text_svd)
    return prediction[0]

# Input teks berita baru
st.subheader("Klasifikasi Berita Baru")
new_text = st.text_area("Masukkan teks berita:")
if st.button("Klasifikasi"):
    if new_text:
        prediction = classify_news(new_text)
        st.write(f"Kategori Prediksi: {prediction}")
    else:
        st.write("Harap masukkan teks berita.")
