# Import library yang dibutuhkan
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca dataset
df = pd.read_csv("preprocessing-kompas.csv")

# Mengganti NaN dengan string kosong
df['stopword_removal'] = df['stopword_removal'].fillna('')

# Menampilkan sampel data pada aplikasi
st.write("Sampel Data:")
st.dataframe(df.head(10))

# Inisialisasi TfidfVectorizer
vectorizer = TfidfVectorizer(norm='l2')
tfidf_matrix = vectorizer.fit_transform(df['stopword_removal'])

# Menentukan fitur dan target
X = tfidf_matrix
y = df['kategori']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan pelatihan model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# # Menampilkan confusion matrix
# st.write("Confusion Matrix:")
# conf_matrix = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots()
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# st.pyplot(fig)

# # Menampilkan akurasi model
# accuracy = (y_pred == y_test).mean()
# st.write(f"Akurasi Model: {accuracy:.2f}")

# Menyediakan fitur prediksi kategori berita
st.write("Prediksi Kategori Berita")
input_text = st.text_area("Masukkan teks berita:")

if st.button("Prediksi"):
    if input_text:
        # Transformasi teks ke TF-IDF
        input_tfidf = vectorizer.transform([input_text])
        
        # Prediksi kategori
        prediction = model.predict(input_tfidf)
        
        # Tampilkan hasil prediksi
        st.write(f"Kategori Prediksi: {prediction[0]}")
    else:
        st.write("Masukkan teks terlebih dahulu.")
