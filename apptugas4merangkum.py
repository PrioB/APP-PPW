import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import nltk

# Force download the punkt tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

# Function to clean text
def cleansing(content):
    content = content.strip()
    content = re.sub(f"[{string.punctuation}]", '', content)
    content = re.sub(r'\d+', '', content)
    content = re.sub(r"\b[a-zA-Z]\b", "", content)
    content = re.sub(r'[^\x00-\x7F]+', '', content)
    content = re.sub(r'\s+', ' ', content)
    return content

# Function to summarize document based on TF-IDF
def summarize_document(melted_tfidf, num_sentences=3):
    tfidf_per_sentence = melted_tfidf.groupby('Sentence')['TF-IDF'].sum()
    top_sentences = tfidf_per_sentence.nlargest(num_sentences)
    summary_sentences = top_sentences.index.tolist()
    return summary_sentences

# Function to create a graph from cosine similarity
def create_similarity_graph(tfidf_matrix, sentences):
    cosine_sim = cosine_similarity(tfidf_matrix)
    G = nx.from_numpy_array(cosine_sim)
    
    # Remove self-loops (edges connecting a node to itself)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    labels = {i: sentences[i] for i in range(len(sentences))}
    return G, labels


# Streamlit app interface
st.title("Ringkasan Dokumen dengan TF-IDF dan Visualisasi Graf")
st.write("""
    Aplikasi ini akan merangkum dokumen yang Anda unggah dengan memilih 
    jumlah kalimat yang ingin dimasukkan ke dalam ringkasan berdasarkan nilai TF-IDF.
    Cukup tempelkan teks Anda di bawah ini dan pilih jumlah kalimat yang diinginkan.
""")

# Input text area
document_text = st.text_area("Paste teks dokumen di sini:")

# Slider to choose number of summary sentences
num_sentences = st.slider("Pilih jumlah kalimat untuk ringkasan:", min_value=1, max_value=10, value=3)

if st.button("Ringkas"):
    if document_text:
        # Step 1: Tokenize sentences
        sentences = sent_tokenize(document_text)
        
        # Step 2: Clean each sentence
        cleaned_sentences = [cleansing(sentence) for sentence in sentences]
        
        # Step 3: Tokenize each sentence into terms
        terms_per_sentence = [word_tokenize(sentence) for sentence in cleaned_sentences]
        
        # Step 4: Combine terms back to string format
        df_terms = pd.DataFrame({'Sentence': sentences, 'Terms_String': [' '.join(terms) for terms in terms_per_sentence]})
        
        # Step 5: Create TF-IDF Vectorizer and transform data
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_terms['Terms_String'])
        
        # Step 6: Create DataFrame for TF-IDF values
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_df['Sentence'] = sentences
        
        # Step 7: Melt the TF-IDF DataFrame
        melted_tfidf = tfidf_df.melt(id_vars=['Sentence'], var_name='Term', value_name='TF-IDF')
        melted_tfidf = melted_tfidf[melted_tfidf['TF-IDF'] != 0]
        
        # Step 8: Summarize the document
        summary = summarize_document(melted_tfidf, num_sentences=num_sentences)
        
        # Display the summary with a more formatted layout
        st.write("## Ringkasan Dokumen")
        if summary:
            for idx, sentence in enumerate(summary, 1):
                st.write(f"**{idx}.** {sentence}")
        else:
            st.warning("Tidak ada kalimat yang dapat dirangkum.")
        
        # Step 9: Create and display the similarity graph
        G, labels = create_similarity_graph(tfidf_matrix, sentences)
        
        # Plot the graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')
        
        st.pyplot(plt)
    else:
        st.warning("Tolong tempelkan teks dokumen untuk dirangkum.")
