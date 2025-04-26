import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Hapus import NLTK jika tidak digunakan
# import nltk
# from nltk.tokenize import word_tokenize
import gensim.downloader as api
from sklearn.decomposition import PCA

# Load pre-trained word embeddings
@st.cache_resource
def load_word_embeddings(model_name='glove-wiki-gigaword-100'):
    return api.load(model_name)

# Function to preprocess text
def preprocess_text(text):
    # Tokenize and lowercase using simple split
    tokens = text.lower().split()
    return tokens

# Function to get word embeddings
def get_word_embeddings(tokens, model):
    embeddings = []
    valid_tokens = []
    
    for token in tokens:
        if token in model:
            embeddings.append(model[token])
            valid_tokens.append(token)
    
    return valid_tokens, np.array(embeddings) if embeddings else np.array([])

# Function to generate positional embeddings (sinusoidal encoding)
def get_positional_embeddings(seq_length, d_model=100):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_embedding = np.zeros((seq_length, d_model))
    pos_embedding[:, 0::2] = np.sin(position * div_term)
    pos_embedding[:, 1::2] = np.cos(position * div_term)
    
    return pos_embedding

# Function to reduce dimensions for visualization
def reduce_dimensions(embeddings, n_components=3):
    if len(embeddings) < n_components:
        # If we have fewer samples than components, pad with zeros
        padded = np.zeros((n_components, embeddings.shape[1]))
        padded[:len(embeddings)] = embeddings
        embeddings = padded
    
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

# Main function
def main():
    st.title("Visualisasi 3D Word Embedding")
    
    # Hapus pemanggilan download_nltk_resources()
    
    # Load pre-trained word embeddings
    model_options = {
        'GloVe (100d)': 'glove-wiki-gigaword-100',
        'Word2Vec (300d)': 'word2vec-google-news-300'
    }
    
    model_choice = st.selectbox(
        "Pilih Model Word Embedding",
        list(model_options.keys())
    )
    
    with st.spinner("Memuat model word embedding..."):
        model = load_word_embeddings(model_options[model_choice])
    
    # Text input
    text_input = st.text_area("Masukkan teks:", "Ini adalah contoh kalimat untuk visualisasi word embedding dalam ruang 3D.")
    
    # Embedding type selection
    embedding_type = st.radio(
        "Pilih Jenis Embedding untuk Visualisasi:",
        ["Word Embedding Saja", "Word + Position Embedding"]
    )
    
    # Process button
    if st.button("Proses dan Visualisasikan"):
        # Preprocess text
        tokens = preprocess_text(text_input)
        
        # Get word embeddings
        valid_tokens, word_embeddings = get_word_embeddings(tokens, model)
        
        if len(valid_tokens) == 0:
            st.error("Tidak ada kata yang ditemukan dalam model embedding. Coba teks lain.")
            return
        
        # Get positional embeddings
        pos_embeddings = get_positional_embeddings(len(valid_tokens), word_embeddings.shape[1])
        
        # Visualisasi berdasarkan pilihan
        if embedding_type == "Word Embedding Saja":
            # Hanya visualisasikan word embedding
            visualize_embeddings(word_embeddings, valid_tokens, "Word Embedding")
        else:
            # Visualisasikan word embedding dan word+position embedding dalam dua grafik terpisah
            st.subheader("Visualisasi Word Embedding")
            visualize_embeddings(word_embeddings, valid_tokens, "Word Embedding")
            
            st.subheader("Visualisasi Word + Position Embedding")
            combined_embeddings = word_embeddings + pos_embeddings
            visualize_embeddings(combined_embeddings, valid_tokens, "Word + Position Embedding")
            
            # Tambahkan visualisasi untuk positional embedding saja
            st.subheader("Visualisasi Position Embedding")
            visualize_embeddings(pos_embeddings, valid_tokens, "Position Embedding")
        
        # Display token information in a table
        st.subheader("Detail Embedding")
        st.write("Klik pada titik dalam grafik untuk melihat detail kata.")
        st.dataframe(
            pd.DataFrame({
                'Kata': valid_tokens,
                'Posisi': list(range(len(valid_tokens))),
                'Dimensi Embedding': [f"{word_embeddings.shape[1]}d"] * len(valid_tokens)
            })
        )

# Fungsi untuk visualisasi embedding
def visualize_embeddings(embeddings, tokens, embedding_label):
    # Reduce dimensions for visualization
    reduced_embeddings = reduce_dimensions(embeddings)
    
    # Ensure all arrays have the same length
    n = min(len(reduced_embeddings), len(tokens))
    
    # Create DataFrame for visualization with consistent lengths
    df = pd.DataFrame({
        'x': reduced_embeddings[:n, 0],
        'y': reduced_embeddings[:n, 1],
        'z': reduced_embeddings[:n, 2],
        'token': tokens[:n],
        'position': list(range(n))
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='position',
        hover_name='token',
        hover_data={
            'position': True,
            'x': False,
            'y': False,
            'z': False
        },
        title=f"Visualisasi 3D {embedding_label}",
        labels={'position': 'Posisi Kata'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Add text labels
    fig.add_trace(
        go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='text',
            text=df['token'],
            hoverinfo='none',
            textposition='top center',
            textfont=dict(size=10, color='white'),
            showlegend=False
        )
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimensi 1',
            yaxis_title='Dimensi 2',
            zaxis_title='Dimensi 3'
        ),
        height=600,  # Sedikit lebih kecil agar bisa menampilkan beberapa grafik
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()