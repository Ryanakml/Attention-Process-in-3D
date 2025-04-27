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
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

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
def reduce_dimensions(embeddings, method='pca', n_components=3, perplexity=30, n_iter=2000):
    if len(embeddings) < n_components:
        # If we have fewer samples than components, pad with zeros
        padded = np.zeros((n_components, embeddings.shape[1]))
        padded[:len(embeddings)] = embeddings
        embeddings = padded
    
    # Convert method to lowercase and remove any hyphens for consistent checking
    method = method.lower().replace('-', '')
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings)
        
        # Store explained variance for display if needed
        explained_variance = reducer.explained_variance_ratio_
        st.session_state['explained_variance'] = explained_variance
        
        return reduced
    elif method == 'tsne':
        # Adjust perplexity if we have few samples
        if len(embeddings) < 30:
            perplexity = min(5, len(embeddings) - 1) if len(embeddings) > 1 else 1
            
        reducer = TSNE(n_components=n_components, 
                      perplexity=perplexity,
                      n_iter=n_iter,
                      learning_rate='auto',
                      init='pca',
                      metric='cosine',
                      random_state=42)
        
        return reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

# Function to calculate word-to-word similarity matrix
def calculate_similarity_matrix(tokens, model):
    n = len(tokens)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i][j] = 1.0  # Self-similarity is 1
            else:
                # Calculate cosine similarity between word vectors
                if tokens[i] in model and tokens[j] in model:
                    sim_matrix[i][j] = model.similarity(tokens[i], tokens[j])
    
    return sim_matrix

# Function to find similar words
def find_similar_words(word, model, n=5, threshold=0.5):
    if word not in model:
        return []
    
    similar_words = model.most_similar(word, topn=n*2)  # Get more candidates
    # Filter by similarity threshold
    similar_words = [(w, sim) for w, sim in similar_words if sim >= threshold]
    return similar_words[:n]  # Return top n that meet threshold

# Main function
def main():
    st.title("Visualisasi 3D Word Embedding")
    
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
    
    # Dimensionality reduction method
    reduction_method = st.radio(
        "Pilih Metode Reduksi Dimensi:",
        ["PCA", "t-SNE"]
    )
    
    # Text input
    text_input = st.text_area("Masukkan teks:", "Ini adalah contoh kalimat untuk visualisasi word embedding dalam ruang 3D.")
    
    # Option to add similar words
    add_similar = st.checkbox("Tambahkan kata-kata yang mirip secara semantik", value=True)
    
    # Embedding type selection
    embedding_type = st.radio(
        "Pilih Jenis Embedding untuk Visualisasi:",
        ["Word Embedding Saja", "Word + Position Embedding"]
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.3, 
            max_value=0.9, 
            value=0.5, 
            step=0.05,
            help="Only show similar words with similarity above this threshold"
        )
        
        if reduction_method == "t-SNE":
            perplexity_value = st.slider(
                "t-SNE Perplexity", 
                min_value=5, 
                max_value=50, 
                value=30 if len(text_input.split()) > 30 else 5,
                help="Higher values consider more global structure, lower values focus on local structure"
            )
            
            n_iter = st.slider(
                "t-SNE Iterations", 
                min_value=500, 
                max_value=5000, 
                value=2000, 
                step=500,
                help="More iterations may produce better results but take longer"
            )
        else:  # PCA
            explained_variance = st.checkbox(
                "Show Explained Variance", 
                value=False,
                help="Display the amount of variance explained by each principal component"
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
        
        # Add similar words if requested
        similar_words_dict = {}
        if add_similar and len(valid_tokens) > 0:
            with st.spinner("Mencari kata-kata yang mirip secara semantik..."):
                for token in valid_tokens:
                    similar = find_similar_words(token, model, n=3)
                    if similar:
                        similar_words_dict[token] = similar
                
                # Add similar words to visualization
                additional_tokens = []
                additional_embeddings = []
                
                for original_token, similars in similar_words_dict.items():
                    for similar_word, similarity in similars:
                        if similar_word not in valid_tokens and similar_word not in additional_tokens:
                            additional_tokens.append(similar_word)
                            additional_embeddings.append(model[similar_word])
                
                if additional_tokens:
                    extended_tokens = valid_tokens + additional_tokens
                    extended_embeddings = np.vstack([word_embeddings, np.array(additional_embeddings)]) if additional_embeddings else word_embeddings
                    
                    # Create a list to track which tokens are original vs similar
                    token_types = ["original"] * len(valid_tokens) + ["similar"] * len(additional_tokens)
                    
                    # Create a list to track which original word each similar word is related to
                    related_to = [None] * len(valid_tokens)
                    for i, token in enumerate(additional_tokens):
                        for original_token, similars in similar_words_dict.items():
                            if any(token == similar_word for similar_word, _ in similars):
                                related_to.append(original_token)
                                break
                        else:
                            related_to.append(None)
                else:
                    extended_tokens = valid_tokens
                    extended_embeddings = word_embeddings
                    token_types = ["original"] * len(valid_tokens)
                    related_to = [None] * len(valid_tokens)
        else:
            extended_tokens = valid_tokens
            extended_embeddings = word_embeddings
            token_types = ["original"] * len(valid_tokens)
            related_to = [None] * len(valid_tokens)
        
        # Get positional embeddings (only for original tokens)
        pos_embeddings = get_positional_embeddings(len(valid_tokens), word_embeddings.shape[1])
        
        # For combined embedding, we need to extend positional embeddings if we added similar words
        if len(extended_tokens) > len(valid_tokens):
            # Create zero embeddings for similar words (they don't have positions)
            additional_pos = np.zeros((len(extended_tokens) - len(valid_tokens), pos_embeddings.shape[1]))
            extended_pos_embeddings = np.vstack([pos_embeddings, additional_pos])
        else:
            extended_pos_embeddings = pos_embeddings
        
        # Visualisasi berdasarkan pilihan
        if embedding_type == "Word Embedding Saja":
            # Hanya visualisasikan word embedding
            visualize_embeddings(
                extended_embeddings, 
                extended_tokens, 
                "Word Embedding", 
                reduction_method.lower(), 
                token_types=token_types,
                related_to=related_to
            )
        else:
            # Visualisasikan word embedding
            st.subheader("Visualisasi Word Embedding")
            visualize_embeddings(
                extended_embeddings, 
                extended_tokens, 
                "Word Embedding", 
                reduction_method.lower(),
                token_types=token_types,
                related_to=related_to
            )
            
            # For combined embedding, only use original tokens
            if len(extended_tokens) > len(valid_tokens):
                st.info("Catatan: Visualisasi Word + Position Embedding hanya menggunakan kata-kata asli dari teks input.")
                
            # Visualisasikan word+position embedding
            st.subheader("Visualisasi Word + Position Embedding")
            combined_embeddings = word_embeddings + pos_embeddings
            visualize_embeddings(
                combined_embeddings, 
                valid_tokens, 
                "Word + Position Embedding", 
                reduction_method.lower(),
                token_types=["original"] * len(valid_tokens),
                related_to=[None] * len(valid_tokens)
            )
            
            # Tambahkan visualisasi untuk positional embedding saja
            st.subheader("Visualisasi Position Embedding")
            visualize_embeddings(
                pos_embeddings, 
                valid_tokens, 
                "Position Embedding", 
                reduction_method.lower(),
                token_types=["original"] * len(valid_tokens),
                related_to=[None] * len(valid_tokens)
            )
        
        # Display token information in a table
        st.subheader("Detail Embedding")
        st.write("Klik pada titik dalam grafik untuk melihat detail kata.")
        
        # Create a more informative dataframe
        df_info = pd.DataFrame({
            'Kata': extended_tokens,
            'Tipe': token_types,
            'Posisi': list(range(len(valid_tokens))) + [-1] * (len(extended_tokens) - len(valid_tokens)),
            'Dimensi Embedding': [f"{extended_embeddings.shape[1]}d"] * len(extended_tokens),
            'Terkait Dengan': related_to
        })
        
        # Display the dataframe
        st.dataframe(df_info)
        
        # Display similar words information if available
        if similar_words_dict and add_similar:
            st.subheader("Kata-kata yang Mirip Secara Semantik")
            for token, similars in similar_words_dict.items():
                st.write(f"**{token}**: " + ", ".join([f"{word} ({similarity:.2f})" for word, similarity in similars]))

# Fungsi untuk visualisasi embedding
def visualize_embeddings(embeddings, tokens, embedding_label, method='pca', token_types=None, related_to=None, similarity_matrix=None):
    # Reduce dimensions for visualization
    reduced_embeddings = reduce_dimensions(embeddings, method=method)
    
    # Ensure all arrays have the same length
    n = min(len(reduced_embeddings), len(tokens))
    
    if token_types is None:
        token_types = ["original"] * n
    else:
        token_types = token_types[:n]
        
    if related_to is None:
        related_to = [None] * n
    else:
        related_to = related_to[:n]
    
    # Create DataFrame for visualization with consistent lengths
    df = pd.DataFrame({
        'x': reduced_embeddings[:n, 0],
        'y': reduced_embeddings[:n, 1],
        'z': reduced_embeddings[:n, 2],
        'token': tokens[:n],
        'position': list(range(len([t for t in token_types[:n] if t == "original"]))) + 
                   [-1] * len([t for t in token_types[:n] if t != "original"]),
        'type': token_types[:n],
        'related_to': related_to[:n]
    })
    
    # Create a custom color scale for different token types
    color_map = {'original': 'position', 'similar': 'related_to'}
    color_column = 'type'
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color=color_column,
        hover_name='token',
        hover_data={
            'position': True,
            'type': True,
            'related_to': True,
            'x': False,
            'y': False,
            'z': False
        },
        title=f"Visualisasi 3D {embedding_label} (Metode: {method.upper()})",
        labels={'position': 'Posisi Kata', 'type': 'Tipe Kata', 'related_to': 'Terkait Dengan'},
        color_discrete_map={'original': '#1f77b4', 'similar': '#ff7f0e'}
    )
    
    # Customize marker size and opacity based on token type
    for i, trace in enumerate(fig.data):
        if trace.name == 'original':
            trace.marker.size = 8
            trace.marker.opacity = 1.0
        else:
            trace.marker.size = 6
            trace.marker.opacity = 0.7
    
    # Add text labels with improved visibility
    for token_type in df['type'].unique():
        subset = df[df['type'] == token_type]
        
        # Different text styling for original vs similar words
        if token_type == 'original':
            textfont = dict(size=12, color='white', family="Arial Black")
            textposition = 'top center'
        else:
            textfont = dict(size=10, color='rgba(255,255,255,0.7)', family="Arial")
            textposition = 'bottom center'
        
        fig.add_trace(
            go.Scatter3d(
                x=subset['x'],
                y=subset['y'],
                z=subset['z'],
                mode='text',
                text=subset['token'],
                hoverinfo='none',
                textposition=textposition,
                textfont=textfont,
                showlegend=False
            )
        )
    
    # Add lines connecting similar words to their original words
    if 'similar' in df['type'].values:
        similar_df = df[df['type'] == 'similar']
        for _, row in similar_df.iterrows():
            if row['related_to'] is not None:
                original_row = df[(df['type'] == 'original') & (df['token'] == row['related_to'])]
                if not original_row.empty:
                    x_line = [original_row['x'].values[0], row['x']]
                    y_line = [original_row['y'].values[0], row['y']]
                    z_line = [original_row['z'].values[0], row['z']]
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=x_line, y=y_line, z=z_line,
                            mode='lines',
                            line=dict(color='rgba(100,100,100,0.4)', width=2),
                            showlegend=False
                        )
                    )
    
    # Add lines connecting semantically related words
    if similarity_matrix is not None and all(t == "original" for t in token_types[:n]):
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i][j] > 0.3:  # Threshold for drawing a line
                    x_line = [df['x'].iloc[i], df['x'].iloc[j]]
                    y_line = [df['y'].iloc[i], df['y'].iloc[j]]
                    z_line = [df['z'].iloc[i], df['z'].iloc[j]]
                    
                    # Line thickness based on similarity
                    line_width = similarity_matrix[i][j] * 5
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=x_line, y=y_line, z=z_line,
                            mode='lines',
                            line=dict(color='rgba(255,255,255,0.7)', width=line_width),
                            name=f"{tokens[i]}-{tokens[j]}",
                            showlegend=False
                        )
                    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimensi 1',
            yaxis_title='Dimensi 2',
            zaxis_title='Dimensi 3',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=30),
        legend_title_text='Tipe Kata'
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

# Calculate similarity matrix for valid tokens
similarity_matrix = calculate_similarity_matrix(valid_tokens, model)

# Display similarity heatmap
st.subheader("Word-to-Word Similarity Matrix")
fig_heatmap = px.imshow(
    similarity_matrix,
    labels=dict(x="Words", y="Words", color="Similarity"),
    x=valid_tokens,
    y=valid_tokens,
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Add connections between words with high similarity
word_connections = []
for i in range(len(valid_tokens)):
    for j in range(i+1, len(valid_tokens)):
        if similarity_matrix[i][j] > 0.3:  # Threshold for connection
            word_connections.append((valid_tokens[i], valid_tokens[j], similarity_matrix[i][j]))

if word_connections:
    st.subheader("Strong Word Connections")
    for word1, word2, sim in sorted(word_connections, key=lambda x: x[2], reverse=True):
        st.write(f"**{word1}** and **{word2}**: Similarity = {sim:.4f}")

if __name__ == "__main__":
    main()