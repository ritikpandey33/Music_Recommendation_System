import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import zipfile
import os

# Function to load data with zip extraction
@st.cache_data
def load_data():
    """Load data with zip extraction if needed"""
    # Try direct CSV first
    if os.path.exists("data.csv"):
        return pd.read_csv("data.csv")
    
    # Try zip extraction
    if os.path.exists("archive.zip"):
        with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if csv_files:
                return pd.read_csv(csv_files[0])
    
    # Try other common names
    for filename in ["music_data.csv", "songs.csv", "spotify_data.csv"]:
        if os.path.exists(filename):
            return pd.read_csv(filename)
    
    st.error("No data file found!")
    st.stop()

# Load data
data = load_data()

# Columns used for similarity calculations
number_cols = [
    'valence', 'year', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
    'mode', 'popularity', 'speechiness', 'tempo'
]

# Scale and standardize the numerical features
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
normalized_data = min_max_scaler.fit_transform(data[number_cols])
scaled_normalized_data = standard_scaler.fit_transform(normalized_data)

# Function to retrieve song data
def get_song_data(name, data):
    try:
        return data[data['name'].str.lower() == name].iloc[0]
    except IndexError:
        return None

# Calculate mean vector of seed songs
def get_mean_vector(song_list, data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song['name'], data)
        if song_data is None:
            return None
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    return np.mean(song_vectors, axis=0)

# Hybrid similarity with weighted features
def hybrid_similarity_with_weights(song_center, scaled_data, weights):
    # Apply weights to the song center and scaled data
    weighted_center = song_center * weights
    weighted_data = scaled_data * weights
    
    # Calculate Euclidean and Cosine distances
    euclidean_distances = cdist([weighted_center], weighted_data, 'euclidean')
    cosine_similarities = cosine_similarity([weighted_center], weighted_data)
    hybrid_scores = 0.5 * (1 - cosine_similarities) + 0.5 * euclidean_distances
    return hybrid_scores[0]

# Recommendation function
def recommend_songs(seed_songs, data, n_recommendations=10, weights=None):
    metadata_cols = ['name', 'artists', 'year']
    song_center = get_mean_vector(seed_songs, data)
    if song_center is None:
        return []

    # Scale and standardize the song center vector
    normalized_song_center = min_max_scaler.transform([song_center])
    scaled_normalized_song_center = standard_scaler.transform(normalized_song_center)[0]

    # Calculate hybrid similarity scores with weights
    hybrid_scores = hybrid_similarity_with_weights(scaled_normalized_song_center, scaled_normalized_data, weights)
    indices = np.argsort(hybrid_scores)

    rec_songs = []
    for idx in indices:
        song_name = data.iloc[idx]['name']
        if song_name not in [song['name'] for song in seed_songs] and song_name not in [song['name'] for song in rec_songs]:
            rec_songs.append(data.iloc[idx])
            if len(rec_songs) == n_recommendations:
                break

    return pd.DataFrame(rec_songs)[metadata_cols].to_dict(orient='records')

# Streamlit app structure
st.title('Music Recommender with Weighted Features')

# Input for song names
song_names = ""
song_input = st.text_input("Enter a song name:")
if song_input:
    matching_songs = data[data['name'].str.lower().str.contains(song_input.lower())]
    if not matching_songs.empty:
        selected_song = st.radio(
            "Select a song:",
            matching_songs.apply(lambda x: f"{x['name']} by {x['artists']} ({x['year']})", axis=1).tolist()
        )
        if selected_song:
            song_name = selected_song.split(" by ")[0]
            song_names += f"\n{song_name}" if song_names else song_name
            st.text_area("Selected songs:", value=song_names, disabled=True)
    else:
        st.warning("No matching songs found")

# Number of recommendations
n_recommendations = st.slider("Select the number of recommendations:", 1, 30, 10)
input_song_names = song_names.strip().split('\n') if song_names else []

# Weights for personalization
st.subheader('Set Feature Weights for Personalization')
weights = {}
for col in number_cols:
    weights[col] = st.slider(f"Weight for {col}", 0.0, 2.0, 1.0)

# Normalize weights to ensure they are on a consistent scale
weights_array = np.array([weights[col] for col in number_cols])
weights_array = weights_array / np.linalg.norm(weights_array)

# Recommendation button
if st.button('Recommend'):
    seed_songs = [{'name': name.lower()} for name in input_song_names]
    seed_songs = [song for song in seed_songs if song['name']]
    if not seed_songs:
        st.warning("Please enter at least one song name.")
    else:
        recommended_songs = recommend_songs(seed_songs, data, n_recommendations, weights_array)
        if not recommended_songs:
            st.warning("No recommendations available based on the provided songs.")
        else:
            recommended_df = pd.DataFrame(recommended_songs)
            recommended_df['text'] = recommended_df.apply(lambda row: f"{row.name + 1}. {row['name']} by {row['artists']} ({row['year']})", axis=1)
            fig = px.bar(recommended_df, y='name', x=range(len(recommended_df), 0, -1), title='Recommended Songs', orientation='h', color='name', text='text')
            fig.update_layout(xaxis_title='Recommendation Rank', yaxis_title='Songs', showlegend=False, uniformtext_minsize=20, uniformtext_mode='show', yaxis_showticklabels=False, height=1000, width=1000)
            fig.update_traces(width=1)
            st.plotly_chart(fig)

st.header('Music Data')

# Display top songs by popularity
st.subheader('Top Songs by Popularity')
top_songs = data.nlargest(20, 'popularity')
fig_popularity = px.bar(top_songs, x='popularity', y='name', orientation='h', title='Top Songs by Popularity', color='name')
fig_popularity.update_layout(showlegend=False, height=1000, width=1000)
st.plotly_chart(fig_popularity)

# Songs per decade
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_decade'] = (data['release_date'].dt.year // 10) * 10
decade_counts = data['release_decade'].value_counts().sort_index()

st.subheader('Number of Songs per Decade')
fig_decades = px.bar(x=decade_counts.index, y=decade_counts.values, labels={'x': 'Decade', 'y': 'Number of Songs'}, title='Number of Songs per Decade', color=decade_counts.values)
fig_decades.update_layout(xaxis_type='category', height=1000, width=1000)
st.plotly_chart(fig_decades)

# Distribution of song attributes
st.subheader('Distribution of Song Attributes')
attribute_to_plot = st.selectbox('Select an attribute to plot:', number_cols)
fig_histogram = px.histogram(data, x=attribute_to_plot, nbins=30, title=f'Distribution of {attribute_to_plot}')
fig_histogram.update_layout(height=1000, width=1000)
st.plotly_chart(fig_histogram)

# Artists with the most songs
st.subheader('Artists with Most Songs')
top_artists = data['artists'].str.replace("[", "").str.replace("]", "").str.replace("'", "").value_counts().head(20)
fig_top_artists = px.bar(top_artists, x=top_artists.index, y=top_artists.values, color=top_artists.index, labels={'x': 'Artist', 'y': 'Number of Songs'}, title='Top Artists with Most Songs')
fig_top_artists.update_xaxes(categoryorder='total descending')
fig_top_artists.update_layout(height=1000, width=1000, showlegend=False)
st.plotly_chart(fig_top_artists)
