import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import zipfile
import os

# Page configuration
st.set_page_config(page_title="Music Recommender", layout="wide")

# Extract data from ZIP file if needed
@st.cache_data
def load_data():
    # Check if data.csv already exists
    if os.path.exists("data.csv"):
        try:
            return pd.read_csv("data.csv")
        except Exception as e:
            st.error(f"Error reading data.csv: {e}")
    
    # Try to extract from archive.zip
    if os.path.exists("archive.zip"):
        try:
            with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
                # List contents to debug
                file_list = zip_ref.namelist()
                st.write("Files in archive:", file_list)
                
                # Extract all files
                zip_ref.extractall(".")
                
                # Try to find CSV file
                csv_files = [f for f in file_list if f.endswith('.csv')]
                if csv_files:
                    csv_file = csv_files[0]  # Use first CSV file found
                    return pd.read_csv(csv_file)
                else:
                    st.error("No CSV file found in archive.zip")
                    return None
                    
        except Exception as e:
            st.error(f"Error extracting archive.zip: {e}")
            return None
    
    # Debug: Show current directory contents
    st.error("Data file not found!")
    st.write("Current directory:", os.getcwd())
    st.write("Files in directory:", os.listdir("."))
    return None

# Load data with error handling
try:
    data = load_data()
    if data is None:
        st.stop()
    st.success(f"Data loaded successfully! Shape: {data.shape}")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Columns used for similarity calculations
number_cols = [
    'valence', 'year', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
    'mode', 'popularity', 'speechiness', 'tempo'
]

# Check if required columns exist
missing_cols = [col for col in number_cols if col not in data.columns]
if missing_cols:
    st.error(f"Missing columns in data: {missing_cols}")
    st.write("Available columns:", list(data.columns))
    st.stop()

# Scale and standardize the numerical features
@st.cache_data
def prepare_scalers(data_df):
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    normalized_data = min_max_scaler.fit_transform(data_df[number_cols])
    scaled_normalized_data = standard_scaler.fit_transform(normalized_data)
    return min_max_scaler, standard_scaler, scaled_normalized_data

min_max_scaler, standard_scaler, scaled_normalized_data = prepare_scalers(data)

# Function to retrieve song data
def get_song_data(name, data):
    try:
        return data[data['name'].str.lower() == name.lower()].iloc[0]
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
    seed_song_names = [song['name'].lower() for song in seed_songs]
    
    for idx in indices:
        song_name = data.iloc[idx]['name']
        if (song_name.lower() not in seed_song_names and 
            song_name.lower() not in [song['name'].lower() for song in rec_songs]):
            rec_songs.append(data.iloc[idx])
            if len(rec_songs) == n_recommendations:
                break

    return pd.DataFrame(rec_songs)[metadata_cols].to_dict(orient='records')

# Streamlit app structure
st.title('🎵 Music Recommender with Weighted Features')

# Initialize session state for selected songs
if 'selected_songs' not in st.session_state:
    st.session_state.selected_songs = []

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Song Search & Selection")
    
    # Input for song names
    song_input = st.text_input("Enter a song name:", placeholder="Type to search for songs...")
    
    if song_input:
        # Search for matching songs
        matching_songs = data[data['name'].str.lower().str.contains(song_input.lower(), na=False)]
        
        if not matching_songs.empty:
            # Display matching songs with radio button
            song_options = []
            for _, row in matching_songs.head(10).iterrows():  # Limit to 10 results
                option_text = f"{row['name']} by {row['artists']} ({row['year']})"
                song_options.append((option_text, row['name']))
            
            if song_options:
                selected_option = st.radio(
                    "Select a song to add:",
                    [opt[0] for opt in song_options],
                    key="song_radio"
                )
                
                if st.button("➕ Add Song"):
                    selected_song_name = next(opt[1] for opt in song_options if opt[0] == selected_option)
                    if selected_song_name not in st.session_state.selected_songs:
                        st.session_state.selected_songs.append(selected_song_name)
                        st.success(f"Added: {selected_song_name}")
                    else:
                        st.warning("Song already selected!")
        else:
            st.warning("No matching songs found")
    
    # Display selected songs
    if st.session_state.selected_songs:
        st.subheader("Selected Songs:")
        for i, song in enumerate(st.session_state.selected_songs):
            col_song, col_remove = st.columns([4, 1])
            with col_song:
                st.write(f"{i+1}. {song}")
            with col_remove:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.selected_songs.remove(song)
                    st.experimental_rerun()

with col2:
    # Number of recommendations
    n_recommendations = st.slider("Number of recommendations:", 1, 30, 10)
    
    # Clear all songs button
    if st.button("🗑️ Clear All Songs"):
        st.session_state.selected_songs = []
        st.experimental_rerun()

# Weights for personalization
st.subheader('🎛️ Customize Your Music Taste')
st.write("Adjust these sliders to personalize your recommendations:")

col1, col2, col3 = st.columns(3)

weights = {}
for i, col in enumerate(number_cols):
    column = [col1, col2, col3][i % 3]
    with column:
        # More user-friendly names for features
        display_names = {
            'valence': 'Positivity',
            'acousticness': 'Acoustic Feel',
            'danceability': 'Danceability',
            'energy': 'Energy Level',
            'explicit': 'Explicit Content',
            'instrumentalness': 'Instrumental',
            'liveness': 'Live Performance',
            'loudness': 'Loudness',
            'speechiness': 'Speech-like',
            'tempo': 'Tempo (BPM)',
            'popularity': 'Popularity',
            'year': 'Release Year',
            'duration_ms': 'Song Length',
            'key': 'Musical Key',
            'mode': 'Major/Minor'
        }
        
        display_name = display_names.get(col, col.title())
        weights[col] = st.slider(f"{display_name}", 0.0, 2.0, 1.0, step=0.1)

# Normalize weights to ensure they are on a consistent scale
weights_array = np.array([weights[col] for col in number_cols])
weights_array = weights_array / np.linalg.norm(weights_array)

# Recommendation section
st.subheader('🎵 Get Your Recommendations')

if st.button('🚀 Get Recommendations', type="primary"):
    if not st.session_state.selected_songs:
        st.warning("Please select at least one song first!")
    else:
        with st.spinner("Finding perfect songs for you..."):
            seed_songs = [{'name': name} for name in st.session_state.selected_songs]
            recommended_songs = recommend_songs(seed_songs, data, n_recommendations, weights_array)
            
            if not recommended_songs:
                st.warning("No recommendations available. Try different songs or adjust weights.")
            else:
                st.success(f"Found {len(recommended_songs)} recommendations!")
                
                # Display recommendations in a nice format
                for i, song in enumerate(recommended_songs, 1):
                    with st.container():
                        st.write(f"**{i}. {song['name']}** by *{song['artists']}* ({song['year']})")
                
                # Create visualization
                recommended_df = pd.DataFrame(recommended_songs)
                fig = px.bar(
                    recommended_df, 
                    y='name', 
                    x=list(range(len(recommended_df), 0, -1)), 
                    title='🎵 Your Personalized Music Recommendations',
                    orientation='h',
                    color=list(range(len(recommended_df))),
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    xaxis_title='Recommendation Rank',
                    yaxis_title='Songs',
                    showlegend=False,
                    height=max(400, len(recommended_songs) * 30),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)

# Data visualization section
st.header('📊 Music Data Insights')

# Check if required columns exist for visualizations
viz_cols = ['popularity', 'release_date', 'artists']
available_viz_cols = [col for col in viz_cols if col in data.columns]

if 'popularity' in data.columns:
    # Display top songs by popularity
    st.subheader('🔥 Top Songs by Popularity')
    top_songs = data.nlargest(20, 'popularity')
    fig_popularity = px.bar(
        top_songs, 
        x='popularity', 
        y='name', 
        orientation='h', 
        title='Most Popular Songs',
        color='popularity',
        color_continuous_scale='reds'
    )
    fig_popularity.update_layout(showlegend=False, height=600)
    st.plotly_chart(fig_popularity, use_container_width=True)

# Songs per decade (if release_date exists)
if 'release_date' in data.columns:
    try:
        data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
        data['release_decade'] = (data['release_date'].dt.year // 10) * 10
        decade_counts = data['release_decade'].value_counts().sort_index()
        
        st.subheader('📅 Songs by Decade')
        fig_decades = px.bar(
            x=decade_counts.index, 
            y=decade_counts.values, 
            labels={'x': 'Decade', 'y': 'Number of Songs'}, 
            title='Music Through the Decades',
            color=decade_counts.values,
            color_continuous_scale='blues'
        )
        fig_decades.update_layout(xaxis_type='category', height=500)
        st.plotly_chart(fig_decades, use_container_width=True)
    except Exception as e:
        st.write("Could not process release date data")

# Distribution of song attributes
st.subheader('📈 Song Feature Distributions')
attribute_to_plot = st.selectbox('Select a feature to analyze:', number_cols)
fig_histogram = px.histogram(
    data, 
    x=attribute_to_plot, 
    nbins=30, 
    title=f'Distribution of {attribute_to_plot.title()}',
    color_discrete_sequence=['skyblue']
)
fig_histogram.update_layout(height=400)
st.plotly_chart(fig_histogram, use_container_width=True)

# Artists with the most songs (if artists column exists)
if 'artists' in data.columns:
    st.subheader('🎤 Most Prolific Artists')
    try:
        # Clean artist names
        artist_series = data['artists'].str.replace(r"[\[\]']", "", regex=True)
        top_artists = artist_series.value_counts().head(15)
        
        fig_top_artists = px.bar(
            x=top_artists.values,
            y=top_artists.index, 
            orientation='h',
            labels={'x': 'Number of Songs', 'y': 'Artist'}, 
            title='Artists with Most Songs',
            color=top_artists.values,
            color_continuous_scale='greens'
        )
        fig_top_artists.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_top_artists, use_container_width=True)
    except Exception as e:
        st.write("Could not process artist data")

# Footer
st.markdown("---")
st.markdown("🎵 **Music Recommender System** - Discover your next favorite song!")
