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

# Title first
st.title('🎵 Music Recommender with Weighted Features')

# Function to load data safely
@st.cache_data
def load_data_safely():
    """Load data with comprehensive error handling and debugging"""
    
    # Show current directory for debugging
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)
    
    st.info("🔍 Checking for data files...")
    st.write(f"Current directory: {current_dir}")
    st.write(f"Files found: {files_in_dir}")
    
    # Method 1: Try to load data.csv directly
    if "data.csv" in files_in_dir:
        try:
            st.info("📂 Found data.csv, loading...")
            data = pd.read_csv("data.csv")
            st.success(f"✅ Successfully loaded data.csv! Shape: {data.shape}")
            return data
        except Exception as e:
            st.warning(f"❌ Could not load data.csv: {str(e)}")
    
    # Method 2: Try to extract from archive.zip
    if "archive.zip" in files_in_dir:
        try:
            st.info("📦 Found archive.zip, extracting...")
            with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
                file_list = zip_ref.namelist()
                st.write(f"Files in archive: {file_list}")
                
                # Extract all files
                zip_ref.extractall(".")
                st.success("✅ Files extracted successfully!")
                
                # Look for CSV files
                csv_files = [f for f in file_list if f.endswith('.csv')]
                
                if csv_files:
                    csv_file = csv_files[0]
                    st.info(f"📊 Loading CSV file: {csv_file}")
                    data = pd.read_csv(csv_file)
                    st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                    return data
                else:
                    st.error("❌ No CSV files found in archive.zip")
                    return None
                    
        except Exception as e:
            st.error(f"❌ Error processing archive.zip: {str(e)}")
            return None
    
    # Method 3: Try different possible filenames
    possible_files = ["data.csv", "music_data.csv", "songs.csv", "spotify_data.csv"]
    for filename in possible_files:
        if filename in files_in_dir:
            try:
                st.info(f"📂 Trying to load {filename}...")
                data = pd.read_csv(filename)
                st.success(f"✅ Successfully loaded {filename}! Shape: {data.shape}")
                return data
            except Exception as e:
                st.warning(f"❌ Could not load {filename}: {str(e)}")
                continue
    
    # If nothing works, show error
    st.error("❌ No valid data file found!")
    st.error("Please ensure one of the following exists in your repository:")
    st.error("- data.csv")
    st.error("- archive.zip containing a CSV file")
    
    return None

# Load data
try:
    data = load_data_safely()
    
    if data is None:
        st.stop()
        
except Exception as e:
    st.error(f"Critical error loading data: {str(e)}")
    st.stop()

# Display data info
st.success(f"🎵 Loaded {len(data)} songs successfully!")
st.write(f"**Columns available:** {', '.join(data.columns.tolist())}")

# Define required columns for similarity calculations
default_number_cols = [
    'valence', 'year', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
    'mode', 'popularity', 'speechiness', 'tempo'
]

# Check which columns actually exist
number_cols = [col for col in default_number_cols if col in data.columns]
missing_cols = [col for col in default_number_cols if col not in data.columns]

if missing_cols:
    st.warning(f"⚠️ Some expected columns are missing: {missing_cols}")
    st.info(f"✅ Using available columns: {number_cols}")

if len(number_cols) < 5:
    st.error("❌ Not enough numeric columns for recommendations. Need at least 5 columns.")
    st.stop()

# Initialize scalers and data preprocessing
@st.cache_data
def prepare_data(data_df, cols):
    """Prepare and scale the data for similarity calculations"""
    try:
        # Fill any missing values
        data_clean = data_df[cols].fillna(data_df[cols].mean())
        
        # Scale the data
        min_max_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()
        
        normalized_data = min_max_scaler.fit_transform(data_clean)
        scaled_normalized_data = standard_scaler.fit_transform(normalized_data)
        
        return min_max_scaler, standard_scaler, scaled_normalized_data, data_clean
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None, None, None, None

# Prepare the data
min_max_scaler, standard_scaler, scaled_normalized_data, clean_data = prepare_data(data, number_cols)

if min_max_scaler is None:
    st.error("❌ Failed to prepare data for processing")
    st.stop()

# Rest of the functions
def get_song_data(name, data):
    """Get song data by name"""
    try:
        matches = data[data['name'].str.lower().str.contains(name.lower(), na=False)]
        if not matches.empty:
            return matches.iloc[0]
        return None
    except Exception:
        return None

def get_mean_vector(song_list, data):
    """Calculate mean vector of seed songs"""
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song['name'], data)
        if song_data is not None:
            try:
                song_vector = song_data[number_cols].values
                song_vectors.append(song_vector)
            except Exception:
                continue
    
    if song_vectors:
        return np.mean(song_vectors, axis=0)
    return None

def hybrid_similarity_with_weights(song_center, scaled_data, weights):
    """Calculate hybrid similarity with weights"""
    try:
        # Apply weights
        weighted_center = song_center * weights
        weighted_data = scaled_data * weights
        
        # Calculate distances
        euclidean_distances = cdist([weighted_center], weighted_data, 'euclidean')
        cosine_similarities = cosine_similarity([weighted_center], weighted_data)
        hybrid_scores = 0.5 * (1 - cosine_similarities) + 0.5 * euclidean_distances
        return hybrid_scores[0]
    except Exception as e:
        st.error(f"Error in similarity calculation: {str(e)}")
        return np.array([])

def recommend_songs(seed_songs, data, n_recommendations=10, weights=None):
    """Generate song recommendations"""
    try:
        # Get required columns for metadata
        metadata_cols = ['name']
        if 'artists' in data.columns:
            metadata_cols.append('artists')
        if 'year' in data.columns:
            metadata_cols.append('year')
        
        song_center = get_mean_vector(seed_songs, data)
        if song_center is None:
            return []

        # Scale the song center
        normalized_song_center = min_max_scaler.transform([song_center])
        scaled_normalized_song_center = standard_scaler.transform(normalized_song_center)[0]

        # Calculate similarity scores
        hybrid_scores = hybrid_similarity_with_weights(scaled_normalized_song_center, scaled_normalized_data, weights)
        
        if len(hybrid_scores) == 0:
            return []
            
        indices = np.argsort(hybrid_scores)

        rec_songs = []
        seed_song_names = [song['name'].lower() for song in seed_songs]
        
        for idx in indices:
            if idx >= len(data):
                continue
                
            song_name = data.iloc[idx]['name']
            if (song_name.lower() not in seed_song_names and 
                song_name.lower() not in [song.get('name', '').lower() for song in rec_songs]):
                
                song_data = {}
                for col in metadata_cols:
                    if col in data.columns:
                        song_data[col] = data.iloc[idx][col]
                
                rec_songs.append(song_data)
                if len(rec_songs) >= n_recommendations:
                    break

        return rec_songs
        
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return []

# Initialize session state
if 'selected_songs' not in st.session_state:
    st.session_state.selected_songs = []

# UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Search & Select Songs")
    
    song_input = st.text_input("Enter a song name:", placeholder="Start typing to search...")
    
    if song_input and len(song_input) > 2:
        # Search for songs
        try:
            matches = data[data['name'].str.lower().str.contains(song_input.lower(), na=False)]
            
            if not matches.empty:
                # Show top 10 matches
                top_matches = matches.head(10)
                
                song_options = []
                for _, row in top_matches.iterrows():
                    artists = row.get('artists', 'Unknown Artist')
                    year = row.get('year', 'Unknown Year')
                    option_text = f"{row['name']} by {artists} ({year})"
                    song_options.append((option_text, row['name']))
                
                if song_options:
                    selected_option = st.radio("Choose a song:", [opt[0] for opt in song_options])
                    
                    if st.button("➕ Add to Selection"):
                        selected_song_name = next(opt[1] for opt in song_options if opt[0] == selected_option)
                        if selected_song_name not in st.session_state.selected_songs:
                            st.session_state.selected_songs.append(selected_song_name)
                            st.success(f"Added: {selected_song_name}")
                            st.rerun()
                        else:
                            st.warning("Song already selected!")
            else:
                st.info("No songs found matching your search")
                
        except Exception as e:
            st.error(f"Search error: {str(e)}")

with col2:
    st.subheader("⚙️ Settings")
    n_recommendations = st.slider("Number of recommendations:", 1, 20, 10)
    
    if st.button("🗑️ Clear All"):
        st.session_state.selected_songs = []
        st.rerun()

# Show selected songs
if st.session_state.selected_songs:
    st.subheader("🎵 Selected Songs:")
    for i, song in enumerate(st.session_state.selected_songs):
        col_song, col_remove = st.columns([5, 1])
        with col_song:
            st.write(f"{i+1}. **{song}**")
        with col_remove:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.selected_songs.remove(song)
                st.rerun()

# Feature weights
st.subheader("🎛️ Customize Your Taste")
st.write("Adjust these to personalize recommendations:")

# Create columns for weights
n_cols = 3
cols = st.columns(n_cols)
weights = {}

for i, col in enumerate(number_cols):
    with cols[i % n_cols]:
        # User-friendly names
        display_names = {
            'valence': 'Happiness',
            'acousticness': 'Acoustic',
            'danceability': 'Danceability',
            'energy': 'Energy',
            'instrumentalness': 'Instrumental',
            'tempo': 'Tempo',
            'popularity': 'Popularity'
        }
        display_name = display_names.get(col, col.title())
        weights[col] = st.slider(display_name, 0.0, 2.0, 1.0, step=0.1, key=f"weight_{col}")

# Normalize weights
weights_array = np.array([weights[col] for col in number_cols])
weights_array = weights_array / (np.linalg.norm(weights_array) + 1e-8)  # Avoid division by zero

# Recommendation button
st.subheader("🚀 Get Recommendations")

if st.button("Find Similar Songs", type="primary"):
    if not st.session_state.selected_songs:
        st.warning("Please select at least one song first!")
    else:
        with st.spinner("Finding recommendations..."):
            seed_songs = [{'name': name} for name in st.session_state.selected_songs]
            recommended_songs = recommend_songs(seed_songs, data, n_recommendations, weights_array)
            
            if recommended_songs:
                st.success(f"Found {len(recommended_songs)} recommendations!")
                
                # Display recommendations
                for i, song in enumerate(recommended_songs, 1):
                    artists = song.get('artists', 'Unknown Artist')
                    year = song.get('year', '')
                    year_str = f" ({year})" if year else ""
                    st.write(f"**{i}. {song['name']}** by *{artists}*{year_str}")
                
                # Create visualization if we have enough data
                if len(recommended_songs) > 1:
                    try:
                        rec_df = pd.DataFrame(recommended_songs)
                        fig = px.bar(
                            rec_df,
                            y='name',
                            x=list(range(len(rec_df), 0, -1)),
                            title="Your Recommendations",
                            orientation='h'
                        )
                        fig.update_layout(
                            xaxis_title="Rank",
                            yaxis_title="Songs",
                            height=max(400, len(rec_df) * 25)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info("Visualization unavailable")
            else:
                st.warning("No recommendations found. Try different songs or adjust weights.")

# Data insights
st.header("📊 Dataset Insights")

try:
    # Basic stats
    st.write(f"**Total Songs:** {len(data):,}")
    
    if 'popularity' in data.columns:
        st.subheader("🔥 Most Popular Songs")
        top_songs = data.nlargest(10, 'popularity')[['name', 'popularity']]
        if 'artists' in data.columns:
            top_songs = data.nlargest(10, 'popularity')[['name', 'artists', 'popularity']]
        st.dataframe(top_songs, use_container_width=True)
    
    # Feature distribution
    if number_cols:
        st.subheader("📈 Feature Distribution")
        selected_feature = st.selectbox("Choose a feature:", number_cols)
        
        fig = px.histogram(data, x=selected_feature, title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.info("Some visualizations unavailable")

st.markdown("---")
st.markdown("🎵 **Music Recommender** - Discover your next favorite songs!")
