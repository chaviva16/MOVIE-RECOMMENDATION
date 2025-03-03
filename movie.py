import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved models
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("cosine_similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

# Load movie dataset
movies = pd.read_csv("movies.csv")


# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations based on what you like!")

# Movie selection
selected_movie = st.selectbox("Choose a movie you like:", movies["title"].values)

def recommend_movies(movie_title):
    if movie_title not in movies["title"].values:
        return ["Movie not found!"]
    
    idx = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended_movie_indices = [i[0] for i in scores[1:6]]
    return movies.iloc[recommended_movie_indices]["title"].values

if st.button("Recommend"): 
    recommendations = recommend_movies(selected_movie)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
