import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import neattext.functions as nfx

# Load the dataset
# The dataset contains Udemy course information
data = pd.read_csv('udemy.csv')

# Check the first few rows to understand the structure
print("Dataset Overview:")
print(data.head())

# Clean the data: remove rows with NaN in course_title, subject, or level
data = data.dropna(subset=['course_title', 'subject', 'level'])

# Create Clean_title by cleaning course_title
data['Clean_title'] = data['course_title'].apply(nfx.remove_stopwords)
data['Clean_title'] = data['Clean_title'].apply(nfx.remove_punctuations)

# Combine relevant text for vectorization: title, subject, level
data['combined_text'] = data['Clean_title'] + ' ' + data['subject'] + ' ' + data['level']

# Initialize TfidfVectorizer for better weighting
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined_text to create the vector space
title_vectors = vectorizer.fit_transform(data['combined_text'])

# Function to get course recommendations based on query
def get_recommendations(query, top_n=10):
    """
    Recommends courses similar to the given query based on cosine similarity.

    Parameters:
    query (str): The search query.
    top_n (int): Number of recommendations to return.

    Returns:
    DataFrame: A DataFrame with recommended courses and their similarity scores.
    """
    # Clean the query
    cleaned_query = nfx.remove_stopwords(query)
    cleaned_query = nfx.remove_punctuations(cleaned_query)

    # Vectorize the query
    query_vector = vectorizer.transform([cleaned_query])

    # Compute cosine similarities
    similarities = cosine_similarity(query_vector, title_vectors).flatten()

    # Get top similar courses
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Return the top similar courses
    recommended_courses = data[['course_title', 'url', 'subject']].iloc[top_indices]
    recommended_courses['similarity_score'] = similarities[top_indices]

    return recommended_courses

# Example usage: Recommend courses based on a query
query = "python for data science"
print(f"\nRecommendations for query: {query}")
recommendations = get_recommendations(query)
print(recommendations)