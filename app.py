import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("customer_support.csv")

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])

# Response function
def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    if similarity[0, idx] < 0.3:
        return "Sorry, I didn’t understand. Can you rephrase?"
    return df['answer'].iloc[idx]

# Streamlit UI
st.title("🤖 Customer Support Chatbot")
st.write("Ask me a question below:")

user_input = st.text_input("You:")

if user_input:
    st.write("Chatbot:", chatbot_response(user_input))
