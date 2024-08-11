import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
with open('logistic_regression_model (1).pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer (1).pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    # Your preprocessing steps here (same as training)
    return text

def predict_article_type(text):
    # Preprocess the input text
    text = preprocess_text(text)
    # Vectorize the preprocessed text
    text_vectorized = vectorizer.transform([text])
    # Make prediction
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Streamlit UI
def main():
    st.title("Article Type Predictor")
    st.write("Enter the text of the article:")

    # Input text area
    user_input = st.text_area("Input")

    # Prediction button
    if st.button("Predict"):
        # Make prediction
        prediction = predict_article_type(user_input)
        article_type = "AI-generated" if prediction == 1 else "Human-written"
        st.write(f"Predicted Article Type: {article_type}")

if __name__ == "__main__":
    main()
