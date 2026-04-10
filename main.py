from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = load_model("simple_RNN.h5")
word_index = imdb.get_word_index()

reverse_word_index = {value: key for key, value in word_index.items()}


# function to preroces the user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# predition function
def predict_sentiment(review):
    preprocesed_input = preprocess_text(review)
    prediction = model.predict(preprocesed_input)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment, prediction[0][0]


# stremlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

# user input
user_input = st.text_area("Movie Review", "Type your review here...")

if st.button("Classify"):
    preporcessed_input = preprocess_text(user_input)
    prediction = model.predict(preporcessed_input)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
else:
    st.write("Please enter a review and click 'Classify' to see the prediction.")
