import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the pre-trained model and tokenizer
model = load_model('model.keras')

tokenizer = Tokenizer(num_words=5000)

st.title("Movie Reviews Sentiment Analysis ")
st.write("This application predicts the sentiment of movie reviews as positive or negative.")
st.write("Enter a movie review below:")

review = st.text_area("Review", "Type your review here...")
if st.button("Predict"):
    if review:
        sequences = tokenizer.texts_to_sequences([review])
        padded_sequences = pad_sequences(sequences, maxlen=200)
        prediction = model.predict(padded_sequences)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        st.write(f"Sentiment: {sentiment}")
