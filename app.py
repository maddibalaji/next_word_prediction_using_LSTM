import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



def predict_next_word(model, tokenizer, input_text, max_sequence_len):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict the next word
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    
    # Get the word corresponding to the predicted index
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

max_sequence_len = model.input_shape[1] + 1  # Adjusting for the input length

st.title("Next Word Prediction")
input_text = st.text_input("Enter a sentence or sequence of words:")
if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted next word: {next_word}")