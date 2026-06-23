import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Load Assets (Cached so they only load once) ---
@st.cache_resource
def load_assets():
    # Load the trained LSTM model
    model = load_model("lstm_model.h5")
    
    # Load the tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    # Load the max_len integer
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
        
    return model, tokenizer, max_len

model, tokenizer, max_len = load_assets()

# Create the reverse dictionary to map IDs back to words
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# --- 2. Define Core Functions ---
def predictor(model, tokenizer, text, max_len):
    text = text.lower()
    # Convert text to sequence, unbox it, and pad it to the exact length
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')
    
    # Predict the next word ID
    pred = model.predict(seq, verbose=0)
    pred_index = np.argmax(pred)
    
    # Return the actual word
    return index_to_word.get(pred_index, "")

def generate_text(model, tokenizer, seed_text, max_len, n_words):
    result_text = seed_text
    for _ in range(n_words):
        next_word = predictor(model, tokenizer, result_text, max_len)
        if next_word == "":
            break
        result_text += " " + next_word
    return result_text

# --- 3. Streamlit UI Design ---
st.title("Next Word Predictor 🔮")
st.write("Enter a starting phrase, and the LSTM will generate the rest of the quote based on its training data.")

# Input fields
seed_input = st.text_input("Start typing here:", placeholder="e.g., the world is full of")
word_count = st.slider("How many words to generate?", min_value=1, max_value=50, value=10)

# Generate Button
if st.button("Generate Quote"):
    if seed_input.strip() == "":
        st.warning("Please enter some starting words first!")
    else:
        with st.spinner("Generating text..."):
            final_output = generate_text(model, tokenizer, seed_input, max_len, word_count)
            st.success(final_output)