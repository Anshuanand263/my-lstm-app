# 🔮 Next-Word Predictor & Quote Generator

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)

## 📖 About the Project
This project is an interactive web application that predicts the next words in a sequence to dynamically generate complete sentences and quotes. 

Under the hood, it uses a **Long Short-Term Memory (LSTM)** neural network trained on a vast dataset of famous quotes. By understanding the context, grammar, and vocabulary of the training data, the model can predict logically and stylistically appropriate word continuations. The project is wrapped in a lightweight **Streamlit** user interface for easy interaction in the browser.

## 💡 Use Cases
While this project is trained specifically on quotes, the underlying LSTM architecture and text-generation principles can be applied to several real-world scenarios:

1. **Creative Writing Assistant:** Helps authors and copywriters overcome writer's block by suggesting the next logical sequence of words based on their initial thought.
2. **Smart Auto-Complete:** Functions similarly to predictive text keyboards on smartphones or smart-compose features in email clients, speeding up the typing process.
3. **Educational Tool for NLP:** Serves as a practical demonstration of how Recurrent Neural Networks (RNNs) and LSTMs handle sequential text data and context memory.
4. **Chatbot Foundation:** The core logic of predicting the next most likely word is the foundational step in building conversational AI agents and specialized chatbots.

## ⚙️ Tech Stack
* **Deep Learning:** TensorFlow & Keras (LSTM, SimpleRNN, Embedding, and Dense layers)
* **Data Processing:** NumPy, Keras Tokenizer, Pickle
* **Frontend UI:** Streamlit
* **Language:** Python

## 📂 Repository Structure
Ensure your local folder contains the following files before running or deploying:

```text
├── app.py                 # The main Streamlit web application script
├── lstm_model.h5          # The pre-trained LSTM neural network
├── tokenizer.pkl          # Saved dictionary mapping words to integer IDs
├── max_len.pkl            # Saved sequence padding length (required by the model)
├── requirements.txt       # Dependencies needed to run the application
└── README.md              # Project documentation