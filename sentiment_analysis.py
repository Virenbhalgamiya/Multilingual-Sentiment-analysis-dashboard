import streamlit as st
import joblib
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from googletrans import Translator
import spacy
# Load the model and vectorizer
nb_model = joblib   .load('naive_bayes_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Title of the app
st.title("Naive Bayes Sentiment Analysis")

# # Sidebar for user input
# st.sidebar.header("Enter Text for Sentiment Analysis")
user_input = st.sidebar.text_area("Input your text here")

def detect_language(text):
    return detect(text)
translator = Translator()
def translate_to_english(text, lang):
    if lang != 'en':
        translated = translator.translate(text, src=lang, dest='en')
        return translated.text
    return text

# # Button to predict sentiment
# if st.sidebar.button("Predict Sentiment"):
#     if user_input:
#         # Preprocess the input text
#         input_tfidf = vectorizer.transform([user_input])
        
#         # Predict sentiment using the Naive Bayes model
#         prediction = nb_model.predict(input_tfidf)
#         sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
#         # Display prediction results
#         st.write(f"Prediction: {sentiment}")
        
#         # Display model's confidence (probability)
#         prediction_prob = nb_model.predict_proba(input_tfidf)
#         positive_prob = prediction_prob[0][1]  # Probability for positive sentiment
#         negative_prob = prediction_prob[0][0]  # Probability for negative sentiment
#         st.write(f"Probability of Positive Sentiment: {positive_prob:.2f}")
#         st.write(f"Probability of Negative Sentiment: {negative_prob:.2f}")
        
#     else:
#         st.warning("Please enter some text to analyze.")

nlp_en = spacy.load('en_core_web_sm')

# Removal part start
# Multi-language preprocessing function
nlp_en = spacy.load('en_core_web_sm')

# Multi-language preprocessing function
def preprocess_text(text, lang='en'):
    if lang != 'en':
        text = translator.translate(text, src=lang, dest='en').text  # Translate to English

    # Preprocess text
    doc = nlp_en(text)
    text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    
    return text

# Language detection function
def detect_language(text):
    return detect(text)

# Sentiment prediction function
def predict_sentiment(text):
    # Detect the language
    lang = detect_language(text)
    
    # Preprocess text based on the detected language
    processed_text = preprocess_text(text, lang)
    
    # Convert text to vector and make prediction
    text_tfidf = vectorizer.transform([processed_text])
    prediction_proba = nb_model.predict_proba(text_tfidf)  # Get probabilities for both classes
    
    positive_prob = prediction_proba[0][0]  # Probability for the positive class
    negative_prob = prediction_proba[0][1]  # Probability for the negative class
    
    # Get predicted sentiment (class with the higher probability)
    sentiment = 1 if positive_prob > negative_prob else 0
    
    return sentiment, positive_prob, negative_prob

# Streamlit app
# st.title('Multilingual Sentiment Analysis')

# Get user input
# user_input = st.text_area("Enter your text")

if st.sidebar.button("Predict Sentiment"):
    if user_input:
        sentiment, positive_prob, negative_prob = predict_sentiment(user_input)
        sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
        
        # Display the result
        st.write(f"Sentiment: {sentiment_label}")
        st.write(f"Probability of Positive: {positive_prob:.2f}")
        st.write(f"Probability of Negative: {negative_prob:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
else:
    st.warning("Please enter some text to analyze.")