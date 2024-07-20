import streamlit as st
import joblib  
import spacy  
import re
import joblib


# Load your model (adjust as per your model loading)
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example preprocessing functions (adjust based on your implementation)
nlp = spacy.load('en_core_web_sm')  # Example, adjust based on your setup

stop_words = nlp.Defaults.stop_words
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags (if any)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert text to lowercase
    doc = nlp(text)  # Tokenize text using SpaCy
    lemmatized_tokens = [token.lemma_ for token in doc]  # Lemmatize tokens
    cleaned_tokens = [token for token in lemmatized_tokens if token.isalpha()]  # Remove non-alphabetic tokens
    doc_no_stopwords = [token for token in cleaned_tokens if token not in stop_words]  # Remove stopwords
    return ' '.join(doc_no_stopwords)

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    if not preprocessed_text.strip():  # Check if the preprocessed text is empty
        return "Error: Review text is empty after preprocessing."
    # Transform the preprocessed text using the pre-fitted vectorizer
    transformed_text = vectorizer.transform([preprocessed_text])
    sentiment_score = model.predict(transformed_text)  # Use transformed_text for prediction
    return sentiment_score


# Streamlit app code
def main():
    st.title('Customer Review Sentiment Analysis')

    # Text input for user to enter a review
    review_text = st.text_area('Enter your review here:')

    if st.button('Analyze'):
        # Perform sentiment analysis when the button is clicked
        if review_text:
            sentiment = predict_sentiment(review_text)
            st.write(f'Sentiment Analysis Result: {sentiment}')
        else:
            st.warning('Please enter a review to analyze.')


if __name__ == '__main__':
    main()
