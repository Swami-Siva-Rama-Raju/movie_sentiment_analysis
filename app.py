import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

nltk.download('punkt')
nltk.download('stopwords')

# Page title and subheader
st.title("🔍 Sentiment Classifier")
st.subheader("Analyze the sentiment of a review - positive or negative")


# Load model and vectorizer
try:
    model = pickle.load(open("xgb_clf.pkl", "rb"))
    cv = pickle.load(open("cv.pkl", "rb"))
except FileNotFoundError:
    st.error("❗ Model or vectorizer file not found. Please check file paths.")

# Input text area
text = st.text_area("Enter review text here", placeholder="Type your review...")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punc = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
stop_words.discard("not")

def preprocess_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review).lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words and word not in punc]
    return ' '.join(review)


# Classify button with custom style
if st.button("📊 Analyze Sentiment"):
    if model and cv:
        with st.spinner("Analyzing..."):

            
            result=preprocess_review(text)
            a = cv.transform([result])
            prediction = model.predict(a)
            
            # Display result with colored labels
            if prediction[0] == 1:
                st.success("😊 Positive sentiment detected!")
            else:
                st.error("😟 Negative sentiment detected.")
    else:
        st.warning("⚠️ Model or vectorizer is not loaded correctly. Please try again.")
