from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__, static_folder='templates', static_url_path='')


# Initialize tokenizer
tokenizer = TweetTokenizer(preserve_case=True)

# Load stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
additional_list = ['amp','rt','u',"can't",'ur']
stop_words.extend(additional_list)

# Text preprocessing functions
def simplify(text):
    import unicodedata
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)

def preprocess_text(text):
    # Apply all preprocessing steps
    text = simplify(text)
    text = re.sub(r'@\w+', '', text)  # Remove user handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    text = re.sub(r'#', '', text)  # Remove # symbols
    text = re.sub(r'\d', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

# Load model and vectorizer
try:
    model = joblib.load('logreg_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except:
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input text
        text = request.form['text']
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize text
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        result = "Hate Text" if prediction == 1 else "Not Hate Text"
        
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
