# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from collections import Counter

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Text preprocessing functions
def simplify(text):
    '''Function to handle the diacritics in the text'''
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)

def remove_stopwords(text):
    '''Function to remove the stop words from the text corpus'''
    clean_text = [word for word in text if not word in stop_words]
    return clean_text

def remove_hashsymbols(text):
    '''Function to remove the hashtag symbol from the text'''
    pattern = re.compile(r'#')
    text = ' '.join(text)
    clean_text = re.sub(pattern, '', text)
    return tokenizer.tokenize(clean_text)

def rem_digits(text):
    '''Function to remove the digits from the list of strings'''
    no_digits = []
    for word in text:
        no_digits.append(re.sub(r'\d', '', word))
    return ' '.join(no_digits)

def rem_nonalpha(text):
    '''Function to remove the non-alphanumeric characters from the text'''
    text = [word for word in text if word.isalpha()]
    return text

# Initialize tokenizer and stopwords
tokenizer = TweetTokenizer(preserve_case=True)
stop_words = stopwords.words('english')
# Add additional stop words
additional_list = ['amp', 'rt', 'u', "can't", 'ur']
for words in additional_list:
    stop_words.append(words)

# Define common aggressive words/hate speech terms
# This is a simplified list, you can expand it with more comprehensive datasets
AGGRESSIVE_WORDS = {
    'hate': 8,
    'kill': 10,
    'die': 7, 
    'stupid': 5,
    'idiot': 6,
    'dumb': 5,
    'fool': 4,
    'garbage': 5,
    'trash': 5,
    'worthless': 7,
    'ugly': 4,
    'disgust': 6,
    'violent': 7,
    'attack': 6,
    'terrorist': 9,
    'disgusting': 7,
    'horrible': 6,
    'evil': 7,
    'terrible': 5,
    'worst': 5,
    'destroy': 7,
    'ugly': 5,
    'sick': 4,
    'pathetic': 6,
    'loser': 5,
    'filthy': 6,
    'racist': 9,
    'bigot': 8,
    'sexist': 8,
    'moron': 6,
    'hate': 8,
    'loathe': 7,
    'despise': 7,
    'damn': 4,
    'hell': 4,
    'retard': 9,
    'scum': 7,
    'crap': 5,
}

def identify_aggressive_words(text):
    """
    Identify aggressive words in the text and return a dictionary with:
    - List of identified aggressive words
    - Severity score
    - Highlighted text with aggressive words marked
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Find aggressive words
    aggressive_found = [word for word in words if word in AGGRESSIVE_WORDS]
    
    # Calculate severity score (0-100)
    severity = 0
    if aggressive_found:
        # Sum the scores of all found aggressive words and normalize to 0-100
        total_weight = sum(AGGRESSIVE_WORDS[word] for word in aggressive_found)
        max_possible = 10 * len(words)  # Maximum possible severity
        severity = min(100, int((total_weight / max_possible) * 100))
    
    # Create highlighted text
    highlighted_text = text
    for word in set(aggressive_found):
        pattern = re.compile(rf'\b{word}\b', re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span class="highlight">{word}</span>', highlighted_text)
    
    return {
        'aggressive_words': aggressive_found,
        'severity': severity,
        'highlighted_text': highlighted_text,
        'count': len(aggressive_found)
    }

# Function to predict hate speech on new data
def predict_hate_speech(text, model, vectorizer):
    """
    Predict whether new text contains hate speech
    
    Parameters:
    -----------
    text : str or list of str
        New text to classify
    model : trained classifier
        The trained model to use for prediction
    vectorizer : fitted TfidfVectorizer
        The vectorizer used to transform text to features
        
    Returns:
    --------
    predictions : array
        Array of predictions (0: not hate speech, 1: hate speech)
    """
    # Preprocess the text
    if isinstance(text, str):
        text = [text]
    
    processed_text = []
    original_text = text[0]  # Store original for aggressive word analysis
    
    for t in text:
        # Apply same preprocessing steps
        t = simplify(t)
        t = re.sub(r'@\w+', '', t)
        t = re.sub(r'http\S+', '', t)
        tokens = tokenizer.tokenize(t)
        tokens = [word for word in tokens if not word in stop_words]
        tokens = remove_hashsymbols(tokens)
        tokens = rem_digits(tokens)
        tokens = tokenizer.tokenize(tokens)
        tokens = rem_nonalpha(tokens)
        processed_text.append(' '.join(tokens))
    
    # Transform text to feature vectors
    X = vectorizer.transform(processed_text)
    
    # Make prediction
    predictions = model.predict(X)
    proba = model.predict_proba(X)
    
    # Add aggressive word analysis
    aggressive_analysis = identify_aggressive_words(original_text)
    
    return predictions, proba, aggressive_analysis

# Load or train the model
def load_or_train_model():
    model_path = './hate_speech_model.joblib'
    vectorizer_path = './vectorizer.joblib'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Loading existing model and vectorizer...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("Training new model...")
        # Load the dataset
        try:
            tweet = pd.read_csv('./TwitterHate.csv', delimiter=',', engine='python', encoding='utf-8-sig')
        except:
            return None, None, "Dataset not found. Please place TwitterHate.csv in the application directory."
        
        # Drop the id column if it exists
        if 'id' in tweet.columns:
            tweet.drop('id', axis=1, inplace=True)
        
        # Create a copy of the original data
        df = tweet.copy()
        
        # Text cleaning pipeline
        # 1. Handle diacritics
        df['tweet'] = df['tweet'].apply(simplify)
        
        # 2. Remove user handles
        df['tweet'].replace(r'@\w+', '', regex=True, inplace=True)
        
        # 3. Remove URLs
        df['tweet'].replace(r'http\S+', '', regex=True, inplace=True)
        
        # 4. Tokenize tweets
        df['tweet'] = df['tweet'].apply(tokenizer.tokenize)
        
        # 5. Remove stopwords
        df['tweet'] = df['tweet'].apply(remove_stopwords)
        
        # 6. Remove hash symbols
        df['tweet'] = df['tweet'].apply(remove_hashsymbols)
        
        # 7. Remove digits
        df['tweet'] = df['tweet'].apply(rem_digits)
        df['tweet'] = df['tweet'].apply(tokenizer.tokenize)
        
        # 8. Remove non-alphanumeric characters
        df['tweet'] = df['tweet'].apply(rem_nonalpha)
        
        # Join tokens back to form strings
        df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x))
        
        # Split data into features and target
        X = df['tweet']
        y = df['label']
        
        # Feature extraction using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(X)
        
        # Define class weights to handle imbalance (1:13 ratio)
        weights = {0: 1.0, 1: 13.0}
        
        # Use the optimal parameters from the RandomizedSearchCV
        model = LogisticRegression(C=0.16731783677034165, penalty='l2', solver='liblinear', class_weight=weights)
        
        # Fit the model
        model.fit(X, y)
        
        # Save the model and vectorizer
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
    
    return model, vectorizer, None

model, vectorizer, error_message = load_or_train_model()

@app.route('/')
def home():
    return render_template('index.html', error=error_message)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/predict', methods=['POST'])


def predict():
    if request.method == 'POST':
        if model is None or vectorizer is None:
            return jsonify({'error': error_message})
        
        text = request.form['tweet']
        predictions, probabilities, aggressive_analysis = predict_hate_speech([text], model, vectorizer)
        
        result = {
            'prediction': int(predictions[0]),
            'probability': float(probabilities[0][1]),
            'is_hate_speech': bool(predictions[0] == 1),
            'processed_text': text,
            'aggressive_words': aggressive_analysis['aggressive_words'],
            'aggressive_count': aggressive_analysis['count'],
            'severity': aggressive_analysis['severity'],
            'highlighted_text': aggressive_analysis['highlighted_text']
        }
        
        return jsonify(result)

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/report', methods=['POST'])

def report_hate_speech():
    """API endpoint to report hate speech for further review"""
    if request.method == 'POST':
        report_data = request.json
        
        # In a production app, this would store the report in a database
        # For demo purposes, we'll just return success
        
        # Example of what you might save:
        # - report_data['text']: The reported text
        # - report_data['reportReason']: Reason for reporting
        # - report_data['timestamp']: When it was reported
        # - report_data['probability']: Model's confidence score
        
        return jsonify({'success': True, 'message': 'Report submitted successfully'})

if __name__ == '__main__':
    app.run(debug=True)
