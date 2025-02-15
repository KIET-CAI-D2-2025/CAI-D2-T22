import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# Initialize tokenizer
tokenizer = TweetTokenizer(preserve_case=True)

# Download stopwords
nltk.download('stopwords')

# Load data
tweet = pd.read_csv('TwitterHate.csv', delimiter=',', engine='python', encoding='utf-8-sig')
tweet.drop('id', axis=1, inplace=True)

# Prepare data
X = tweet['tweet']
y = tweet['label']

# Enhanced preprocessing
def enhanced_preprocess(text):
    # Add comprehensive aggressive language indicators
    aggressive_words = ['kill', 'rape', 'murder', 'attack', 'hurt', 'harm', 'die', 
                       'assault', 'beat', 'stab', 'shoot', 'threat', 'violence']
    stop_words = stopwords.words('english')
    stop_words.extend(['amp','rt','u',"can't",'ur'] + aggressive_words)
    
    # Enhanced text cleaning with hate speech patterns
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    
    # Detect and emphasize hate speech patterns
    hate_patterns = r'\b(kill|rape|murder|attack|hurt|harm|die|assault|beat|stab|shoot|threat|violence)\b'
    text = re.sub(hate_patterns, r'HATE_SPEECH_\1', text)
    
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply enhanced preprocessing
X = X.apply(enhanced_preprocess)

# Create and fit vectorizer with more features
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,3))
X_vectorized = vectorizer.fit_transform(X)

# Train model with even higher class weight for hate speech
model = LogisticRegression(C=0.05, 
                          penalty='l2', 
                          solver='liblinear', 
                          class_weight={0:1.0, 1:30.0})  # Further increased weight for hate speech
model.fit(X_vectorized, y)

# Save model and vectorizer
joblib.dump(model, 'logreg_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Further enhanced model and vectorizer saved successfully!")
