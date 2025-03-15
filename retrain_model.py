import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize tokenizer
tokenizer = TweetTokenizer(preserve_case=True)

# Download stopwords
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

# Load your dataset
df = pd.read_csv('TwitterHate.csv')  # Replace with your dataset path

# Print column names to debug
print(df.columns)

# Ensure the correct column name is used
df['processed_text'] = df['tweet'].apply(preprocess_text)  # Replace 'tweet' with the correct column name if different

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['label']  # Replace with your label column

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model and vectorizer
joblib.dump(model, 'logreg_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer retrained and saved successfully.")
