from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
import logging

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/images", StaticFiles(directory="templates/images"), name="images")

templates = Jinja2Templates(directory="templates")


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
    logger.info("Model and vectorizer loaded successfully.")
except Exception as e:
    model = None
    vectorizer = None
    logger.error(f"Error loading model or vectorizer: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model or vectorizer not loaded properly.")
    
    warnings = ""
    marked_text = ""

    # Preprocess text
    processed_text = preprocess_text(text)

    # Vectorize text
    try:
        text_vector = vectorizer.transform([processed_text])
    except Exception as e:
        logger.error(f"Error transforming text: {e}")
        raise HTTPException(status_code=500, detail="Error transforming text.")

    # Make prediction
    prediction = model.predict(text_vector)[0]
    if prediction == 1:
        result = "Hate Text"
        marked_text = re.sub(r'\b(kill|rape|murder|attack|hurt|harm|die|assault|beat|stab|shoot|threat|violence)\b', r'<mark>\g<0></mark>', text)
        warnings = "Warning: This tweet contains hate speech."
    else:
        result = "Not Hate Text"

    return templates.TemplateResponse("result.html", {"request": request, "prediction": result, "warnings": warnings, "marked_text": marked_text})

@app.post("/retrain", response_class=HTMLResponse)
async def retrain(request: Request):
    # Load your dataset
    df = pd.read_csv('your_dataset.csv')  # Replace with your dataset path
    df['processed_text'] = df['text'].apply(preprocess_text)

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

    return templates.TemplateResponse("retrain.html", {"request": request, "message": "Model retrained and saved successfully."})

if __name__ == '__main__':
    app.run(debug=True)
