from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles

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
except:
    model = None
    vectorizer = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    warnings = ""
    marked_text = ""

    # Preprocess text
    processed_text = preprocess_text(text)

    # Vectorize text
    text_vector = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(text_vector)[0]
    if prediction == 1:
        result = "Hate Text"
        marked_text = re.sub(r'\b(kill|rape|murder|attack|hurt|harm|die|assault|beat|stab|shoot|threat|violence)\b', r'<mark>\g<0></mark>', text)
        warnings = "Warning: This tweet contains hate speech."
    else:
        result = "Not Hate Text"

    return templates.TemplateResponse("result.html", {"request": request, "prediction": result, "warnings": warnings, "marked_text": marked_text})

if __name__ == '__main__':
    app.run(debug=True)
