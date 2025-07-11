# CAI-D2-T22

# Hate Text Detection on Twitter Using NLP and Machine Learning

## 📌 Project Overview
This project aims to detect hate Text (racist or sexist content) on Twitter using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. The model is trained to classify tweets as either **hate Text (1)** or **non-hate Text (0)** based on their textual content.

## 🛠️ Features
- **Text Preprocessing**: Cleaning tweets by removing mentions, URLs, special characters, and stop words.
- **Feature Extraction**: Using TF-IDF vectorization to convert text into numerical representations.
- **Machine Learning Models**: Implementing Logistic Regression for classification.
- **Hyperparameter Tuning**: Optimizing model performance using GridSearchCV.
- **Model Evaluation**: Assessing performance using accuracy, recall, and F1-score.

---

## 📂 Dataset
The dataset contains:
- `id`: Unique identifier for each tweet.
- `label`: 0 (Non-hate) or 1 (Hate Text).
- `tweet`: The actual text content of the tweet.

📥 **[Dataset Source](TwitterHate.csv)**

---

## 🚀 Installation & Setup
### Prerequisites
Ensure you have **Python 3.x** installed.

### Clone the Repository
```bash
git clone https://github.com/KIET-CAI-D2-2025/CAI-D2-T22.git
cd CAI-D2-T22
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Project
```bash
python save_model.py
python app.py
```

---

## 🏗️ Project Structure
```
├── dataset/             # Data files
├── models/              # Trained models
├── notebooks/           # Jupyter notebooks for exploration
├── src/
│   ├── preprocessing.py # Data cleaning and preprocessing
│   ├── feature_extraction.py # TF-IDF vectorization
│   ├── model.py         # ML model training & evaluation
│   ├── main.py          # Main script
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── .gitignore           # Ignored files
```

---

## 📊 Results & Performance
- **Accuracy:** 90%+
- **F1-score:** High recall and precision in detecting hate Text.
- **Scalability:** Optimized for processing large volumes of tweets in real-time.

---

## 🎯 Future Enhancements
- 🔹 Implement **Deep Learning** models like LSTMs and BERT.
- 🔹 Add **real-time Twitter API integration**.
- 🔹 Expand detection to **multiple languages**.
- 🔹 Improve feature engineering using **word embeddings (Word2Vec, GloVe)**.

---

## 📜 References
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Kaggle Dataset](https://www.kaggle.com/datasets)

---

## 🤝 Contributing
Contributions are welcome! Feel free to **fork**, create a new branch, and submit a **pull request**.

```bash
git checkout -b feature-branch
git commit -m "Add new feature"
git push origin feature-branch
```

---

## 📧 Contact
For any inquiries, reach out via:
- 📩 Email: arjunnadipalli@gmail.com

### ⭐ If you found this project useful, give it a **star**! ⭐
