from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data (if not present)
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # We'll also need to save vectorizer in model.py

def preprocess_text(text):
    # Clean the text: remove special characters, lowercase, etc.
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize
    tokens = text.split()
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        processed = preprocess_text(text)
        # Transform using vectorizer
        vec = vectorizer.transform([processed])
        prediction = model.predict(vec)[0]
        # Map to sentiment
        sentiment = ['Negative', 'Neutral', 'Positive'][prediction]
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data['text']
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    sentiment = ['Negative', 'Neutral', 'Positive'][prediction]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
