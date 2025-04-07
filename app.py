from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the model and tokenizer
model = load_model('lstm_text_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(str(text).lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

# Parameters
MAX_SEQUENCE_LENGTH = 150

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        message = request.form['message']
        email_id = request.form['email_id']
        
        # Preprocess and combine
        processed_message = preprocess_text(message)
        processed_email = preprocess_text(email_id)
        combined_text = processed_message + " " + processed_email
        
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([combined_text])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded)[0][0]
        if prediction >= 0.7:
            result = 'ham'
        elif prediction < 0.3:
            result = 'spam'
        else:
            result = 'phishing'
        
        return render_template('analysis.html', result=result, confidence=prediction * 100)
    return render_template('welcome.html')

@app.route('/report', methods=['GET', 'POST'])
def report():
    if request.method == 'POST':
        email_content = request.form['email_content']
        category = request.form['category']
        # Here you could process the report (e.g., save to a database)
        return render_template('report.html', submitted=True)
    return render_template('report.html', submitted=False)

if __name__ == '__main__':
    app.run(debug=True)