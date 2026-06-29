from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Load pre-trained sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english")

# Store analysis history
history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '')
    
    if not text:
        return jsonify({'error': 'Please enter some text'}), 400
    
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    
    sentiment_map = {
        'POSITIVE': 'Positive 😊',
        'NEGATIVE': 'Negative 😞',
        'NEUTRAL': 'Neutral 😐'
    }
    
    history.append({
        'text': text[:100] + '...' if len(text) > 100 else text,
        'sentiment': label,
        'score': score,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    })
    
    return jsonify({
        'sentiment': sentiment_map.get(label, label),
        'score': round(score * 100, 2),
        'label': label.lower()
    })

@app.route('/history')
def get_history():
    return jsonify(history[-20:])

@app.route('/visualize')
def visualize():
    if not history:
        return jsonify({'error': 'No data to visualize'}), 400
    
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#f39c12'}
    plt.pie(sentiment_counts.values, 
            labels=sentiment_counts.index, 
            autopct='%1.1f%%',
            colors=[colors.get(s, '#95a5a6') for s in sentiment_counts.index])
    plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'image': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
