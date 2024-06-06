from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = 'secret_key'


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, 'data', 'tokenizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'turkish_emotion_model.keras')


with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model(MODEL_PATH)

emotions = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust']
emotion_sentences = {emotion: [] for emotion in emotions}
current_emotion_index = 0

def predict_emotion(sentences):
    sequences = tokenizer.texts_to_sequences(sentences)
    data = pad_sequences(sequences, maxlen=100)
    predictions = model.predict(data)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play', methods=['GET', 'POST'])
def play():
    global current_emotion_index
    if request.method == 'POST':
        sentences = request.form.getlist('sentence')
        if len(sentences) < 3 or any(not sentence for sentence in sentences):
            flash("Please enter 3 different sentences.")
            return redirect(url_for('play'))
        emotion_sentences[emotions[current_emotion_index]] = sentences
        current_emotion_index += 1
        if current_emotion_index >= len(emotions):
            return redirect(url_for('results'))
        return redirect(url_for('play'))
    return render_template('play.html', emotion=emotions[current_emotion_index])

@app.route('/results')
def results():
    global emotion_sentences
    results = {}
    total_success = 0

    for emotion, sentences in emotion_sentences.items():
        predictions = predict_emotion(sentences)
        emotion_index = emotions.index(emotion)
        success_rates = []

        for i, sentence in enumerate(sentences):
            predicted_emotion_index = np.argmax(predictions[i])
            confidence_score = predictions[i][predicted_emotion_index]
            if predicted_emotion_index == emotion_index:
                success_rate = confidence_score
            else:
                success_rate = 0
            success_rates.append(success_rate)


        best_score = max(success_rates)
        results[emotion] = int(best_score * 100)
        total_success += best_score

    overall_success = int((total_success / len(emotions)) * 100)
    return render_template('results.html', results=results, overall=overall_success)

@app.route('/reset')
def reset():
    global current_emotion_index, emotion_sentences
    emotion_sentences = {emotion: [] for emotion in emotions}
    current_emotion_index = 0
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
