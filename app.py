from flask import Flask, render_template, request, redirect, url_for, flash


app = Flask(__name__)
app.secret_key = 'your_secret_key'

emotions = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust']
emotion_sentences = {emotion: [] for emotion in emotions}
current_emotion_index = 0

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


@app.route('/reset')
def reset():
    global current_emotion_index, emotion_sentences
    emotion_sentences = {emotion: [] for emotion in emotions}
    current_emotion_index = 0
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
