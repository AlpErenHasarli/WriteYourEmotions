import pandas as pd
import re
import nltk
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional

# Load the dataset
file_path = 'C:\\Users\\alper\\Desktop\\turkish_emotion_manual_base_dataset.csv'
df = pd.read_csv(file_path, names=['text', 'emotion'], skiprows=1)
print(f'Total lines read (excluding header): {len(df)}')

# Download NLTK data files
nltk.download('stopwords')
from nltk.corpus import stopwords

# Get the stopwords list
stop_words = set(stopwords.words('turkish'))

# Cleaning the text and removing stopwords
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply the cleaning function to the text data
df['text'] = df['text'].apply(clean_text)

# Check the unique values in the emotion column
print('Unique values in the emotion column:', df['emotion'].unique())

# Filter out any rows with unexpected emotion labels
valid_emotions = ['anger', 'fear', 'joy', 'disgust', 'sadness', 'surprise']
df = df[df['emotion'].isin(valid_emotions)]

# Limit the number of data samples to 2700 for each emotion label
df = df.groupby('emotion').apply(lambda x: x.sample(n=2700, random_state=42)).reset_index(drop=True)

# Print the number of lines after filtering and sampling
print(f'Total lines after filtering and sampling: {len(df)}')

# Encode the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['emotion'])

# Split the data into training and testing sets
X = df['text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Save the tokenizer
with open('data/tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

# Convert text data to sequences
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
max_len = 100
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Accuracy: {accuracy*100:.2f}%')

# Save the model
model.save('model/turkish_emotion_model.keras')
