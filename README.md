# Multi-Class-Emotion-Detection-from-Text
from google.colab import files
uploaded = files.upload()  # Upload your archive.zip here

import zipfile
import os
# Unzip uploaded archive
with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("emotion_data")
# Confirm files
os.listdir("emotion_data")

import pandas as pd
# Load train.txt
train_path = "emotion_data/train.txt"
df = pd.read_csv(train_path, sep=';', header=None, names=['text', 'emotion_label'])
# Show sample data
df.head()

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)         # remove links
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    return text
# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)
df[['text', 'clean_text']].head()


from sklearn.model_selection import train_test_split
X = df['clean_text']
y = df['emotion_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train size:", len(X_train))
print("Test size:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)
# Fit and transform the data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF Train shape:", X_train_tfidf.shape)
print("TF-IDF Test shape:", X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
# Predict on test set
y_pred = model.predict(X_test_tfidf)
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_emotion(text):
    # Preprocess the text (same as training)
    clean = clean_text(text)
    vec = tfidf_vectorizer.transform([clean])
    # Predict
    prediction = model.predict(vec)[0]
    # Emojis (optional)
    emoji_map = {
        'joy': '😊',
        'anger': '😠',
        'sadness': '😢',
        'fear': '😨',
        'love': '❤️',
        'surprise': '😲'
    }
    emoji = emoji_map.get(prediction, '')
    print(f"Input: {text}")
    print(f"Predicted Emotion: {emoji} {prediction}")

!pip install gradio


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import gradio as gr
# STEP 2: Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()
df["clean_text"] = df["text"].apply(clean_text)
# STEP 4: Train Model
model = LogisticRegression(max_iter=300)
model.fit(X, y)
# STEP 5: Prediction Function
def predict_emotion(text):
    try:
        clean = clean_text(text)
        vec = tfidf_vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        emoji_map = {
            'joy': '😊',
            'anger': '😠',
            'sadness': '😢',
            'fear': '😨',
            'love': '❤️',
            'surprise': '😲'
        }
        emoji = emoji_map.get(prediction.lower(), '🙂')
        return f"{emoji} {prediction.capitalize()}"
    except Exception as e:
        return f"Error: {str(e)}"
# STEP 6: Gradio Interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Type your sentence here..."),
    outputs=gr.Textbox(label="Predicted Emotion"),
    title="Emotion Detection App 😊",
    description="Enter a sentence to detect the emotion."
)
# STEP 7: Launch the App
interface.launch()
