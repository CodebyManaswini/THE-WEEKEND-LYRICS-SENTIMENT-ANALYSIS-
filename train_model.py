
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


try:
    df = pd.read_csv("lyrics_dataset.csv")
    print(f"Dataset loaded: {df.shape[0]} songs from {df['year'].min()}–{df['year'].max()}")
    print("Sample sentiments:", df['sentiment'].value_counts().to_dict())
except FileNotFoundError:
    print("Error: Create 'lyrics_dataset.csv' first! Use the content provided.")
    exit()


X = df['lyrics']
y = df['sentiment']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))),  # Reduced for small dataset
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Train
pipeline.fit(X, y)

joblib.dump(pipeline, 'weeknd_sentiment_model.pkl')
joblib.dump(df, 'lyrics_dataset_processed.pkl')  # Save processed data too
print("Model trained and saved as 'weeknd_sentiment_model.pkl'")
print("Sample prediction on 'Blinding Lights':", pipeline.predict(["I'm blinded by the lights"])[0])
