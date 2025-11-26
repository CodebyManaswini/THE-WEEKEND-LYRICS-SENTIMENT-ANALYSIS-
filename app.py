
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

if not os.path.exists("weeknd_sentiment_model.pkl"):
    with st.spinner("First time launch – training The Weeknd sentiment model... (30–60 sec)"):
        # Load the dataset directly from the repo
        df = pd.read_csv("lyrics_dataset.csv")

        X = df['lyrics']
        y = df['sentiment']

        # Build and train the model
        model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        model.fit(X, y)

        # Save it for next time
        joblib.dump(model, "weeknd_sentiment_model.pkl")
        
    st.success("Model trained and ready!")
    st.balloons()

model = joblib.load("weeknd_sentiment_model.pkl")
df = pd.read_csv("lyrics_dataset.csv")

st.set_page_config(page_title="The Weeknd Lyrics Mood", page_icon="🌙", layout="wide")
st.title("🌙 The Weeknd Lyrics Sentiment Analysis")
st.markdown("**Feel the vibe of every song** – AI-powered mood detection")

option = st.sidebar.radio("Choose", ["Analyze Lyrics", "Explore Discography"])

if option == "Analyze Lyrics":
    st.header("Paste any The Weeknd lyrics")
    user_input = st.text_area("Try: 'I saw you dancing in a crowded room'", height=150)

    if st.button("Predict Mood", type="primary"):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            probs = model.predict_proba([user_input])[0]
            confidence = max(probs)

            mood_colors = {
                "Positive": "🟢", "Negative": "🔴", "Neutral": "⚪",
                "Melancholic": "🟣", "Dark": "⚫"
            }
            st.markdown(f"### {mood_colors.get(prediction, '⚪')} **{prediction}**")
            st.progress(confidence)
            st.write(f"**Confidence**: {confidence:.1%}")

            # Word cloud
            wc = WordCloud(background_color='black', width=800, height=400, colormap='plasma').generate(user_input)
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

else:
    st.header("Emotional Journey Through the Years")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df, y='sentiment', order=df['sentiment'].value_counts().index, palette='dark')
        plt.title("Mood Distribution")
        st.pyplot(fig)
    
    with col2:
        year_sent = df.groupby(['year', 'sentiment']).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(10,6))
        year_sent.plot(kind='area', stacked=True, ax=ax, colormap='Set1', alpha=0.8)
        plt.title("Sentiment Evolution Over Time")
        plt.legend(title="Mood")
        st.pyplot(fig)

st.caption("XO till we overdose | Made for The Weeknd fans")
