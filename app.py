
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

@st.cache_resource
def load_model():
    return joblib.load("weeknd_sentiment_model.pkl")

model = load_model()

@st.cache_data
def load_data():
    return pd.read_csv("lyrics_dataset.csv")

df = load_data()

st.set_page_config(page_title="The Weeknd Lyrics Sentiment", page_icon="🌙", layout="wide")

st.title("🌙 The Weeknd Lyrics Sentiment Analysis")
st.markdown("### Dive into the emotional highs & lows of Abel's discography")

st.sidebar.header("Quick Tools")
option = st.sidebar.radio("Choose Mode", [" Analyze Lyrics", " Explore Discography"])

if option == "Analyze Lyrics":
    st.header("Enter Your Lyrics")
    user_input = st.text_area("Paste lyrics here (or try: 'Blinding lights, no I can't sleep'):", height=150, placeholder="I'm running out of time...")
    
    if st.button(" Predict Mood", type="primary"):
        if user_input.strip():
            # Preprocess
            prediction = model.predict([user_input])[0]
            proba = model.predict_proba([user_input])[0]
            confidence = max(proba)
            
            # Mood colors & icons
            moods = {
                "Positive": {"color": "🟢 #4CAF50", "icon": "😊"},
                "Negative": {"color": "🔴 #F44336", "icon": "😠"},
                "Neutral": {"color": "⚪ #9E9E9E", "icon": "😐"},
                "Melancholic": {"color": "🟣 #9C27B0", "icon": "😢"},
                "Dark": {"color": "⚫ #212121", "icon": "🌑"}
            }
            mood_info = moods.get(prediction, moods["Neutral"])
            
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"### {mood_info['icon']} **Predicted: {prediction}**")
            with col2:
                st.markdown(f"<span style='color:{mood_info['color'].split('#')[1]}; font-size: 2em;'>●</span>", unsafe_allow_html=True)
            
            st.progress(confidence)
            st.info(f"**Confidence**: {confidence:.1%}")
            
            # Breakdown
            st.subheader("Mood Breakdown")
            probs_df = pd.DataFrame({"Mood": model.classes_, "Probability": proba}).sort_values("Probability", ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(data=probs_df.head(3), x="Probability", y="Mood", ax=ax, palette="viridis")
            ax.set_title("Top 3 Predicted Moods")
            st.pyplot(fig)
            
            # Word cloud
            st.subheader("Your Lyrics Word Cloud")
            wc = WordCloud(width=800, height=300, background_color='black', 
                           colormap='plasma', max_words=50).generate(user_input)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning(" Enter some lyrics to analyze!")

else:
    
    st.header(" Discography Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sentiment_counts = df['sentiment'].value_counts()
        sns.countplot(data=df, y='sentiment', order=sentiment_counts.index, palette='dark:plasma')
        ax1.set_title("Mood Distribution")
        ax1.set_xlabel("Song Count")
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots()
        album_counts = df['album'].value_counts().head(10)
        sns.barplot(y=album_counts.index, x=album_counts.values, palette='viridis')
        ax2.set_title("Songs per Album (Top 10)")
        ax2.set_xlabel("Song Count")
        st.pyplot(fig2)
    
  
    st.subheader("Emotional Journey Over Time")
    timeline_df = df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    timeline_df.plot(kind='area', stacked=True, ax=ax3, colormap='Set1', alpha=0.8)
    ax3.set_title("Sentiment Evolution (2011–2022)")
    ax3.set_ylabel("Number of Songs")
    ax3.legend(title="Mood", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    
  
    st.subheader("Sample Tracks")
    sample_df = df[['song_title', 'album', 'year', 'sentiment']].head(10)
    st.dataframe(sample_df, use_container_width=True)

