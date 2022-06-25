import streamlit as st
import pandas as pd

st.title("Chrome Reviews - positive reviews with low ratings")
#df = pd.read_csv("car-sales.csv")
uploaded_file = st.file_uploader("Choose a file for checking review",type=["csv"])
st.write("Waiting for input")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(5))

import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()


df.drop(["ID","Review URL", "User Name","Developer Reply", "Version","Review Date","App ID","Thumbs Up"],axis=1,inplace=True)

df_1star= df[df.Star == 1]
st.write("Reviews with 1 rating")
st.write(df_1star.head(100))

polarity = [round(sent.polarity_scores(i)['compound'], 2) for i in df_1star['Text']]
df_1star['sentiment_score'] = polarity
#st.write(df_1star.head(10))


df_final=df_1star[df_1star.sentiment_score >= .4]

st.write("The list of reviews where the reviews and ratings probably don't match are as below")
st.write(df_final)
