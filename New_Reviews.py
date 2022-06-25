import streamlit as st
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob
lemmatizer = WordNetLemmatizer()

import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('vader_lexicon')


st.title("List Of Reviews In cHrome App - positive reviews with low ratings")

df = pd.read_csv("chrome_reviews.csv")
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


stop_words = set(stopwords.words('english'))
stop_words.remove('not')
stop_words.remove('no')

cleaning_text =[]
for text_review in df['Text']:
    text_review= re.sub(r'[^\w\s]','',str(text_review))
    text_review = re.sub(r'\d','',text_review)
    review_token = word_tokenize(text_review.lower().strip()) #convert reviews into lower case and strip leading and tailing spaces followed by spliting sentnece into words
    review_without_stopwords=[]
    for token in review_token:
        if token not in stop_words:
            token= lemmatizer.lemmatize(token)
            review_without_stopwords.append(token)
    cleaned_review = " ".join(review_without_stopwords)
    cleaning_text.append(cleaned_review)

df["cleaned_review"] = cleaning_text
Single_star_reviews = df[df.Star ==1]

sia = SentimentIntensityAnalyzer()
senti_list = []

for i in Single_star_reviews["cleaned_review"]:
    score = sia.polarity_scores(i)
    blob_score = TextBlob(i).sentiment.polarity
    if (score['pos'] >= 0.7):
        senti_list.append('Positive')
    else:
        senti_list.append('Negative/Neutral')
        
Single_star_reviews["sentiment"]= senti_list


st.write(Single_star_reviews.head())

positive_review_with_1_star = Single_star_reviews[Single_star_reviews.sentiment == 'Positive']
positive_review_with_1_star.drop("cleaned_review",axis = 1,inplace=True)

positive_review_with_1_star.head()

st.write(positive_review_with_1_star.head())


df_final=positive_review_with_1_star.head()[positive_review_with_1_star.head().sentiment_score >= .4]

st.write("The list of reviews where the reviews and ratings probably don't match are as below")
st.write(df_final)
