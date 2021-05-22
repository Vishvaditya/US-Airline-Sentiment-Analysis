import pandas as pd
import numpy as np
import streamlit as st

st.title("US Airline Sentiment Analysis")
st.sidebar.title("Sentiment Analysis of Tweets using Streamlit")

st.markdown(
    "This Dashboard visualises sentiment anlaysis of tweets regarding US Airlines"
)


@st.cache(persist=True)  # Function decorator to cache data
def load_data():
    tweets = pd.read_csv("Tweets.csv")  # loaded Tweets.csv file in pandas dataframe
    tweets["tweet_created"] = pd.to_datetime(tweets["tweet_created"])
    return tweets


tweets = load_data()

st.sidebar.subheader("Display Tweet")
random_tweet = st.sidebar.radio("Sentiment", ("Positive", "Negative", "Neutral"))
st.sidebar.markdown(
    tweets.query("airline_sentiment == @random_tweet.lower()")[["text"]]
    .sample(n=1)
    .iat[0, 0]
)


# st.write(tweets)

