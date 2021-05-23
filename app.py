import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

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

st.sidebar.markdown("---Numner of Tweets by Sentiment---")
select = st.sidebar.selectbox("Graph Type", ["Histogram", "Pie Chart"], key="1")

sentiment_count = tweets["airline_sentiment"].value_counts()
sentiment_count = pd.DataFrame(
    {"Sentiment": sentiment_count.index, "Tweets": sentiment_count.values}
)

if not st.sidebar.checkbox("Hide", False):
    st.markdown("---Number of Tweets by Sentiment---")
    if select == "Histogram":
        fig = px.bar(
            sentiment_count, x="Sentiment", y="Tweets", color="Tweets", height=500
        )
        st.plotly_chart(fig)

    else:
        fig = px.pie(sentiment_count, values="Tweets", names="Sentiment")
        st.plotly_chart(fig)
