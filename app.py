import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import wordcloud

st.set_option(
    "deprecation.showPyplotGlobalUse", False
)  # function to remove Warnings from dashboard

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


# Code snippet to display random tweet based on user selected sentiment
st.sidebar.subheader("Display Tweet")
random_tweet = st.sidebar.radio("Sentiment", ("Positive", "Negative", "Neutral"))
st.sidebar.markdown(
    tweets.query("airline_sentiment == @random_tweet.lower()")[["text"]]
    .sample(n=1)
    .iat[0, 0]
)


st.sidebar.subheader("Number of Tweets by Sentiment")
select = st.sidebar.selectbox("Graph Type", ["Bar Chart", "Pie Chart"], key="1")

sentiment_count = tweets["airline_sentiment"].value_counts()
sentiment_count = pd.DataFrame(
    {"Sentiment": sentiment_count.index, "Tweets": sentiment_count.values}
)


# Creating a Bar chart or Pie chart based on user selection
if not st.sidebar.checkbox("Hide", False):
    st.subheader("---Number of Tweets by Sentiment---")
    if select == "Bar Chart":
        fig = px.bar(
            sentiment_count, x="Sentiment", y="Tweets", color="Tweets", height=500
        )
        st.plotly_chart(fig)

    else:
        fig = px.pie(sentiment_count, values="Tweets", names="Sentiment")
        st.plotly_chart(fig)


# Generating a map to display the  origin of tweets based on time of day selected by the user
st.sidebar.subheader("Location and Time of Tweets")
hour = st.sidebar.slider("Hour of Day", 0, 23)
time_data = tweets[tweets["tweet_created"].dt.hour == hour]
if not st.sidebar.checkbox("Hide", False, key="2"):
    st.subheader("---Tweet location based on time of day---")
    st.markdown(
        "%i tweets between %i:00 and %i:00" % (len(time_data), hour, (hour + 1) % 24)
    )
    st.map(time_data)
    if st.sidebar.checkbox("Show data", False):
        st.write(time_data)


# Plotting a histogram to compare number of tweets by sentiment for different airlines selected by User
st.sidebar.subheader("Airline Tweets by sentiment")
choice = st.sidebar.multiselect(
    "Select Airline(s)",
    ("US Airways", "United", "American", "Southwest", "Delta", "Virgin"),
)

if len(choice) > 0:
    choice_data = tweets[tweets.airline.isin(choice)]
    st.subheader("---Histogram to Compare twets by airlines---")
    fig_choice = px.histogram(
        choice_data,
        x="airline",
        y="airline_sentiment",
        histfunc="count",
        color="airline_sentiment",
        facet_col="airline_sentiment",
        labels={"airline_sentiment": "Tweets"},
        height=600,
        width=800,
    )
    st.plotly_chart(fig_choice)


# Generating a word cloud based on sentiment selected by the user
st.sidebar.subheader("Word Cloud")
sentiment_word = st.sidebar.radio(
    "Sentiment", ("Positive", "Negative", "Neutral"), key="2"
)
if not st.sidebar.checkbox("Hide", False, key="3"):
    st.subheader("---Word Cloud for %s sentiment---" % (sentiment_word))
    df = tweets[tweets["airline_sentiment"] == sentiment_word.lower()]
    words = " ".join(df["text"])
    processed_words = " ".join(
        [
            word
            for word in words.split()
            if "http" not in word and not word.startswith("@") and word != "RT"
        ]
    )  # processed the word string to remove hyperlinks and retweets from the wordcloud
    wordcloud = WordCloud(
        stopwords=STOPWORDS, background_color="white", height=650, width=800
    ).generate(processed_words)

    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

