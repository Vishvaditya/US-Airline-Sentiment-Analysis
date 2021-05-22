import pandas as pd
import numpy as np
import streamlit as st

st.title("US Airline Sentiment Analysis")
st.sidebar.title("Sentiment Analysis of Tweets using Streamlit")

st.markdown(
    "This Dashboard visualises sentiment anlaysis of tweets regarding US Airlines"
)

tweets = pd.read_csv("Tweets.csv")  # loaded Tweets.csv file in pandas dataframe

