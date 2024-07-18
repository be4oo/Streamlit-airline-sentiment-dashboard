import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


DATA_URL = ("Tweets.csv")

st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")
st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")

@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
# This line of code performs several operations in sequence to display a random tweet from a DataFrame in a Streamlit sidebar using Markdown formatting.
# data.query("airline_sentiment == @random_tweet"): Filters the 'data' DataFrame for rows where the 'airline_sentiment' column matches the value of 'random_tweet'.
# [["text"]]: Selects the 'text' column from the filtered DataFrame, resulting in a DataFrame with just the tweet texts.
# .sample(n=1): Randomly selects 1 row from the DataFrame containing only the tweet texts. This is used to get a random tweet.
# .iat[0, 0]: Accesses the first element of the sampled DataFrame, which is the text of the randomly selected tweet.
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour to look at", 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", True, key='1a'):
    st.markdown("### Tweet locations based on time of day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)


st.sidebar.subheader("Total number of tweets for each airline")
each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Close", True, key='2a'):
    if each_airline == 'Bar plot':
        st.subheader("Total number of tweets for each airline")
        fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_1)
    if each_airline == 'Pie chart':
        st.subheader("Total number of tweets for each airline")
        fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
        st.plotly_chart(fig_2)

st.sidebar.subheader("Breakdown airline tweets by sentiment")
choice = st.sidebar.multiselect('Pick airlines', ('United', 'Southwest', 'American', 'US Airways', 'Delta', 'Virgin America'), key='0')
if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    # This line of code creates a histogram using Plotly Express, assigning it to the variable 'fig_choice'.
    # px.histogram: Calls the histogram function from Plotly Express (px) to create a histogram.
    # choice_data: The DataFrame containing the data to be plotted.
    # histfunc='count': Specifies the histogram function to count occurrences, effectively counting the number of tweets per airline per sentiment.
    # facet_col='airline_sentiment': Creates separate subplots (facets) for each unique value in the 'airline_sentiment' column, allowing for comparison across sentiments.
    # labels={'airline_sentiment':'tweets'}: Renames the 'airline_sentiment' axis label to 'tweets' for clarity in the plot.
    fig_choice = px.histogram(choice_data, x='airline', y= 'airline_sentiment', histfunc= 'count', color='airline_sentiment',
                              facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'}, height=600, width=800)
    st.plotly_chart(fig_choice)


st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))

if not st.sidebar.checkbox("Close", True, key='3a'):
    st.subheader(f'Word cloud for {word_sentiment} sentiment')
    df = data[data['airline_sentiment'] == word_sentiment]
    # This line of code concatenates all the strings found in the 'text' column of
    # the DataFrame 'df' into a single string, with each word separated by a space.
    # ' '.join(...): Joins elements of the list with a space character as the separator, resulting in a single string.
    words = ' '.join(df['text'])
    # This line of code filters and joins words from a string into a new string, excluding specific patterns.
    # [word for word in words.split() ...]: List comprehension that iterates over each word in the 'words' string, which is split into a list of words based on spaces.
    # if 'http' not in word ...: Filters out any word containing 'http', typically used to remove URLs.
    # and not word.startswith('@') ...: Further filters out any word that starts with '@', commonly used to remove mentions in social media texts.
    # and word != 'RT': Additionally, filters out the exact word 'RT', which is often used in social media to indicate a retweet.
    # The overall effect is to create a string of 'processed_words' that excludes URLs, mentions, and retweet indicators from the original 'words' string.
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    # This line of code generates a word cloud image from a string of 'processed_words'.
    # WordCloud(...): Instantiates a WordCloud object from the WordCloud class.
    # stopwords=STOPWORDS: Specifies a collection of words to be ignored when generating the word cloud. 
    # These are typically common words that do not contribute to the meaning of the text.
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    
    st.pyplot(fig)
