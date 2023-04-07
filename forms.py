from wtforms import DateField,SelectField, StringField
from flask_wtf import FlaskForm
from datetime import datetime
from wtforms.validators import InputRequired



class DateForm(FlaskForm):
    startDate = DateField(default=datetime.strptime('2023-01-01','%Y-%m-%d'),validators=[InputRequired()])
    endDate = DateField(default=datetime.strptime('2023-12-31','%Y-%m-%d'),validators=[InputRequired()])
    query = StringField('Text', validators=[InputRequired()])




# import pandas as pd
# from textblob import TextBlob
# import matplotlib.pyplot as plt

# # load the tweets data into a pandas dataframe
# df = pd.read_csv('sentiment.csv')

# # calculate the sentiment polarity of each tweet
# df['sentiment_polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# # create a histogram of the sentiment polarities
# plt.hist(df['sentiment_polarity'], bins=20)
# plt.xlabel('Sentiment Polarity')
# plt.ylabel('Count')
# plt.title('Distribution of Sentiment Polarities')
# plt.show()

# #This would give you a visualization of how positive or negative the tweets are towards the brand, and could help you identify areas for improvement in your brand's online reputation.


# # #identify key topics using LDA
# # from sklearn.feature_extraction.text import CountVectorizer
# # from sklearn.decomposition import LatentDirichletAllocation

# # # create a count vectorizer object
# # count_vectorizer = CountVectorizer(stop_words='english')

# # # fit and transform the processed tweets
# # count_data = count_vectorizer.fit_transform(df['text'])

# # # create an LDA model instance
# # lda = LatentDirichletAllocation(n_components=10, random_state=0)

# # # fit the LDA model with the count data
# # lda.fit(count_data)

# # # display the 10 topics
# # print(lda.components_)
# # print(count_vectorizer.get_feature_names())

# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# import matplotlib.pyplot as plt

# # load the tweets data into a pandas dataframe
# df = pd.read_csv('sentiment.csv')

# # define stopwords
# stop_words = set(stopwords.words('english'))

# # tokenize the tweet text and remove stopwords
# tokenized_tweets = []
# for tweet in df['text']:
#     tokens = word_tokenize(tweet.lower())
#     filtered_tokens = [token for token in tokens if token not in stop_words]
#     tokenized_tweets.append(' '.join(filtered_tokens))

# # create a document-term matrix
# vectorizer = CountVectorizer(max_features=1000)
# doc_term_matrix = vectorizer.fit_transform(tokenized_tweets)

# # fit an LDA model to the document-term matrix
# lda_model = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online', random_state=0)
# lda_model.fit(doc_term_matrix)

# # print the most common words for each topic
# topic_words = {}
# for i, topic in enumerate(lda_model.components_):
#     word_idx = np.argsort(topic)[::-1][:10]
#     topic_words[i] = [vectorizer.get_feature_names_out()[i] for i in word_idx]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words[i])))

# # plot the distribution of topics in the tweets
# topic_distribution = lda_model.transform(doc_term_matrix)
# plt.hist(topic_distribution.argmax(axis=1), bins=10)
# plt.xlabel('Topic')
# plt.ylabel('Count')
# plt.title('Distribution of Topics in Tweets')
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose

# # load the tweets data into a pandas dataframe
# df = pd.read_csv('Corona_NLP_test.csv')

# # convert the 'created_at' column to a datetime object and specify the format of the date
# df['created_at'] = pd.to_datetime(df['TweetAt'], format='%d-%m-%Y')


# # set the 'created_at' column as the index
# df.set_index('created_at', inplace=True)

# # resample the data to a daily frequency
# daily_counts = df.resample('D').size()

# # plot the time series of daily tweet counts
# plt.plot(daily_counts)
# plt.xlabel('Date')
# plt.ylabel('Tweet Count')
# plt.title('Daily Tweet Counts')
# plt.show()

# # #resample the data to hourly frequency
# # hourly_counts = df.resample('H').size()

# # # plot the time series of hourly tweet counts
# # plt.plot(hourly_counts)
# # plt.xlabel('Date')
# # plt.ylabel('Tweet Count')
# # plt.title('Hourly Tweet Counts')
# # plt.show()

# # decompose the time series into trend, seasonal, and residual components
# decomposition = seasonal_decompose(daily_counts, model='additive', period=7)
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid

# # plot the decomposed components
# plt.subplot(411)
# plt.plot(daily_counts, label='Original')
# plt.legend(loc='upper left')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='upper left')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='upper left')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()






