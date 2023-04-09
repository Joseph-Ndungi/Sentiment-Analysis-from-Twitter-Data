from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import json
import pandas as pd
from forms import *
import csv
import json
import itertools
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
import plotly
import plotly.express as px
import base64
from io import BytesIO
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle


def tag_followers(followers):
    if followers >= 500:
        return True
    else:
        return False

def count_characters(text):
    if isinstance(text, str):
        return len(text)
    else:
        return 0
    
def tag_characters(characters):
    if characters >= 100:
        return True
    else:
        return False  
client = tweepy.Client(bearer_token=os.environ["Bearer_Token"])
ALLOWED_EXTENSIONS = set(['csv'])


app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

@app.route('/' , methods=['GET', 'POST'])
def index():
    # form = DateForm()
    # if form.validate_on_submit():
    #     print(form.startDate.data)
    #     print(form.endDate.data)
    #     print(form.query.data)
    #     return redirect(url_for('index'))
    return render_template('dashboard.html', title='Dashboard')



@app.route('/sentiments', methods=['GET', 'POST'])
def sentiments():
    
    form = DateForm()
    # graph1 = ''
    # wordcloud = []
    if request.method == 'POST':
    #if form.validate_on_submit():
        query = form.query.data + ' -is:retweet lang:en'
        tweets = tweepy.Paginator(client.search_recent_tweets, query=query,
                              tweet_fields = ["author_id","text","geo","public_metrics","possibly_sensitive",
                                    "referenced_tweets","reply_settings","source","withheld"], max_results=100).flatten(limit=1000)
        
        tweets_list = list(itertools.chain(tweets))

        flash(f'Number of tweets: {len(tweets_list)}', 'success')

        tweets_dict = [dict((k, v) for k, v in d.items() if k != 'id' and k != 'edit_history_tweet_ids') for d in tweets_list]

        df = pd.json_normalize(tweets_dict)

        df= df['text']
        df = df.str.replace('[^a-zA-Z0-9\s]', '')

        #STOPWORDS = set(stopwordlist)
        STOPWORDS= set(stopwords.words('english'))

        df = df.apply(lambda x: " ".join(x for x in x.split() if x not in STOPWORDS))
        dfData=df
        df = df.to_json(orient='records')
        df = json.loads(df)


        hf_token = "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"
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"

        #flash(f'Using model: {model}', 'success')

        API_URL = "https://api-inference.huggingface.co/models/" + model
        headers = {"Authorization": "Bearer %s" % (hf_token)}


        def analysis(payload):
            data = json.dumps(payload)
            response = requests.request("POST", API_URL, headers=headers, data=data)
            return json.loads(response.content.decode("utf-8"))


        payload = {"inputs": df}
        response = analysis(payload)

        jsonFile= response
        df = pd.DataFrame(columns=['label', 'score'])
        #df.head()
        for lst in jsonFile:  
            maxScore = max(lst, key=lambda d: d['score'])
            # df = df.append({'label': maxScore['label'], 'score': maxScore['score']}, ignore_index=True)
            df = pd.concat([df, pd.DataFrame({'label': maxScore['label'], 'score': maxScore['score']}, index=[0])], ignore_index=True)

        
        df = pd.concat([df, dfData], axis=1)
        #df to csv
        df.to_csv('sentiment.csv', index = False)

        sentiment_counts = df.groupby(['label']).size()
        print(sentiment_counts)

        #visualize in a pie chart
        fig= px.pie(df, values=sentiment_counts, names=sentiment_counts.index, title= f'Sentiment Analysis Using model: {model}')
        graph1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


        
        # sentiment_counts.plot(kind='pie', autopct='%1.0f%%')
        # plt.show()
        #flash('Generating Wordclouds...', 'info')

        positive_tweets = df['text'][df["label"] == 'positive']
        #positive_tweets.to_csv('positivetweets.csv', index = False)

        stop_words = ["https", "co", "RT"] + list(STOPWORDS)

        pwordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = stop_words, max_words = 200).generate(str(positive_tweets))
        #use px to create a plotly figure of the wordcloud
        pfig = px.imshow(pwordcloud)
        pfig.update_layout(
            title="Wordcloud of Positive Tweets",
            xaxis_title="Word",
            yaxis_title="Frequency",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        #fig.show()
        #dump the plotly figure into a json string
        pfig = json.dumps(pfig, cls=plotly.utils.PlotlyJSONEncoder)

        #negative_tweets wordcloud
        negative_tweets = df['text'][df["label"] == 'negative']
        nWordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = stop_words, max_words = 200).generate(str(negative_tweets))
        #use px to create a plotly figure of the wordcloud
        fig1 = px.imshow(nWordcloud)
        fig1.update_layout(
            title="Wordcloud of Negative Tweets",
            xaxis_title="Word",
            yaxis_title="Frequency",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        #fig.show()
        #dump the plotly figure into a json string
        fig1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        #neutral_tweets wordcloud
        neutral_tweets = df['text'][df["label"] == 'neutral']
        wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = stop_words, max_words = 200).generate(str(neutral_tweets))
        #use px to create a plotly figure of the wordcloud
        fig = px.imshow(wordcloud)
        fig.update_layout(
            title="Wordcloud of Neutral Tweets",
            xaxis_title="Word",
            yaxis_title="Frequency",
            font=dict(
                
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )

        #dump the plotly figure into a json string
        fig2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        

        return render_template('query.html', form=form, graph1=graph1, pfig=pfig,
                               fig1=fig1, fig2=fig2, title='Sentiment Analysis')
    else:
        return render_template('query.html', form=form, title='Sentiment Analysis') 

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    #open the csv file
    df = pd.read_csv('sentiment.csv')

    #get total number of positive, negative and neutral tweets each
    positive_tweets = df['text'][df["label"] == 'positive']
    negative_tweets = df['text'][df["label"] == 'negative']
    neutral_tweets = df['text'][df["label"] == 'neutral']

    #get the totals of each
    positive_tweets_count = positive_tweets.count()
    negative_tweets_count = negative_tweets.count()
    neutral_tweets_count = neutral_tweets.count()

    # #visualize the sentiment distribution using a histogram
    # fig = px.histogram(df, x="label", title="Sentiment Distribution")
    # fig.update_layout(
    #     title="Sentiment Distribution",
    #     xaxis_title="Sentiment",
    #     yaxis_title="Frequency",
    #     font=dict(
    #         family="Courier New, monospace",
    #         size=18,
    #         color="#7f7f7f"
    #     )
    # )
    # #dump the plotly figure into a json string
    # fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #visualize the sentiment distribution using boxplot
    fig2 = px.box(df, x="label", y="score", title="Sentiment Distribution")
    fig2.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Score",
        font=dict(
            
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    #dump the plotly figure into a json string
    fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # calculate the sentiment polarity of each tweet
    df['sentiment_polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)


    #visualize the sentiment polarity distribution using a histogram
    fig3 = px.histogram(df, x="sentiment_polarity", title="Sentiment Polarity Distribution")

    fig3.update_layout(
        title="Sentiment Polarity Distribution",
        xaxis_title="Sentiment Polarity",
        yaxis_title="Frequency",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )

    #dump the plotly figure into a json string
    fig3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)


    '''
    We then create a document-term matrix using the CountVectorizer from scikit-learn,
      which converts each tweet into a vector of word frequencies. We fit an LDA model to this matrix,
        with 10 topics. Finally, we print the top 10 most common words for each topic, and visualize the
          distribution of topics in the tweets using a histogram.
    '''

    #define stop words
    stop_words = set(stopwords.words('english'))

    # tokenize the tweet text and remove stopwords
    tokenized_tweets = []
    for tweet in df['text']:
        tokens = word_tokenize(tweet.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words]
        tokenized_tweets.append(' '.join(filtered_tokens))

    #create a document-term matrix  
    vectorizer = CountVectorizer(max_features=1000)
    doc_term_matrix = vectorizer.fit_transform(tokenized_tweets)
    # fit an LDA model to the document-term matrix
    lda_model = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online', random_state=0)
    lda_model.fit(doc_term_matrix)

    topic_words = {}
    message = ""

    for i, topic in enumerate(lda_model.components_):
        word_idx = np.argsort(topic)[::-1][:10]
        topic_words[i] = [vectorizer.get_feature_names_out()[i] for i in word_idx]
        line = 'Topic {}: {}\n'.format(i, ' '.join(topic_words[i]))
        message += line
 


    #plot the distribution of topics in the tweets
    fig4 = px.histogram(lda_model.transform(doc_term_matrix), title="Topic Distribution")
    fig4.update_layout(
        title="Topic Distribution",
        xaxis_title="Topic",
        yaxis_title="Frequency",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )

    #dump the plotly figure into a json string
    fig4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('analysis.html',
                            fig2=fig2,fig3=fig3,fig4=fig4,
                             title='Analysis',
                             tPos=positive_tweets_count,
                             tNeg=negative_tweets_count,
                             tNeu=neutral_tweets_count,
                             message=message)



@app.route('/timeSeries', methods=['GET', 'POST'])
def timeSeries():
    #network analysis
    #open the csv file
    df = pd.read_csv('Corona_NLP_test.csv')
    df['created_at'] = pd.to_datetime(df['TweetAt'], format='%d-%m-%Y')
    # set the 'created_at' column as the index
    df.set_index('created_at', inplace=True)

    # resample the data to a daily frequency
    daily_counts = df.resample('D').size()

    # plot the time series of daily tweet counts and dump the plotly figure into a json string
    fig = px.line(daily_counts, title="Daily Tweet Counts")
    fig.update_layout(
        title="Daily Tweet Counts",
        xaxis_title="Date",
        yaxis_title="Frequency",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

  # decompose the time series into trend, seasonal, and residual components
    decomposition = seasonal_decompose(daily_counts, model='additive', period=7)
    # plot the trend, seasonal, and residual components
    fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig2.add_trace(go.Scatter(x=daily_counts.index, y=decomposition.trend, name='Trend'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=daily_counts.index, y=decomposition.seasonal, name='Seasonality'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=daily_counts.index, y=decomposition.resid, name='Residuals'), row=3, col=1)
    fig2.add_trace(go.Scatter(x=daily_counts.index, y=daily_counts, name='Original'), row=4, col=1)
    fig2.update_layout(
        title="Time Series Decomposition",
        xaxis_title="Date",
        yaxis_title="Frequency",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # # create a list of the top 10 most frequently used hashtags
    # hashtags = []
    # for tweet in df['OriginalTweet']:
    #     hashtags.extend(re.findall(r"#(\w+)", tweet))
    # top_10_hashtags = Counter(hashtags).most_common(10)
    # top_10_hashtags = dict(top_10_hashtags)

    # # plot the top 10 most frequently used hashtags
    # fig3 = px.bar(top_10_hashtags, x=list(top_10_hashtags.keys()), y=list(top_10_hashtags.values()), title="Top 10 Hashtags")
    # fig3.update_layout(
    #     title="Top 10 Hashtags",
    #     xaxis_title="Hashtag",
    #     yaxis_title="Frequency",
    #     font=dict(
    #         family="Courier New, monospace",
    #         size=18,
    #         color="#7f7f7f"
    #     )
    # )
    # fig3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('network.html',
                            fig=fig,fig2=fig2,
                                title='Network Analysis')

@app.route('/digitalNudging', methods=['GET', 'POST'])
def digitalNudging(): 
        if request.method == 'POST':
            # get the file from the post request
            file = request.files['file']
            df = pd.read_csv(file)
    
            df['Followers'] = df['Followers'].apply(tag_followers)
  
            df['Character'] = df['Text'].apply(count_characters)

            df["Character"] = df["Character"].apply(tag_characters)
            df.fillna(1, inplace=True)  # Convert NA values to 1
            
            features=['Verified', 'Protected', 'Followers', 'VerifiedRetweet', 'Character']

            df=df[features]

            # load the model from disk
            with open('finalized_model.sav', 'rb') as f:
                model = pickle.load(f)

            # make a prediction
            prediction = model.predict(df)

            # add the prediction as a new column
            df['Prediction'] = prediction

            # convert the dataframe to a csv file
            df.to_csv('prediction.csv', index=False)

            # download the csv file
            return send_file('prediction.csv',
                            mimetype='text/csv',
                            as_attachment=True)
        else:
            return render_template('nudge.html', title='Digital Nudging')

if __name__ == '__main__':
    app.run(debug=True)