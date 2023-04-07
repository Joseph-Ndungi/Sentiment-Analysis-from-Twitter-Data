from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import json
import pandas as pd
from forms import *
import csv
import json
import itertools
import pandas as pd
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



client = tweepy.Client(bearer_token=os.environ["Bearer_Token"])



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
                                    #"promoted_metrics","organic_metrics","non_public_metrics"
                                    "referenced_tweets","reply_settings","source","withheld"], max_results=100).flatten(limit=1000)
        
        tweets_list = list(itertools.chain(tweets))

        tweets_dict = [dict((k, v) for k, v in d.items() if k != 'id' and k != 'edit_history_tweet_ids') for d in tweets_list]

        df = pd.json_normalize(tweets_dict)

        df= df['text']
        df = df.str.replace('[^a-zA-Z0-9\s]', '')
        stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an','http'
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

        STOPWORDS = set(stopwordlist)
        df = df.apply(lambda x: " ".join(x for x in x.split() if x not in STOPWORDS))
        dfData=df
        df = df.to_json(orient='records')
        df = json.loads(df)


        hf_token = "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"
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"

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
        fig= px.pie(df, values=sentiment_counts, names=sentiment_counts.index, title='Sentiment Analysis')
        graph1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


        
        # sentiment_counts.plot(kind='pie', autopct='%1.0f%%')
        # plt.show()

        positive_tweets = df['text'][df["label"] == 'positive']
        #positive_tweets.to_csv('positivetweets.csv', index = False)

        stop_words = ["https", "co", "RT"] + list(STOPWORDS)

        wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = stop_words, max_words = 200).generate(str(positive_tweets))
        #use px to create a plotly figure of the wordcloud
        fig = px.imshow(wordcloud)
        fig.update_layout(
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
        fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        #negative_tweets wordcloud
        negative_tweets = df['text'][df["label"] == 'negative']
        wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = stop_words, max_words = 200).generate(str(negative_tweets))
        #use px to create a plotly figure of the wordcloud
        fig = px.imshow(wordcloud)
        fig.update_layout(
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
        fig1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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

        

        return render_template('query.html', form=form, graph1=graph1, wordcloud=fig,
                               wordcloud1=fig1, wordcloud2=fig2, title='Sentiment Analysis')


    
    return render_template('query.html', form=form,title='Sentiment Analysis')

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
        #print('Topic {}: {}'.format(i, ' '.join(topic_words[i])))
        #message += 'Topic {}: {}'.format(i, ' '.join(topic_words[i]))
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

if __name__ == '__main__':
    app.run(debug=False)