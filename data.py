# OAuth2.0 Version 
import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

#Put your Bearer Token in the parenthesis below
client = tweepy.Client(bearer_token=os.environ["Bearer_Token"])


query = '(twitter) -is:retweet lang:en'
tweets = tweepy.Paginator(client.search_recent_tweets, query=query,
                              tweet_fields = ["author_id","text","geo","public_metrics","possibly_sensitive",
                                    #"promoted_metrics","organic_metrics","non_public_metrics"
                                    "referenced_tweets","reply_settings","source","withheld"], max_results=100).flatten(limit=1000)

import itertools

# Flatten the tweets into a list
tweets_list = list(itertools.chain(tweets))
print(tweets_list)
