import os
import tweepy
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

consumer_key = os.environ["API_KEY"]
consumer_secret = os.environ["API_KEY_SECRET"]
access_token = os.environ["ACCESS_TOKEN"]
access_token_secret = os.environ["ACCESS_TOKEN_SECRET"]

auth = tweepy.OAuth1UserHandler(
  consumer_key, 
  consumer_secret, 
  access_token, 
  access_token_secret
)

api = tweepy.API(auth)

#fetch tweets and save as a df
tweets = api.search_tweets("ukraine", tweet_mode="extended", count=100)

for tweet in tweets:
    # tweets = api.user_timeline(screen_name="#MUFC", count=200, include_rts=False, tweet_mode="extended")
    df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=["Tweets"])
    print(df.head())
    
