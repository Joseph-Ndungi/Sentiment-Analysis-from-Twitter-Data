import os
import tweepy as tw
from dotenv import load_dotenv
import pandas as pd
#import tweepy


#import time
import requests
load_dotenv()

# consumer_key = os.environ["API_KEY"]
# consumer_secret = os.environ["API_KEY_SECRET"]
# access_token = os.environ["ACCESS_TOKEN"]
# access_token_secret = os.environ["ACCESS_TOKEN_SECRET"]

# auth = tweepy.OAuth1UserHandler(
#   consumer_key, 
#   consumer_secret, 
#   access_token, 
#   access_token_secret
# )

# api = tweepy.API(auth)

# #fetch tweets and save as a df
# tweets = api.search_tweets("MUFC", tweet_mode="extended", count=100)

# for tweet in tweets:
#     # tweets = api.user_timeline(screen_name="#MUFC", count=200, include_rts=False, tweet_mode="extended")
#     df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=["Tweets"])
#     print(df.head())
#def crypto_sentiments():
    
client = tw.Client(bearer_token = os.environ["Bearer_Token"], 
                consumer_key = os.environ["API_KEY"], 
                consumer_secret = os.environ["API_KEY_SECRET"], 
                access_token = os.environ["ACCESS_TOKEN"], 
                access_token_secret =  os.environ["ACCESS_TOKEN_SECRET"], 
                return_type = requests.Response,
                wait_on_rate_limit=True)

# Define query
query = '(crypto OR cryptocurrency OR cryptocurrencies) -is:retweet lang:en'

# Get 100 tweets
tweets = client.search_recent_tweets(query = query, 
                                    #tweet_fields = ['author_id', 'created_at'],
                                    max_results = 100)
# Save data as dictionary
tweets_dict = tweets.json() 

# Extract "data" value from dictionary
tweets_data = tweets_dict['data'] 

# Transform to pandas Dataframe
df = pd.json_normalize(tweets_data) 
print(df.head())
print(df.columns)

#return df.head()

