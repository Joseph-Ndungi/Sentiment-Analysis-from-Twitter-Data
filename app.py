import os
import tweepy as tw
from dotenv import load_dotenv
import pandas as pd
import json


#import time
import requests
load_dotenv()
    
client = tw.Client(bearer_token = os.environ["Bearer_Token"], 
                consumer_key = os.environ["API_KEY"], 
                consumer_secret = os.environ["API_KEY_SECRET"], 
                access_token = os.environ["ACCESS_TOKEN"], 
                access_token_secret =  os.environ["ACCESS_TOKEN_SECRET"], 
                return_type = requests.Response,
                wait_on_rate_limit=True)

# Define query and avoid retweets and qouted tweets
query = '(amerix) -is:retweet -is:quote lang:en'
'''
[attachments,author_id,context_annotations,conversation_id,created_at,edit_controls,edit_history_tweet_ids,
entities,geo,id,in_reply_to_user_id,lang,non_public_metrics,organic_metrics,possibly_sensitive,promoted_metrics
,public_metrics,referenced_tweets,reply_settings,source,text,withheld]'''
# Get 100 tweets
tweets = client.search_recent_tweets(query = query, 
                                    tweet_fields = ["author_id","text","geo","public_metrics","possibly_sensitive",
                                    #"promoted_metrics","organic_metrics","non_public_metrics"
                                    "referenced_tweets","reply_settings","source","withheld"],max_results = 100)
# Save data as dictionary
tweets_dict = tweets.json() 
#print(tweets_dict)
# Extract "data" value from dictionary
tweets_data = tweets_dict['data'] 

# Transform to pandas Dataframe
df = pd.json_normalize(tweets_data) 

# Save to csv
df.to_csv('tweets.csv', index = False)



