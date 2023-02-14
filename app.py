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

# Define query
query = '(HELB) -is:retweet lang:en'
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
# print(df.head())
df= df['text']
#remove special characters
df = df.str.replace('[^a-zA-Z0-9\s]', '')

#remove stopwords
'''
TODO:
 Text Preprocessing
 Use roberta model to predict the sentiment of the tweets
'''

# from nltk.corpus import stopwords
# stop = stopwords.words('english')
# df = df.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
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

#save csv
df.to_csv('cleaned.csv', index = False)




#make it json serializable
df = df.to_json(orient='records')
df = json.loads(df)
# print(df)



'''
TODO:
 Text Preprocessing
 Use roberta model to predict the sentiment of the tweets
'''

model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
hf_token = "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"


API_URL = "https://api-inference.huggingface.co/models/" + model
headers = {"Authorization": "Bearer %s" % (hf_token)}

def analysis(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


payload = {"inputs": df}
response = analysis(payload)


print(response)



  

