import csv
import json
import pandas as pd
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS
#read csv
df = pd.read_csv('tweets.csv')
df.describe()
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
#df.to_csv('cleaned.csv', index = False)


#make it json serializable
df = df.to_json(orient='records')
df = json.loads(df)

'''
TODO:
 Text Preprocessing
 Use roberta model to predict the sentiment of the tweets
'''

#model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
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
#print(response)

jsonFile= response
df = pd.DataFrame(columns=['label', 'score'])
df.head()
for lst in jsonFile:  
    maxScore = max(lst, key=lambda d: d['score'])
    df = df.append({'label': maxScore['label'], 'score': maxScore['score']}, ignore_index=True)

# print(df.head(10))
#append df to df with tweets
df = pd.concat([df, pd.read_csv('tweets.csv')], axis=1)
#df = df.drop(columns=['Unnamed: 0'])
#print(df.head(10))

#save csv
#df.to_csv('sentiment.csv', index = False)

# sentiment_counts = df.groupby(['label']).size()
# print(sentiment_counts)

# #plot
# sentiment_counts.plot(kind='pie', autopct='%1.0f%%')
# plt.show()

#print(df)

#wordcloud
positive_tweets = df['text'][df["label"] == 'positive']
positive_tweets.to_csv('positivetweets.csv', index = False)

stop_words = ["https", "co", "RT"] + list(STOPWORDS)

wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = stop_words, max_words = 200).generate(str(positive_tweets))
# Display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()