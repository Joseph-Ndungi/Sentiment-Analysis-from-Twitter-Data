import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "(twitter) -is:retweet lang:en)"
tweets = []
limit = 5000

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > limit:
        break
    tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

df = pd.DataFrame(tweets, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
df.to_csv('tweets.csv', index=False)