import nltk
# nltk.download('twitter_samples')
from nltk.corpus import twitter_samples

def load_twitter_data():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')  

    return all_positive_tweets, all_negative_tweets 
