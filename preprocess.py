import re
import string
import nltk

# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet 
from nltk.stem import WordNetLemmatizer

stopwords_english = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def pos_tag_convert(nltk_tag):
    '''
        Converts NLTK tag to Wordnet tag
    '''
    if nltk_tag.startswith('J'):   return wordnet.ADJ
    elif nltk_tag.startswith('V'): return wordnet.VERB
    elif nltk_tag.startswith('N'): return wordnet.NOUN
    elif nltk_tag.startswith('R'): return wordnet.ADV
    else:                          return wordnet.NOUN

def process_tweet(tweet):
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    word_tag_pairs = nltk.pos_tag(tokenizer.tokenize(tweet))

    cleaned_tweet = []
    for word_tag_pair in word_tag_pairs:

        word = word_tag_pair[0]
        tag = word_tag_pair[1]

        if word not in string.punctuation and word not in stopwords_english:
            stem_word = lemmatizer.lemmatize(word, pos_tag_convert(tag))
            cleaned_tweet.append(stem_word)
    
    return cleaned_tweet