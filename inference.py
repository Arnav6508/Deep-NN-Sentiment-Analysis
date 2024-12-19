import pickle
import numpy as np
import tensorflow as tf
from preprocess import process_tweet
from utils import *

def get_prediction_from_tweet(tweet):

    with open('model_weights/side_weights.pkl', 'rb') as file:
        side_weights = pickle.load(file)
    vocab = side_weights['vocab']
    max_length = side_weights['max_len']

    tweet = process_tweet(tweet)
    tweet = pad_seq(tweet, vocab, max_length)
    tweet = np.array([tweet])

    model = tf.keras.models.load_model('model_weights/model.h5')
    pred = model.predict(tweet)[0][0]
    if pred>0.5: print('Positive Tweet')
    else: print('negative Tweet')

    return pred

def visualise_embeddings():
    with open('model_weights/side_weights.pkl', 'rb') as file:
        side_weights = pickle.load(file)
    vocab = side_weights['vocab']
    embeddings = side_weights['embeddings']
    plot_embeddings(embeddings, vocab)