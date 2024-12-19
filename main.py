import pickle
import numpy as np
import tensorflow as tf
from load import load_twitter_data
from preprocess import process_tweet
from utils import *

def main():
    all_positive_tweets, all_negative_tweets = load_twitter_data()
    
    # Preprocess tweets
    all_positive_tweets_processed = [process_tweet(tweet) for tweet in all_positive_tweets]
    all_negative_tweets_processed = [process_tweet(tweet) for tweet in all_negative_tweets]

    train_x, val_x, train_y, val_y = train_test_split(all_positive_tweets_processed, all_negative_tweets_processed)

    ###################################################

    vocab = create_vocab(train_x)
    max_length = max_len(train_x, val_x)
    side_weights = {'vocab': vocab, 'max_len': max_length}

    with open('model_weights/side_weights.pkl', 'wb') as file:
        pickle.dump(side_weights, file)

    print('Vocab and max_len saved')

    ###################################################

    train_x_padded = [pad_seq(x, vocab, max_length) for x in train_x]
    val_x_padded = [pad_seq(x, vocab, max_length) for x in val_x]

    train_x_prepared = np.array(train_x_padded)
    val_x_prepared = np.array(val_x_padded)

    train_y_prepared = np.array(train_y)
    val_y_prepared = np.array(val_y)

    model = create_model(len(vocab), 16, max_length)
    history = model.fit(train_x_prepared, train_y_prepared, epochs=20, validation_data=(val_x_prepared, val_y_prepared))

    model.save("model_weights/model.h5")
    print(f"Model saved")

    ###################################################

    plot_metrics(history, 'accuracy')
    plot_metrics(history, 'loss')

    print('Accuracy of model is:', max(history.history['accuracy']))


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