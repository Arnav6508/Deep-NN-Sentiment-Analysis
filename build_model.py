import pickle
import numpy as np
from load import load_twitter_data
from preprocess import process_tweet
from utils import *

def build_model():
    all_positive_tweets, all_negative_tweets = load_twitter_data()
    
    ##################### PREPROCESS #####################

    all_positive_tweets_processed = [process_tweet(tweet) for tweet in all_positive_tweets]
    all_negative_tweets_processed = [process_tweet(tweet) for tweet in all_negative_tweets]

    train_x, val_x, train_y, val_y = train_test_split(all_positive_tweets_processed, all_negative_tweets_processed)

    ###################################################

    vocab = create_vocab(train_x)
    max_length = max_len(train_x, val_x)

    ##################### MODEL #####################

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

    ###################### SIDE WEIGHTS ######################

    embeddings_layer = model.layers[0]
    embeddings = embeddings_layer.get_weights()[0]

    side_weights = {'vocab': vocab, 'max_len': max_length, 'embeddings': embeddings}

    with open('model_weights/side_weights.pkl', 'wb') as file:
        pickle.dump(side_weights, file)

    print('Vocab, max_len and word embeddings saved')

    ##################### METRICS #####################

    plot_metrics(history, 'accuracy')
    plot_metrics(history, 'loss')

    print('Accuracy of model is:', max(history.history['accuracy']))
