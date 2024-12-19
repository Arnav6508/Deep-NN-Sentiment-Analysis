import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def train_test_split(pos_tweets, neg_tweets):
    # Split positive set into validation and training
    val_pos = pos_tweets[4000:]
    train_pos = pos_tweets[:4000]

    # Split negative set into validation and training
    val_neg = neg_tweets[4000:]
    train_neg = neg_tweets[:4000]

    train_x = train_pos + train_neg 
    val_x  = val_pos + val_neg

    train_y = [1 for i in train_pos] + [0 for i in train_neg]
    val_y = [1 for i in val_pos] + [0 for i in val_neg]

    return train_x, val_x, train_y, val_y

def create_vocab(corpus):
    vocab = {'': 0, '[UNK]': 1}
    idx = 2

    for tweet in corpus:
        for word in tweet:
            if vocab.get(word,0) == 0: 
                vocab[word] = idx
                idx += 1
    return vocab

def max_len(train_x, val_x):
    '''
        Return the max length of tweet out of all tweets
    '''
    max_train_len =  max([len(tweet) for tweet in train_x])
    max_val_len = max([len(tweet) for tweet in val_x])

    return max(max_train_len, max_val_len)

def pad_seq(tweet, vocab, max_len, unk_token = '[UNK]'):
    unk_id = vocab[unk_token]
    indexed_seq = [vocab.get(word,unk_id) for word in tweet]
    padded_seq = indexed_seq + [0]*(max_len-len(tweet))
    return padded_seq

def create_model(vocab_size, embedding_dim, max_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    return model

def plot_metrics(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric.title())
    plt.legend([metric, f'val_{metric}'])
    plt.show()

def plot_embeddings(embeddings, vocab):
    pca = PCA(n_components = 2)
    embeddings_2D = pca.fit_transform(embeddings)

    pos_words = ['best', 'good', 'nice', 'love', 'better', ':)']
    neg_words = ['bad', 'hurt', 'sad', 'hate', 'worst', ':(']

    pos_idx = [vocab[word] for word in pos_words]
    neg_idx = [vocab[word] for word in neg_words]

    plt.scatter(embeddings_2D[pos_idx][:,0], embeddings_2D[pos_idx][:,1], color = 'g')
    for i, text in enumerate(pos_words):
        plt.annotate(text, (embeddings_2D[pos_idx][i,0], embeddings_2D[pos_idx][i,1]))

    plt.scatter(embeddings_2D[neg_idx][:,0], embeddings_2D[neg_idx][:,1], color = 'r')
    for i, text in enumerate(neg_words):
        plt.annotate(text, (embeddings_2D[neg_idx][i,0], embeddings_2D[neg_idx][i,1]))
    
    plt.title('Word embeddings in 2D')
    plt.show()