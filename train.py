import json
#import keras
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize


from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input, Activation, CuDNNGRU, Bidirectional, Dropout, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from keras.layers.recurrent import GRU


num_classes = 5
embedding_dim = 300
epochs = 50
batch_size = 128
max_len = 150

# to deal with class unbalanced classes
class_weights = {0: 7,
                 1: 7,
                 2: 4,
                 3: 3,
                 4: 1}


stopwords = set(stopwords.words('english'))
# Detokenizer combines tokenized elements
detokenizer = TreebankWordDetokenizer()


y = np.loadtxt("targets.txt", delimiter=',')
reviews_frame = pd.read_pickle("pandas_reviews.pkl")

X_train, X_val, y_train, y_val = train_test_split(reviews_frame["cleanedReviews"], y, test_size=0.20)

print("Dataset split performed, Validation set size: ", str(len(X_val)), "Traning set size:", str(len(X_train)))


# Prepare embeddings ---------------------------------------------------------------------------------------------


embeddings_index = {}

# Read pre-trained word vectors and populate to dictionary
f = open("glove.840B.300d.txt", encoding="utf8")

for line in f:
    values = line.split()
    word = ''.join(values[:-embedding_dim])
    coefs = np.asarray(values[-embedding_dim:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# train tokenizer
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(X_train)

# fit tokenizer
sequences_train = tokenizer.texts_to_sequences(X_train)

# Padding any short sequences with 0s
X_train = pad_sequences(sequences_train, maxlen=max_len)

sequences_val = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(sequences_val, maxlen=max_len)

word_index = tokenizer.word_index

# create embedding layer
# We can designate "Out of Vocabulary" word vectors here
# In this case, they are initialized to zero vector
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

print("Creating emmbeding matrix...")

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# TRAINING ------------------------------------------------------------------------------------------------------

# create embedding layer to match the populated embeddings
embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False)

print("embedding finished")

input = Input(shape=(max_len,), dtype='int32')
embedded_sequences = embedding_layer(input)

# Structure model architecture with bidirectional GRU
# Uncomment for GPU usage and comment the other command
#x = Bidirectional(CuDNNGRU(50, return_sequences=True))(embedded_sequences)
x = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)

x = GlobalMaxPooling1D()(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.1)(x)

output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=input, outputs=output)
model.summary()

# adjust training parameters
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', mode='min', patience=6)
callback = [checkpoint, early]

print("Model configuration finished, training started...")

# train the model
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_val, y_val),
          callbacks=callback,
          class_weight=class_weights)


# save the used tokenizer to use on future evaluations of the model

print("saving tokenizer to test")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("saving finished")


