from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import logging
import logging.config

logging.config.fileConfig("log.conf")

# create logger
logger = logging.getLogger("main")

max_len = 0
word_freqs = collections.Counter()
num_items = 0

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40 #max number of words in a sentence

logger.info(f'Opening file ' + str(__file__))
data_file = open(os.path.join(os.path.dirname(__file__), 'training.txt'), 'rb')

for line in data_file:
    label, sentence = line.strip().split(b'\t')
    sentence_line = sentence.decode('ascii','ignore').lower()
    words = nltk.word_tokenize(sentence_line)
    if (len(words) > max_len):
        max_len = len(words)
    for word in words:
        word_freqs[word] += 1
    num_items += 1

data_file.close()

logger.info("maxlen: " + str(max_len))

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0] : i+2 for i,x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index['PAD'] = 0 # padding
word2index['UNK'] = 1 # unknown token

index2word = {v:k for v,k in word2index.items()}

# Populating dataframes...
X = np.empty((num_items, ), dtype=list)
y = np.zeros((num_items, ))
i = 0
ftrain = open(os.path.join(os.path.dirname(__file__), 'training.txt'), 'rb')
for line in ftrain:
    label, sentence = line.strip().split(b'\t')
    words = nltk.word_tokenize(sentence.decode("ascii", "ignore").lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X[i] = seqs
    y[i] = int(label)
    i += 1
ftrain.close()

X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# KERAS MODEL
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

logger.info("Training started")
history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))

#Plot
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()
