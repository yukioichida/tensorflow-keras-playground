import logging
import logging.config

logging.config.fileConfig("log.conf")
# create logger
logger = logging.getLogger("main")

logger.info("importing keras")
from keras.models import load_model
from keras.preprocessing import sequence
import collections
import nltk
import numpy as np
import os

'''
Building the vocabulary
'''
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

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0] : i+2 for i,x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index['PAD'] = 0 # padding
word2index['UNK'] = 1 # unknown token

index2word = {v:k for k,v in word2index.items()}

'''
Predicting
'''
logger.info("Predicting")
model = load_model('lstm_trained.h5')

# outside examples
my_sentence = "this movie was good but i don't like this genre"

my_sentence_words = nltk.word_tokenize(my_sentence)
seqs = []
for word in my_sentence_words:
    if word in word2index:
        seqs.append(word2index[word])
    else:
        seqs.append(word2index["UNK"])


new_x = np.empty((1,), dtype=list)
new_x[0] = seqs
# padding
new_x = sequence.pad_sequences(new_x, maxlen=MAX_SENTENCE_LENGTH)

my_ypred = model.predict(new_x)[0][0]
logger.info("Test using a outside example")
logger.info("%.0f\t%d\t%s" % (my_ypred, 0, my_sentence))