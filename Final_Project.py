import itertools
import nltk
import sys
import os
import theano
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano, gradient_check_theano
import re, math
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

vocabulary_size = 2000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "S1"
sentence_end_token = "END"

_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '300'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def generate_setlist(model):
    # We start the sentence with the start token
    setlist = [song_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not setlist[-1] == song_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(setlist)
        sampled_song = song_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_song == song_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_song = np.argmax(samples)
        setlist.append(sampled_song)
    setlist_str = [index_to_word[x] for x in setlist]
    return setlist_str

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


#open the file and split it to sentences
with open("data_new.yem", 'rb') as f:
    shows_data = [line.strip() for line in f]
print("Parsed %d shows." % (len(shows_data)))

# Tokenize the sentences into words
shows_tokenized = [nltk.tokenize.regexp_tokenize(sent, pattern="[\number|\w|'|?|!|.| |/|:|-|(|)]+", gaps=False) for sent in shows_data]

# Count the word frequencies
song_freq = nltk.FreqDist(itertools.chain(*shows_tokenized))
print ("Found %d unique songs." % len(song_freq.items()))

# Prevent blank words in case of enclosed set of words
if vocabulary_size > len(song_freq.items()):
    vocabulary_size = len(song_freq.items())

# Get the most common words and build index_to_word and song_to_index vectors
vocab = song_freq.most_common(vocabulary_size)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
song_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
print("Using vocabulary size %d." % vocabulary_size)
print("\nExample show: '%s'" % shows_data[0])
print("\nExample show after Pre-processing: '%s'" % shows_tokenized[0])

# Create the training data
X_train = np.asarray([[song_to_index[w] for w in sent[:-1]] for sent in shows_tokenized])
y_train = np.asarray([[song_to_index[w] for w in sent[1:]] for sent in shows_tokenized])

# Print an training data example
x_example, y_example = X_train[5], y_train[5]
print ("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
print ("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

# Run the model with gradientcheck using the GPU
model = RNNTheano(vocabulary_size, hidden_dim=200)
gradient_check_theano(model, [0,1,2,3], [1,2,3,4])

# Re-Create model parameteres - alreday done
#train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

load_model_parameters_theano('./data/rnn-theano-200-867-2016-09-13-19-28-06.npz', model)


num_sentences = 10
senten_min_length = 15
string = 'show: '
new_sentences = []
for i in range(num_sentences):
    print(string + str(i))
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_setlist(model)
    new_sentences.append(sent)
    print (",".join(sent))

WORD = re.compile(r"[\number|\w|'|?|!|.| |/|:|-|(|)]+")
plot_point = []
temp = 0;
for i in range(len(new_sentences)):
    text1 = ','.join(str(e) for e in new_sentences[i])
    for j in range(len(shows_tokenized)):
        text2 = ','.join(str(e) for e in shows_tokenized[j])
        vector1 = text_to_vector(text1)
        vector2 = text_to_vector(text2)
        if temp < get_cosine(vector1, vector2):
            temp = get_cosine(vector1, vector2)
    plot_point.append(temp)

print("similarity average by cosine similarity:" + str(np.mean(plot_point)))
plt.plot(plot_point)
plt.ylabel('Cosine similarity')
plt.show()

plot_point2 = []
temp = 0;
for i in range(len(new_sentences)):
    text1 = ','.join(str(e) for e in new_sentences[i])
    set_sentence1 = set(text1.split())
    for j in range(len(shows_tokenized)):
        text2 = ','.join(str(e) for e in shows_tokenized[j])
        set_sentence2 = set(text2.split())
        similarity = (1.0 + len(set_sentence1.intersection(set_sentence2))) / (1.0 + max(len(set_sentence1), len(set_sentence2)))
        if temp < similarity:
            temp = similarity
    plot_point2.append(temp)

print("similarity average by SET similarity:" + str(np.mean(plot_point2)))
plt.plot(plot_point2)
plt.ylabel('set similarity')
plt.show()
