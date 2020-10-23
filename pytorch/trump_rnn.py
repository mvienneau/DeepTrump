"""
Adopted from: https://github.com/dennybritz/rnn-tutorial-rnnlm
"""

import numpy as np
from csv import DictReader
import re, nltk, itertools, sys
from joblib import dump, load
from numba import cuda
from numba import *

unknown_token = "UNKNOWN_TOKEN"
START = "SENT_START"
END = "SENT_END"

nltk.download('punkt')

momentum = dict()

def WTFsoftmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))

def softmax(vec):
    vec -= min(np.array(vec))
    if max(vec) > 700:
        a = np.argsort(vec)
        aa = np.argsort(a)
        vec = vec[a]
        i = 0
        while max(vec) > 700:
            i += 1
            vec -= vec[i]
        vec = vec[aa]
    e = np.exp(vec)
    return e/np.sum(e)

class tweet:
    def __init__(self):
        raw_data = list(DictReader(open("new_realDonaldTrump_tweets.csv", 'r', encoding="utf-8")))
        raw_data = [tweet['text'].lower() for tweet in raw_data]
        self.tweets = ["%s %s %s" % (START, x, END) for x in raw_data]
        self.vocab_size = 4000

    def tokenize_and_build(self, data):
        self.token_tweets = [nltk.word_tokenize(sent) for sent in data]
        word_freq = nltk.FreqDist(itertools.chain(*self.token_tweets))
        self.vocab = word_freq.most_common(self.vocab_size-1)
        xx = []
        yy = []
        for w in enumerate(word_freq):
            xx.append(w[0])
            yy.append(word_freq.freq(w[1]))
            yy.append(word_freq[w[1]])

        self.index_to_word = [x[0] for x in self.vocab]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([(w,i) for i,w in enumerate(self.index_to_word)])

        for i, sent in enumerate(self.token_tweets):
            self.token_tweets[i] = [w if w in self.word_to_index else unknown_token for w in sent]

    def create_train(self, tokenized_data):
        self.X_train = np.asarray([[self.word_to_index[w] for w in sent[:-1]] for sent in tokenized_data])
        self.y_train = np.asarray([[self.word_to_index[w] for w in sent[1:]] for sent in tokenized_data])


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=110, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim #size of vocab
        self.hidden_dim = hidden_dim #hidden layer size
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_prop(self, x):
        T = len(x)
        s = np.zeros((T+1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))
        #Propagate!
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o,s]

    def predict(self, x):
        o, s = self.forward_prop(x)
        return np.argmax(o, axis=1)

    def calculate_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward_prop(x[i])
            correct_word_pred = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correct_word_pred))
        N = np.sum((len(y_i) for y_i in y))
        return L/N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_prop(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Delta_0 intiail guesss
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = self.bptt(x, y)
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            parameter = operator.attrgetter(pname)(self)
            print ("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]
                # estmate: (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                parameter[ix] = original_value
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print ("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print ("+h Loss: %f" % gradplus)
                    print ("-h Loss: %f" % gradminus)
                    print ("Estimated_gradient: %f" % estimated_gradient)
                    print ("Backpropagation gradient: %f" % backprop_gradient)
                    print ("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print ("Gradient check for parameter %s passed." % (pname))

    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= (learning_rate * dLdU)
        self.V -= (learning_rate * dLdV)
        self.W -= (learning_rate * dLdW)

    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        #TODO: Add momentum!
        losses = []
        ex_seen = 0
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((ex_seen, loss))
                print ("Loss after num examples seen = %d epoch = %d: %f" % (ex_seen, epoch, loss))
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print ("Setting learning rate to: ", learning_rate)
                sys.stdout.flush()
            for i in range(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                ex_seen += 1


def generate_sentence(t, model):
    new_sentence = [t.word_to_index[START]]
    while not new_sentence[-1] == t.word_to_index[END]:
        next_word_probs = model.forward_prop(new_sentence)
        #print next_word_probs[0][-1]
        sampled_word = t.word_to_index[unknown_token]
        while sampled_word == t.word_to_index[unknown_token]:
            samples = np.random.multinomial(5, next_word_probs[0][-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [t.index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


if __name__ == "__main__":
    t = tweet()
    t.tokenize_and_build(t.tweets)
    t.create_train(t.token_tweets)
    model = RNNNumpy(t.vocab_size)
    losses = model.train_with_sgd(t.X_train[:11000], t.y_train[:11000], nepoch=40, evaluate_loss_after=1)
    dump(model, 'trained_new.pkl')
    #m = joblib.load('trained_model_update.pkl')

    st = ""
    # for s in generate_sentence(t, m):
    #     if (s in ['!', '#', ':', '@', '.', '\'']):
    #         st += s
    #     else:
    #         st += s + " "
    print (st)