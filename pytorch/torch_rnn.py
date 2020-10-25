import numpy as np
from csv import DictReader
import pandas as pd
import re, nltk, itertools, sys
from joblib import dump, load
from numba import cuda
from numba import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from collections import Counter


unknown_token = "UNKNOWN_TOKEN"
START = "SENT_START"
END = "SENT_END"

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)
        # Embedding Layer -- Converts word indexes to word vectors
        # embedding(torch.LongTensor([3,4])) returns the embedding vectors corresponding to the
        # words 3 and 4 in the vocab. Key: Value (key = word index, value = corresponding word vector)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        ).cuda()

        # LSTM Layer -- The main learnable part of the network (RNN)
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        ).cuda()
        # Linear Layer
        self.fc = nn.Linear(self.lstm_size, n_vocab).cuda()

    # Feed Forward (with previous state)
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    # Init the right shape
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size).cuda(),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size).cuda())

class Dataset(torch.utils.data.Dataset):
    def __init__(self,args):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    # Load the csv and tokenize the text
    def load_words(self):
        train_df = pd.read_csv('trump_all.csv')
        train_df['text'] = train_df['text'].str.lower()
        train_df['text'] = "<START> " + train_df['text'].astype(str) + " <END>"
        text = train_df['text'].str.cat(sep=' ')
        return text.split(' ')


    # Vocab Size
    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]).cuda(),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]).cuda(),
        )

def train(dataset, model, args):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).cuda()
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words

# def make_tweet(dataset, model, text):
#     full_text = text
#     for i in range(0, 10):
#         words = predict(dataset, model, text)
#         full_text = full_text + " " + words[1]
#         text = words[1]
#     return full_text

def make_tweet(dataset, model, text):
    last_word = text
    full_text = text
    while not(last_word == "<END>") and (len(full_text.split(' ')) <= 10):
        words = predict(dataset, model, text)
        new_word = words[len(text.split(' '))]
        full_text = full_text + " " + new_word
        text = new_word
        last_word = new_word


    return full_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sequence-length', type=int, default=4)
    parser.add_argument('--train', type=bool, default=False)
    args = parser.parse_args()

    # t = tweet()
    # #t.tokenize_and_build(t.tweets)
    # #t.create_train(t.token_tweets)
    # dataset = t.tweets

    dataset = Dataset(args)
    model = Model(dataset)
    model.cuda()

    if args.train:
        train(dataset, model, args)
        torch.save(model.state_dict(), 'trained_model')
    else:
        model.load_state_dict(torch.load('trained_model'))

    #print(predict(dataset, model, text='Knock knock. Whos there?'))
    print(make_tweet(dataset, model, "<START>"))