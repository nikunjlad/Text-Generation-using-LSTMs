import os
from io import open
import torch


class Dictionary:
    def __init__(self):
        # initializing word to index dictionary and index to word list
        # word2idx maps every word uniquely with an integer index in a dictionary
        self.word2idx = {}
        self.idx2word = []   # appending words into the list

    def add_word(self, word):

        # check if word exists in the dictionary.  If not then go inside (a boy is sitting on a chair)
        if word not in self.word2idx:
            self.idx2word.append(word)    # adding word inside dictionary
            self.word2idx[word] = len(self.idx2word) - 1    #
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))   # tokenize training data
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))  # tokenize validation data
        self.test = self.tokenize(os.path.join(path, 'test.txt'))   # tokenize testing data

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                # split the line based on empty spaces and add <eos> tag (end of sentence tag) at the end.
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)     # add the words in the dictionary

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
