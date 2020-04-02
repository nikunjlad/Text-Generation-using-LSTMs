import matplotlib.pyplot as plt
from mdl import LSTM


def rnn_chars():
    data = open('paul_graham_essay.txt').read().lower()
    chars = set(data)
    vocab_size = len(chars)
    print('data has %d characters, %d unique' % (len(data), vocab_size))
    return data, chars, vocab_size


def rnn_words():

    data = open("paul_graham_essay.txt").read()
    punctuations = ["!", ".", "?", ";", ",", "*", "+", "/", ":", "\n"]
    all_chars = list()
    words = data.split(" ")

    for ind, wd in enumerate(words):
        splits = list()
        if "\n\n" in wd:
            wd = wd.replace("\n\n", "")

        for pid, p in enumerate(punctuations):
            if p in wd:
                splits = wd.split(p)
                if "" in splits:
                    splits.remove("")

                if len(splits) == 1:
                    splits.append(p)
                else:
                    splits.insert(1, p)

        if len(splits) != 0:
            all_chars.extend(splits)
        else:
            all_chars.append(wd)

    chars = set(all_chars)
    vocab_size = len(chars)
    print('data has {} characters, {} words and {} unique'.format(str(len(data)), str(len(all_chars)), str(vocab_size)))
    return all_chars, chars, vocab_size


# data, chars, vocab_size = rnn_chars()
data, chars, vocab_size = rnn_words()

# creating dictionaries for mapping chars to ints and vice versa
char_to_idx = {w: i for i, w in enumerate(chars)}
idx_to_char = {i: w for i, w in enumerate(chars)}

# train the model
model = LSTM(char_to_idx, idx_to_char, vocab_size, epochs=10, lr=0.05)
J, params = model.train(data)

# plot the results
plt.plot([i for i in range(len(J))], J)
plt.xlabel("#training iterations")
plt.ylabel("training loss")
