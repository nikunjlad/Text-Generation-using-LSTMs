import matplotlib.pyplot as plt
from mdl import LSTM
data = open('paul_graham_essay.txt').read().lower()
chars = set(data)
vocab_size = len(chars)
print('data has %d characters, %d unique' % (len(data), vocab_size))

# creating dictionaries for mapping chars to ints and vice versa
char_to_idx = {w: i for i,w in enumerate(chars)}
idx_to_char = {i: w for i,w in enumerate(chars)}

# train the model
model = LSTM(char_to_idx, idx_to_char, vocab_size, epochs = 100, lr = 0.0005)
J, params = model.train(data)

# plot the results
plt.plot([i for i in range(len(J))], J)
plt.xlabel("#training iterations")
plt.ylabel("training loss")