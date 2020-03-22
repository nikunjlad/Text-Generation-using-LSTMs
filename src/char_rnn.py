import numpy as np

# first read the entire data from the text file
data = open('paul_graham_essay.txt', 'r').read()    # should be simple plain text file
chars = list(set(data))   # creating a list of unique characters in the text corpus
data_size, vocab_size = len(data), len(chars)
print(data_size, vocab_size)
print(chars)

# print 'data has %d characters, %d unique.' % (data_size, vocab_size)
# # char_to_ix = { ch:i for i,ch in enumerate(chars) }
# # ix_to_char = { i:ch for i,ch in enumerate(chars) }