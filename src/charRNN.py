
import numpy as np
import matplotlib.pyplot as plt


# loading the datasets
data = open("paul_graham_essay.txt").read().lower()

# computing text characteristics like size of vocabulary and unique elements
data_size = len(data)   # total length of the text corpus including repetitions
unique_chars = set(data)   # set of unique characters
vocab_size = len(unique_chars)   # number of unique characters
print("Data has total of {} characters with {} unique character elements!".format(str(vocab_size), str(vocab_size)))

# creating dictionaries to be used for encoding characters for training and decoding them once predicted by the model
chars_to_inds = {chars: ids for ids, chars in enumerate(unique_chars)}    # encoding each character with a number for training
inds_to_chars = {ids: chars for ids, chars in enumerate(unique_chars)}   # decoding an integer once trained by network to corresponding character

