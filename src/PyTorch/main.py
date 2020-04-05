# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import data
import model
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

###############################################################################
# Parsing command line arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/paul_graham',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--plt-name', type=str, default='',
                    help='path to export the loss curve plots')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()  # get argument parser object

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:3" if args.cuda else "cpu")  # set to cuda if using cuda or use the cpu

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

# function to convert data into batches.
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    print("Data size: {}, batch size: {}".format(str(data.size(0)), str(bsz)))
    print("Number of batches: {}".format(str(nbatch)))
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    print("Trimmed data shape: {}".format(str(data.shape)))
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print()
    return data.to(device)


# we want to have a sequence of length 10 for the validation and the test dataset and of length 20 for training dataset
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
print("Training data shape: {}".format(str(train_data.shape)))
val_data = batchify(corpus.valid, eval_batch_size)
print("Validation data shape: {}".format(str(val_data.shape)))
test_data = batchify(corpus.test, eval_batch_size)
print("Testing data shape: {}".format(str(test_data.shape)))

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)  # size of the vocabulary
print("Size of vocabulary: {}".format(str(ntokens)))
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(
        device)

criterion = nn.NLLLoss()

print("Encoder shape: {}".format(str(model.encoder.weight.shape)))
print("Decoder shape: {}".format(str(model.decoder.weight.shape)))


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def plot_curves(tr_loss, val_loss, epochs):
    epochs_list = [i + 1 for i in range(epochs)]
    plt.plot(epochs_list, tr_loss, color='blue', label="Training loss")
    plt.plot(epochs_list, val_loss, color='green', label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train-Valid Loss Curves")
    plt.legend()
    plt.savefig(args.plt_name)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)  # if transformer model is not there then initialize weights
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()  # train the model
    total_loss = 0.  # initialize total loss of the epoch to 0
    start_time = time.time()  # start the time initially for the batch calculation
    ntokens = len(corpus.dictionary)  # vocabulary length /// SAME

    # if model is not a Transformer, initialize hidden weights
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
        print("Hidden layer shape: {}".format(str(hidden[0].shape)))

    count = 0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        count += 1

    return total_loss / count


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr  # learning rate
best_val_loss = None  # best validation loss

# At any point you can hit Ctrl + C to break out of training early.
train_loss = list()
valid_loss = list()
ct = 0
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()  # start the time for the epoch run
        train_loss.append(train())  # call the training function and append the avg loss of epoch over batches in a list
        val_loss = evaluate(val_data)   # calculate the validation loss
        valid_loss.append(val_loss)  # append the validation loss in a list
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)   # keep saving the best model as and when the validation loss falls below best loss
            best_val_loss = val_loss   # new best loss is the recently found validation loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
        ct += 1
    plot_curves(train_loss, valid_loss, args.epochs)   # plot the loss curves against the epochs
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')     # on pressing Ctrl + C, early stop the model and exit training
    ct = len(train_loss)
    plot_curves(train_loss, valid_loss, ct)

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()  #

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
