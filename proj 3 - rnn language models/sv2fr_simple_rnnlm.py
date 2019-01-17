# -*- coding: utf-8 -*-
"""
@author: sri01

11/30/2018
"""

from data_processing import *
from utils import *
import time
import math

corpus = Corpus(os.getcwd())
train_data = corpus.train
dev_data = corpus.dev
test_data = corpus.test


# training step
def train(model, batch_size, criterion, n_tokens, optimizer, clip_grad_val, lr, log_interval, epoch):
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(batch_size)

    for batch, sentence in enumerate(train_data):
        data, targets = get_batch(train_data[batch: batch + batch_size])

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try back-propagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, n_tokens), targets)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, lr,
                              elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# evaluation step
def evaluate(model, dataset, batch_size, criterion, n_tokens):
    model.eval()
    total_loss = 0.

    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for batch, sentence in enumerate(dataset):
            data, targets = get_batch(dataset[batch: batch + batch_size])
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, n_tokens)
            # total_loss += len(data) * criterion(output_flat, targets).item()
            total_loss += data.shape[1] * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)

    return total_loss / (len(data) - 1)