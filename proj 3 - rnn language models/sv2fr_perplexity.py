# -*- coding: utf-8 -*-
"""
@author: sri01

11/30/2018
"""

from sv2fr_simple_rnnlm import *
from language_model import *
import numpy as np

# input parameters
input_size = 32
hidden_size = 32
batch_size = 1
n_layers = 1

# model parameters
lr = 0.01
n_tokens = len(corpus.dictionary)
epochs = 25
clip_grad_val = 2
log_interval = 1000

# model
model = LanguageModel(n_tokens, input_size, hidden_size, n_layers).to(get_device())
optimizer = torch.optim.SGD(model.parameters(), lr)  # SGD with no momentum, no weight decay
criterion = nn.CrossEntropyLoss()


def get_perplexity(model, datas):
    model.eval()
    total_loss = 0.
    perplexity = 0
    tot_targets = 0

    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for batch, sentence in enumerate(datas):
            data, targets = get_batch(datas[batch: batch + batch_size])
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, n_tokens)
            # total_loss += len(data) * criterion(output_flat, targets).item()
            total_loss += data.shape[1] * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
            tot_targets += len(targets) / batch_size
            probs = [output_flat[position, word_id].tolist() for position, word_id in enumerate(targets)]
            perplexity += sum(probs)

    return perplexity / (tot_targets)


def write_test(model, data_source):
    model.eval()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for batch, i in enumerate(data_source):
            data, targets = get_batch(test_data[batch: batch + batch_size])
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            print(output_flat.size())
            tokens = targets

            for i, token in enumerate(tokens):
                word = corpus.dictionary.idx2word[token]
                with open('sv2fr-tst-logprob.txt', 'w+') as the_file:
                    print(str(word) + '\t' + str(output_flat[i][token].tolist()))
                    the_file.write(str(word) + '\t' + str(output_flat[i][token].tolist()) + '\n')


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model, batch_size, criterion, n_tokens, optimizer, clip_grad_val, lr, log_interval, epoch)

train_perplexity = get_perplexity(model, train_data)
dev_perplexity = get_perplexity(model, dev_data)

np.exp(-train_perplexity)
np.exp(-dev_perplexity)
write_test(model)
