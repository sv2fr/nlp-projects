import torch.nn.functional as F
import torch.nn as nn

class LanguageModel(nn.Module):
    """Language Model using RNN with encoder, LSTM units and decoder."""

    def __init__(self, n_token, n_input, n_hidden, n_layers=1):
        super(LanguageModel, self).__init__()
        self.encoder = nn.Embedding(n_token, n_input)
        self.lstm = nn.LSTM(n_input, n_hidden, n_layers)
        self.decoder = nn.Linear(n_hidden, n_token)

        self.init_weights()

        self.n_hidden = n_hidden
        self.n_layers = n_layers

    # initialize first set of weights with randomly
    def init_weights(self):
        n_range = 0.1
        self.encoder.weight.data.uniform_(-n_range, n_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-n_range, n_range)

    # forward step
    def forward(self, input, hidden):
        embedding = self.encoder(input)
        lstm_output, hidden = self.lstm(embedding, hidden)

        decoded = self.decoder(lstm_output.view(lstm_output.size(0) * lstm_output.size(1), lstm_output.size(2)))

        soft_max_out = F.log_softmax(decoded, dim=1)

        return soft_max_out, hidden

    # initialize first hidden layer with zeros
    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
