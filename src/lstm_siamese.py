import os
import torch
import torch.nn as nn


class LSTM_char_classifier(nn.Module):
    def __init__(
            self, n_channels,
            lstm_hidden_dim, lstm_n_layers,
            fc_hidden_dim, num_output,
            bidirectional, use_all_lstm_layer_state):

        super(LSTM_char_classifier, self).__init__()
        self.n_channels = n_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_n_layers = lstm_n_layers
        self.num_dir = 2 if bidirectional else 1
        self.use_all_lstm_layer_state = use_all_lstm_layer_state

        self.lstm = nn.LSTM(
            n_channels, lstm_hidden_dim, lstm_n_layers,
            batch_first=True,
            bidirectional=bidirectional)

        if use_all_lstm_layer_state:
            fc_dim = lstm_hidden_dim * lstm_n_layers * self.num_dir * 2
        else:
            fc_dim = lstm_hidden_dim * 1 * self.num_dir * 2

        self.fc = nn.Linear(
            fc_dim,
            fc_hidden_dim,
            bias=True
        )
        self.fc2 = nn.Linear(
            fc_hidden_dim,
            num_output,
            bias=True
        )

    def forward(self, x):
        init_h = torch.randn(
            self.lstm_n_layers * self.num_dir,
            x.shape[0],
            self.lstm_hidden_dim)
        init_c = torch.randn(
            self.lstm_n_layers * self.num_dir,
            x.shape[0],
            self.lstm_hidden_dim)
        if torch.cuda.is_available():
            init_h = init_h.cuda()
            init_c = init_c.cuda()

        x = x.permute(0, 2, 1)

        output, (h_n, c_n) = self.lstm(x, (init_h, init_c))
        # c_n (num_layers * num_directions, batch, hidden_size)

        if self.use_all_lstm_layer_state:
            c_n_transposed = c_n.permute(1, 0, 2)
            h_n_transposed = h_n.permute(1, 0, 2)
            # c_n_transposed (batch, num_layers * num_directions, hidden_size)
            c_n_flat = c_n_transposed.reshape(
                -1, self.lstm_n_layers * self.num_dir * self.lstm_hidden_dim)
            h_n_flat = h_n_transposed.reshape(
                -1, self.lstm_n_layers * self.num_dir * self.lstm_hidden_dim)
            # c_n_flat (batch, num_layers * num_directions * hidden_size)
            combined = torch.cat([c_n_flat, h_n_flat], dim=1)
        else:
            c_n_last = c_n.view(
                self.lstm_n_layers, self.num_dir, -1, self.lstm_hidden_dim)[-1]
            h_n_last = h_n.view(
                self.lstm_n_layers, self.num_dir, -1, self.lstm_hidden_dim)[-1]
            # c_n_last (ndir, batch, hiddensize)
            c_n_last_flat = c_n_last.permute(1, 0, 2).reshape(
                -1, self.num_dir * self.lstm_hidden_dim)
            h_n_last_flat = h_n_last.permute(1, 0, 2).reshape(
                -1, self.num_dir * self.lstm_hidden_dim)
            # cn_last_flat (batch, ndir * hiddensize)
            combined = torch.cat([c_n_last, h_n_last], dim=1)

        out = self.fc(combined)
        embedding = torch.nn.functional.relu(out)
        out = self.fc2(out)
        return out, embedding


config_keys = {
    'n_input_channels': 'dimension of each frame in raw data, yaw_pitch_roll input use 3',
    'lstm_hidden_dim': 'dimension of hidden lstm state',
    'lstm_n_layers': 'number of lstm layers',
    'fc_hidden_dim': 'number of units in hidden fc layer',
    'num_output': 'number of classes, use 26 for no nonclass, ues 27 for yes nonclass',
    'bidirectional': 'whether to use bidirectional lstm layers',
    'use_all_lstm_layer_state': 'if true, feed all lstm hidden state into fc, else feed hidden state from only last layer',
}

config = [
    (3, 200, 3, 200, 27, False, True),
    (3, 200, 5, 200, 27, False, True),
    (3, 100, 5, 200, 27, False, True),
    (3, 100, 5, 100, 27, False, True),
]
