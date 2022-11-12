import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, seq_len, n_features=1, embedding_dim=64):
        super(BiLSTMEncoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = x.reshape((x.size(0), self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (embedding, _) = self.rnn2(x)

        return embedding.reshape((x.size(0), self.embedding_dim * 2))


class BiLSTMDecoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(BiLSTMDecoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim * 2,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.output_layer = nn.Linear(self.hidden_dim * 2, n_features)

    def forward(self, x):
        x = x.reshape((x.size(0), 1, -1))
        x = x.repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.output_layer(x)
        return x.reshape((x.size(0), self.seq_len))


class BiLSTMVariationalAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, device='cpu'):
        super(BiLSTMVariationalAutoencoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder = BiLSTMEncoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = BiLSTMDecoder(seq_len, embedding_dim, n_features).to(device)
        self.fc_mu = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc_sigma = nn.Linear(embedding_dim * 2, embedding_dim)
        

    def forward(self, x):
        embedding = self.encoder(x)
        mu = self.fc_mu(embedding)
        sigma = self.fc_sigma(embedding)

        e = torch.randn_like(sigma)
        embedding_new = torch.exp(sigma) * e + mu
        out = self.decoder(embedding_new)
        return out, mu, sigma

    def encode(self, x):
        embedding = self.encoder(x)
        mu = self.fc_mu(embedding)
        # sigma = self.fc_sigma(embedding)
        # sigma = self.sigma_ln(sigma)
        # e = torch.randn_like(sigma)
        return mu
