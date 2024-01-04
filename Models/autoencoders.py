import torch
from torch import nn

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(CVAE, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        self.encoder_input = nn.Linear(input_dim + condition_dim, hidden_dim).to(self.device)
        self.fc_mu_logvar = nn.Linear(hidden_dim, latent_dim * 2).to(self.device)

        self.encoder = nn.Sequential(
            self.encoder_input,
            nn.Sigmoid(),
            self.fc_mu_logvar,
        ).to(self.device)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + condition_dim, hidden_dim).to(self.device)
        self.decoder_output = nn.Linear(hidden_dim, input_dim).to(self.device)

        self.decoder = nn.Sequential(
            self.decoder_input,
            nn.Sigmoid(),
            self.decoder_output,
            nn.Sigmoid()
        ).to(self.device)

    def encode(self, x, c):
        x = torch.cat((x, c), dim=1)
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

    def decode(self, z, c):
        z = torch.cat((z, c), dim=1)
        x = self.decoder(z)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        z = mu + eps * std
        return z

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        self.encoder_input = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.fc_mu_logvar = nn.Linear(hidden_dim, latent_dim * 2).to(self.device)

        self.encoder = nn.Sequential(
            self.encoder_input,
            nn.Tanh(),
            self.fc_mu_logvar,
        ).to(self.device)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim).to(self.device)
        self.decoder_output = nn.Linear(hidden_dim, input_dim).to(self.device)

        self.decoder = nn.Sequential(
            self.decoder_input,
            nn.Tanh(),
            self.decoder_output,
            nn.Tanh(),
        ).to(self.device)

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # z = torch.cat((mu, logvar), dim=1)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
