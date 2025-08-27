import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Norm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H


class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=output_dim, num_layers=num_layers
        )
        self.fc = nn.Linear(output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        g_outputs, _ = self.rnn(input)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E


class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat


def batch_generator(data, time, batch_size):
    """Generate batches of data with time information"""
    data_len = len(data)
    indices = np.random.permutation(data_len)

    for i in range(0, data_len, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_data = [data[j] for j in batch_indices]
        batch_time = [time[j] for j in batch_indices]

        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_data)
        padded_data = []
        for seq in batch_data:
            if len(seq) < max_len:
                # Pad with last value
                padded = np.vstack([seq, np.tile(seq[-1], (max_len - len(seq), 1))])
                padded_data.append(padded)
            else:
                padded_data.append(seq)

        yield np.array(padded_data), batch_time


def random_generator(batch_size, z_dim, time_info, max_seq_len, mean=0.0, std=1.0):
    """Generate random noise for generator"""
    Z = np.random.normal(mean, std, (batch_size, max_seq_len, z_dim))
    return Z


def extract_time(data):
    """Extract time information from data"""
    time_info = [len(seq) for seq in data]
    max_seq_len = max(time_info)
    return time_info, max_seq_len


def NormMinMax(data):
    """Normalize data to [0,1] range"""
    data_flat = np.concatenate(data, axis=0)
    min_val = np.min(data_flat, axis=0)
    max_val = np.max(data_flat, axis=0)

    normalized_data = []
    for seq in data:
        norm_seq = (seq - min_val) / (max_val - min_val + 1e-8)
        normalized_data.append(norm_seq)

    return normalized_data, min_val, max_val


class TimeGAN:
    def __init__(
        self,
        seq_len,
        feat_dim,
        hidden_dim=24,
        num_layers=3,
        z_dim=6,
        batch_size=16,
        lr=0.001,
    ):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.nete = Encoder(feat_dim, hidden_dim, num_layers).to(self.device)
        self.netr = Recovery(hidden_dim, feat_dim, num_layers).to(self.device)
        self.netg = Generator(z_dim, hidden_dim, num_layers).to(self.device)
        self.netd = Discriminator(hidden_dim, num_layers).to(self.device)
        self.nets = Supervisor(hidden_dim, num_layers).to(self.device)

        # Loss functions
        self.l_mse = nn.MSELoss()
        self.l_bce = nn.BCELoss()

        # Optimizers
        self.optimizer_e = optim.Adam(self.nete.parameters(), lr=lr)
        self.optimizer_r = optim.Adam(self.netr.parameters(), lr=lr)
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=lr)
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=lr)
        self.optimizer_s = optim.Adam(self.nets.parameters(), lr=lr)

    def train(self, data, iterations=1000):
        """Train TimeGAN model"""
        # Normalize data
        self.ori_data, self.min_val, self.max_val = NormMinMax(data)
        self.ori_time, self.max_seq_len = extract_time(self.ori_data)

        print("Training TimeGAN...")

        # Phase 1: Train encoder and recovery
        for iter in range(iterations):
            self.train_encoder_recovery()
            if iter % 100 == 0:
                print(f"Encoder training: {iter}/{iterations}")

        # Phase 2: Train supervisor
        for iter in range(iterations):
            self.train_supervisor()
            if iter % 100 == 0:
                print(f"Supervisor training: {iter}/{iterations}")

        # Phase 3: Joint training
        for iter in range(iterations):
            for _ in range(2):
                self.train_generator()
                self.train_encoder_recovery_joint()
            self.train_discriminator()
            if iter % 100 == 0:
                print(f"Joint training: {iter}/{iterations}")

        print("Training completed!")

    def train_encoder_recovery(self):
        """Train encoder and recovery networks"""
        self.nete.train()
        self.netr.train()

        # Get batch
        batch_data, _ = next(
            batch_generator(self.ori_data, self.ori_time, self.batch_size)
        )
        X = torch.tensor(batch_data, dtype=torch.float32).to(self.device)

        # Forward pass
        H = self.nete(X)
        X_tilde = self.netr(H)

        # Backward pass
        loss = self.l_mse(X_tilde, X)

        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        loss.backward()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def train_supervisor(self):
        """Train supervisor network"""
        self.nets.train()

        batch_data, _ = next(
            batch_generator(self.ori_data, self.ori_time, self.batch_size)
        )
        X = torch.tensor(batch_data, dtype=torch.float32).to(self.device)

        H = self.nete(X)
        H_supervise = self.nets(H)

        loss = self.l_mse(H[:, 1:, :], H_supervise[:, :-1, :])

        self.optimizer_s.zero_grad()
        loss.backward()
        self.optimizer_s.step()

    def train_generator(self):
        """Train generator network"""
        self.netg.train()
        self.nets.train()

        batch_data, time_info = next(
            batch_generator(self.ori_data, self.ori_time, self.batch_size)
        )
        X = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
        Z = random_generator(self.batch_size, self.z_dim, time_info, self.max_seq_len)
        Z = torch.tensor(Z, dtype=torch.float32).to(self.device)

        H = self.nete(X)
        E_hat = self.netg(Z)
        H_hat = self.nets(E_hat)
        X_hat = self.netr(H_hat)

        # Generator loss
        Y_fake = self.netd(H_hat)
        Y_fake_e = self.netd(E_hat)

        loss_g = self.l_bce(Y_fake, torch.ones_like(Y_fake)) + self.l_bce(
            Y_fake_e, torch.ones_like(Y_fake_e)
        )

        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()
        self.optimizer_s.step()

    def train_encoder_recovery_joint(self):
        """Joint training of encoder and recovery with supervisor"""
        self.nete.train()
        self.netr.train()

        batch_data, _ = next(
            batch_generator(self.ori_data, self.ori_time, self.batch_size)
        )
        X = torch.tensor(batch_data, dtype=torch.float32).to(self.device)

        H = self.nete(X)
        X_tilde = self.netr(H)
        H_supervise = self.nets(H)

        loss_er = self.l_mse(X_tilde, X)
        loss_s = self.l_mse(H[:, 1:, :], H_supervise[:, :-1, :])
        loss = loss_er + 0.1 * loss_s

        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        loss.backward()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def train_discriminator(self):
        """Train discriminator network"""
        self.netd.train()

        batch_data, time_info = next(
            batch_generator(self.ori_data, self.ori_time, self.batch_size)
        )
        X = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
        Z = random_generator(self.batch_size, self.z_dim, time_info, self.max_seq_len)
        Z = torch.tensor(Z, dtype=torch.float32).to(self.device)

        H = self.nete(X)
        E_hat = self.netg(Z)
        H_hat = self.nets(E_hat)

        Y_real = self.netd(H)
        Y_fake = self.netd(H_hat)
        Y_fake_e = self.netd(E_hat)

        loss_d = (
            self.l_bce(Y_real, torch.ones_like(Y_real))
            + self.l_bce(Y_fake, torch.zeros_like(Y_fake))
            + self.l_bce(Y_fake_e, torch.zeros_like(Y_fake_e))
        )

        if loss_d > 0.15:
            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

    def generate(self, num_samples):
        """Generate synthetic time series data"""
        self.nete.eval()
        self.netr.eval()
        self.netg.eval()
        self.nets.eval()

        with torch.no_grad():
            batch_data, time_info = next(
                batch_generator(self.ori_data, self.ori_time, num_samples)
            )
            Z = random_generator(num_samples, self.z_dim, time_info, self.max_seq_len)
            Z = torch.tensor(Z, dtype=torch.float32).to(self.device)

            E_hat = self.netg(Z)
            H_hat = self.nets(E_hat)
            generated_data = self.netr(H_hat).cpu().numpy()

            # Denormalize
            generated_data = (
                generated_data * (self.max_val - self.min_val) + self.min_val
            )

            # Trim to original sequence lengths
            final_data = []
            for i, seq_len in enumerate(time_info):
                final_data.append(generated_data[i, :seq_len, :])

            return final_data

    def save_model(self, path):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "nete_state_dict": self.nete.state_dict(),
                "netr_state_dict": self.netr.state_dict(),
                "netg_state_dict": self.netg.state_dict(),
                "netd_state_dict": self.netd.state_dict(),
                "nets_state_dict": self.nets.state_dict(),
            },
            os.path.join(path, "timegan_model.pth"),
        )

    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(os.path.join(path, "timegan_model.pth"))
        self.nete.load_state_dict(checkpoint["nete_state_dict"])
        self.netr.load_state_dict(checkpoint["netr_state_dict"])
        self.netg.load_state_dict(checkpoint["netg_state_dict"])
        self.netd.load_state_dict(checkpoint["netd_state_dict"])
        self.nets.load_state_dict(checkpoint["nets_state_dict"])
