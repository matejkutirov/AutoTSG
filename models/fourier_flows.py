import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import warnings

warnings.filterwarnings("ignore")


def flip(x, dim):
    """flipping helper

    Takes a vector as an input, then flips its elements from left to right

    :x: input vector of size N x 1
    :dim: splitting dimension

    """

    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :,
        getattr(
            torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
        )().long(),
        :,
    ]
    return x.view(xsize)


def reconstruct_DFT(x, component="real"):
    """prepares input to the DFT inverse

    Takes a cropped frequency and creates a symmetric or anti-symmetric mirror of it before applying inverse DFT

    """

    if component == "real":
        x_rec = torch.cat([x[0, :], flip(x[0, :], dim=0)[1:]], dim=0)

    elif component == "imag":
        x_rec = torch.cat([x[1, :], -1 * flip(x[1, :], dim=0)[1:]], dim=0)

    return x_rec


class DFT(nn.Module):
    """
    Discrete Fourier Transform (DFT) torch module

    >> attributes <<

    :N_fft: size of the DFT transform, conventionally set to length of the input time-series or a fixed
            number of desired spectral components

    :crop_size: always equals to the size of non-redundant frequency components, i.e. N_fft / 2 since we deal with
                real-valued inputs, then the DFT is symmetric around 0 and half of the spectral components are redundant

    :base_dist: base distribution of the flow, always defined as a multi-variate normal distribution

    """

    def __init__(self, N_fft=100):
        super(DFT, self).__init__()

        self.N_fft = N_fft
        self.crop_size = int(self.N_fft / 2) + 1
        base_mu, base_cov = (
            torch.zeros(self.crop_size * 2),
            torch.eye(self.crop_size * 2),
        )
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x):
        """forward steps

        Step 1: Convert the input vector to numpy format

        Step 2: Apply FFT in numpy with FFTshift to center the spectrum around 0

        Step 3: Crop the spectrum by removing half of the real and imaginary components. Note that the FFT output size
                is 2 * N_fft because the DFT output is complex-valued rather than real-valued. After cropping, the size
                remains N_fft, similar to the input time-domain signal. In this step we also normalize the spectrum by N_fft

        Step 4: Convert spectrum back to torch tensor format

        Step 5: Compute the flow likelihood and Jacobean. Because DFT is a Vandermonde linear transform, Log-Jacob-Det = 0

        """

        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        x_numpy = x.detach().float()
        X_fft = [np.fft.fftshift(np.fft.fft(x_numpy[k, :])) for k in range(x.shape[0])]
        X_fft_train = np.array(
            [
                np.array(
                    [
                        np.real(X_fft[k])[: self.crop_size] / self.N_fft,
                        np.imag(X_fft[k])[: self.crop_size] / self.N_fft,
                    ]
                )
                for k in range(len(X_fft))
            ]
        )
        x_fft = torch.from_numpy(X_fft_train).float()

        log_pz = self.base_dist.log_prob(
            x_fft.view(-1, x_fft.shape[1] * x_fft.shape[2])
        )
        log_jacob = 0

        return x_fft, log_pz, log_jacob

    def inverse(self, x):
        """Inverse steps

        Step 1: Convert the input vector to numpy format with size NUM_SAMPLES x 2 x N_fft
                Second dimension indexes the real and imaginary components.

        Step 2: Apply FFT in numpy with FFTshift to center the spectrum around 0

        Step 3: Crop the spectrum by removing half of the real and imaginary components. Note that the FFT output size
                is 2 * N_fft because the DFT output is complex-valued rather than real-valued. After cropping, the size
                remains N_fft, similar to the input time-domain signal. In this step we also normalize the spectrum by N_fft

        Step 4: Convert spectrum back to torch tensor format

        Step 5: Compute the flow likelihood and Jacobean. Because DFT is a Vandermonde linear transform, Log-Jacob-Det = 0

        """

        x_numpy = x.view((-1, 2, self.crop_size))

        x_numpy_r = [
            reconstruct_DFT(x_numpy[u, :, :], component="real").detach().numpy()
            for u in range(x_numpy.shape[0])
        ]
        x_numpy_i = [
            reconstruct_DFT(x_numpy[u, :, :], component="imag").detach().numpy()
            for u in range(x_numpy.shape[0])
        ]

        x_ifft = [
            self.N_fft
            * np.real(np.fft.ifft(np.fft.ifftshift(x_numpy_r[u] + 1j * x_numpy_i[u])))
            for u in range(x_numpy.shape[0])
        ]
        x_ifft_out = torch.from_numpy(np.array(x_ifft)).float()

        return x_ifft_out


class SpectralFilter(nn.Module):
    """
    Spectral Filter torch module

    >> attributes <<

    :d: number of input dimensions

    :k: dimension of split in the input space

    :FFT: number of FFT components

    :hidden: number of hidden units in the spectral filter layer

    :flip: boolean indicator on whether to flip the split dimensions

    :RNN: boolean indicator on whether to use an RNN in spectral filtering

    """

    def __init__(self, d, k, FFT, hidden, flip=False, RNN=False):
        super().__init__()

        self.d, self.k = d, k

        # For flows, we always split the input in half
        # k should be d/2, and out_size should also be d/2
        if FFT:
            # Ensure k is exactly half of d for proper splitting
            self.k = d // 2
            self.out_size = self.k  # Both halves should be the same size
            self.pz_size = self.d  # Total output size
            self.in_size = self.k
        else:
            # Ensure k is exactly half of d for proper splitting
            self.k = d // 2
            self.out_size = self.k  # Both halves should be the same size
            self.pz_size = self.d  # Total output size
            self.in_size = self.k

        self.sig_net = (
            nn.Sequential(  # RNN(mode="RNN", HIDDEN_UNITS=20, INPUT_SIZE=1,),
                nn.Linear(self.in_size, hidden),
                nn.Sigmoid(),  # nn.LeakyReLU(),
                nn.Linear(hidden, hidden),
                nn.Sigmoid(),  # nn.Tanh(),
                nn.Linear(hidden, self.out_size),
            )
        )

        self.mu_net = nn.Sequential(  # RNN(mode="RNN", HIDDEN_UNITS=20, INPUT_SIZE=1,),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),  # nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),  # nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        # Initialize weights with smaller values for stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

        base_mu, base_cov = torch.zeros(self.pz_size), torch.eye(self.pz_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x):
        """forward steps

        Similar to RealNVP, see:
        Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio.
        "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016).

        """

        # Always split the same way: x1 = first k dimensions, x2 = remaining dimensions
        x1, x2 = x[:, : self.k], x[:, self.k :]

        # Apply transformation using x1 as conditioning
        sig = self.sig_net(x1).view(-1, self.out_size)
        mu = self.mu_net(x1).view(-1, self.out_size)

        # Transform x2 using the parameters from x1
        z1, z2 = x1, x2 * torch.exp(sig) + mu

        # If flip is True, swap the output order but keep the same transformation logic
        if hasattr(self, "flip") and self.flip:
            z1, z2 = z2, z1

        z_hat = torch.cat([z1, z2], dim=-1)

        # Check for NaN values and replace them
        if torch.isnan(z_hat).any():
            z_hat = torch.nan_to_num(z_hat, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clip z_hat to reasonable bounds
        z_hat = torch.clamp(z_hat, -20, 20)

        log_pz = self.base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)

        return z_hat, log_pz, log_jacob

    def inverse(self, Z):
        # Always split the same way: z1 = first k dimensions, z2 = remaining dimensions
        z1, z2 = Z[:, : self.k], Z[:, self.k :]

        # Use z1 as conditioning to get transformation parameters
        x1 = z1
        sig_in = self.sig_net(z1).view(-1, self.out_size)
        # Clip sig_in values
        sig_in = torch.clamp(sig_in, -10, 10)

        mu = self.mu_net(z1).view(-1, self.out_size)
        # Clip mu values
        mu = torch.clamp(mu, -10, 10)

        # Use safe exponential to prevent overflow
        exp_neg_sig = torch.exp(torch.clamp(-sig_in, -20, 20))

        # Inverse transform z2 using the parameters from z1
        x2 = (z2 - mu) * exp_neg_sig

        result = torch.cat([x1, x2], -1)

        # Check for NaN values and replace them
        if torch.isnan(result).any():
            result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result


class AttentionFilter(nn.Module):
    """
    Attention Filter torch module

    >> attributes <<

    :d: number of input dimensions

    :k: dimension of split in the input space

    :FFT: number of FFT components

    :hidden: number of hidden units in the spectral filter layer

    :flip: boolean indicator on whether to flip the split dimensions

    :RNN: boolean indicator on whether to use an RNN in spectral filtering

    """

    def __init__(self, d, k, FFT, hidden, flip=False):
        super().__init__()

        self.d, self.k = d, k

        # For flows, we always split the input in half
        # k should be d/2, and out_size should also be d/2
        if FFT:
            # Ensure k is exactly half of d for proper splitting
            self.k = d // 2
            self.out_size = self.k  # Both halves should be the same size
            self.pz_size = self.d  # Total output size
            self.in_size = self.k
        else:
            # Ensure k is exactly half of d for proper splitting
            self.k = d // 2
            self.out_size = self.k  # Both halves should be the same size
            self.pz_size = self.d  # Total output size
            self.in_size = self.k

        # LSTM layer for attention - adjust hidden size for small inputs
        lstm_hidden = min(20, max(5, self.in_size * 2))  # Adaptive hidden size
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)

        # Add projection layer to match LSTM output to expected input size
        self.lstm_projection = nn.Linear(lstm_hidden, self.in_size)

        self.sig_net = nn.Sequential(
            nn.Linear(self.in_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        self.mu_net = nn.Sequential(
            nn.Linear(self.in_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        # Initialize weights with smaller values for stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

        base_mu, base_cov = torch.zeros(self.pz_size), torch.eye(self.pz_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x):
        """forward steps

        Similar to RealNVP, see:
        Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio.
        "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016).

        """

        # Always split the same way: x1 = first k dimensions, x2 = remaining dimensions
        x1, x2 = x[:, : self.k], x[:, self.k :]

        # Apply LSTM attention using x1 as conditioning
        x1_lstm = x1.unsqueeze(-1)  # Add feature dimension for LSTM
        lstm_out, _ = self.lstm(x1_lstm)
        x1_attended = lstm_out[:, -1, :]  # Take last LSTM output

        # Project LSTM output to expected input size
        x1_attended = self.lstm_projection(x1_attended)

        # Clip attended values to prevent extreme values
        x1_attended = torch.clamp(x1_attended, -10, 10)

        # Generate transformation parameters using x1_attended as conditioning
        sig = self.sig_net(x1_attended).view(-1, self.out_size)
        # Clip sig values to prevent extreme scaling that leads to NaN
        sig = torch.clamp(sig, -10, 10)

        mu = self.mu_net(x1_attended).view(-1, self.out_size)
        # Clip mu values to prevent extreme shifts
        mu = torch.clamp(mu, -10, 10)

        # Transform x2 using the parameters from x1_attended
        z1, z2 = x1, x2 * torch.exp(sig) + mu

        # If flip is True, swap the output order but keep the same transformation logic
        if hasattr(self, "flip") and self.flip:
            z1, z2 = z2, z1

        z_hat = torch.cat([z1, z2], dim=-1)

        # Check for NaN values and replace them
        if torch.isnan(z_hat).any():
            z_hat = torch.nan_to_num(z_hat, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clip z_hat to reasonable bounds
        z_hat = torch.clamp(z_hat, -20, 20)

        log_pz = self.base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)

        return z_hat, log_pz, log_jacob

    def inverse(self, Z):
        # Always split the same way: z1 = first k dimensions, z2 = remaining dimensions
        z1, z2 = Z[:, : self.k], Z[:, self.k :]

        # Use z1 as conditioning to get transformation parameters
        x1 = z1
        x1_lstm = x1.unsqueeze(-1)
        lstm_out, _ = self.lstm(x1_lstm)
        x1_attended = lstm_out[:, -1, :]

        # Project LSTM output to expected input size
        x1_attended = self.lstm_projection(x1_attended)

        # Clip attended values
        x1_attended = torch.clamp(x1_attended, -10, 10)

        # Generate transformation parameters using x1_attended as conditioning
        sig_in = self.sig_net(x1_attended).view(-1, self.out_size)
        # Clip sig_in values
        sig_in = torch.clamp(sig_in, -10, 10)

        mu = self.mu_net(x1_attended).view(-1, self.out_size)
        # Clip mu values
        mu = torch.clamp(mu, -10, 10)

        # Use safe exponential to prevent overflow
        exp_neg_sig = torch.exp(torch.clamp(-sig_in, -20, 20))

        # Inverse transform z2 using the parameters from x1_attended
        x2 = (z2 - mu) * exp_neg_sig

        result = torch.cat([x1, x2], -1)

        # Check for NaN values and replace them
        if torch.isnan(result).any():
            result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result


class FourierFlow(nn.Module):
    def __init__(self, hidden, fft_size, n_flows, FFT=True, flip=True, normalize=False):
        super().__init__()

        self.hidden = hidden
        self.fft_size = fft_size
        self.n_flows = n_flows
        self.FFT = FFT
        self.normalize = normalize

        if self.FFT:
            self.FourierTransform = DFT(self.fft_size)

        self.bijectors = nn.ModuleList()
        if flip:
            self.flips = [True if i % 2 else False for i in range(n_flows)]

        else:
            self.flips = [False for i in range(n_flows)]

        for i in range(n_flows):
            if i % 2 == 0:
                self.bijectors.append(
                    SpectralFilter(
                        d=self.fft_size + 1,  # +1 for FFT
                        k=int(np.floor((self.fft_size + 1) / 2)),
                        FFT=self.FFT,
                        hidden=self.hidden,
                        flip=self.flips[i],
                    )
                )
            else:
                self.bijectors.append(
                    AttentionFilter(
                        d=self.fft_size + 1,  # +1 for FFT
                        k=int(np.floor((self.fft_size + 1) / 2)),
                        FFT=self.FFT,
                        hidden=self.hidden,
                        flip=self.flips[i],
                    )
                )

    def forward(self, x):
        if self.FFT:
            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / self.fft_std

            # Use the actual FFT flattened size that was set during training
            if hasattr(self, "fft_flat_size"):
                x = x.view(-1, self.fft_flat_size)  # [6280, 32]
            else:
                # Fallback to original behavior
                x = x.view(-1, self.d + 1)

        log_jacobs = []

        for bijector in self.bijectors:
            x, log_pz, lj = bijector(x)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):
        for bijector in reversed(self.bijectors):
            z = bijector.inverse(z)

        if self.FFT:
            if self.normalize:
                # Use the actual FFT flattened size that was set during training
                if hasattr(self, "fft_flat_size"):
                    # Check if z has the expected shape for reshaping
                    if z.shape[1] == self.fft_flat_size:
                        # Calculate the second dimension for reshape
                        # fft_flat_size = self.d * second_dim, so second_dim = fft_flat_size // self.d
                        second_dim = self.fft_flat_size // self.d
                        print(
                            f"  FFT denormalization: reshaping {z.shape} to [{z.shape[0]}, {self.d}, {second_dim}]"
                        )

                        # Reshape z to match FFT dimensions before denormalization
                        z_reshaped = z.view(
                            z.shape[0], self.d, second_dim
                        )  # [n_samples, 2, 16]

                        # Reshape fft_std and fft_mean to match z_reshaped dimensions
                        # fft_std.shape = [2, 16], need to broadcast to [1, 2, 16]
                        fft_std_broadcast = self.fft_std.view(1, self.d, second_dim)
                        fft_mean_broadcast = self.fft_mean.view(1, self.d, second_dim)

                        z_denorm = z_reshaped * fft_std_broadcast + fft_mean_broadcast
                        z = z_denorm.view(z.shape[0], -1)  # Flatten back
                        print(f"  FFT denormalization successful")
                    else:
                        print(
                            f"  Warning: z.shape={z.shape} doesn't match fft_flat_size={self.fft_flat_size}"
                        )
                        print(f"  Skipping FFT denormalization")
                else:
                    # Fallback: use z's actual shape
                    z = z * self.fft_std.view(1, -1) + self.fft_mean.view(1, -1)

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):
        X_train = torch.from_numpy(np.array(X)).float()

        # for normalizing the spectral transforms
        X_train_spectral = self.FourierTransform(X_train)[0]

        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        self.fft_std = torch.std(X_train_spectral, dim=0)

        # Use actual FFT output dimensions, not assumed ones
        self.d = X_train_spectral.shape[1]  # Use actual FFT output dimension
        self.k = int(np.floor(self.d / 2))  # Split in half

        # Calculate the actual flattened FFT size for normalization
        self.fft_flat_size = (
            X_train_spectral.shape[1] * X_train_spectral.shape[2]
        )  # 2 * 16 = 32

        # Recreate bijectors with correct dimensions
        self.bijectors = nn.ModuleList()
        for i in range(self.n_flows):
            if i % 2 == 0:
                self.bijectors.append(
                    SpectralFilter(
                        d=self.fft_flat_size,  # Use flattened FFT size
                        k=self.fft_flat_size // 2,  # Split flattened size in half
                        FFT=self.FFT,
                        hidden=self.hidden,
                        flip=self.flips[i],
                    )
                )
            else:
                self.bijectors.append(
                    AttentionFilter(
                        d=self.fft_flat_size,  # Use flattened FFT size
                        k=self.fft_flat_size // 2,  # Split flattened size in half
                        FFT=self.FFT,
                        hidden=self.hidden,
                        flip=self.flips[i],
                    )
                )

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        all_epochs = int(np.floor(epochs / display_step))

        for step in range(epochs):
            optim.zero_grad()

            try:
                z, log_pz, log_jacob = self(X_train)

                # Check for invalid values before computing loss
                if (
                    torch.isnan(z).any()
                    or torch.isnan(log_pz).any()
                    or torch.isnan(log_jacob).any()
                ):
                    continue

                loss = (-log_pz - log_jacob).mean()

                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at step {step}, skipping...")
                    continue

                losses.append(loss.detach().numpy())

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optim.step()
                scheduler.step()

            except Exception as e:
                print(f"Error at step {step}: {e}")
                continue

            if ((step % display_step) == 0) | (step == epochs - 1):
                current_epochs = int(np.floor((step + 1) / display_step))
                remaining_epochs = int(all_epochs - current_epochs)

                progress_signs = current_epochs * "|" + remaining_epochs * "-"
                display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"

                print(display_string % (step, epochs, loss))

            if step == epochs - 1:
                print("Finished training!")

        return losses

    def sample(self, n_samples):
        if self.FFT:
            # Use the actual FFT flattened size that was set during training
            if hasattr(self, "fft_flat_size"):
                mu, cov = torch.zeros(self.fft_flat_size), torch.eye(self.fft_flat_size)
            else:
                # Fallback to a safe default
                mu, cov = torch.zeros(32), torch.eye(32)
        else:
            if hasattr(self, "fft_flat_size"):
                mu, cov = torch.zeros(self.fft_flat_size), torch.eye(self.fft_flat_size)
            else:
                # Fallback to a safe default
                mu, cov = torch.zeros(30), torch.eye(30)

        # Create base distribution with correct dimensions
        base_dist = MultivariateNormal(mu, cov)
        z = base_dist.sample((n_samples,))

        # Ensure z has the correct shape for the bijectors
        if hasattr(self, "fft_flat_size") and z.shape[1] != self.fft_flat_size:
            # Pad or truncate z to match the expected size
            if z.shape[1] < self.fft_flat_size:
                # Pad with zeros
                padding = torch.zeros(n_samples, self.fft_flat_size - z.shape[1])
                z = torch.cat([z, padding], dim=1)
            else:
                # Truncate
                z = z[:, : self.fft_flat_size]

        # Apply inverse transform through bijectors
        for i, bijector in enumerate(self.bijectors):
            # The bijector already has its flip state set during creation
            z = bijector.inverse(z)

        # Denormalize if using FFT
        if self.FFT and self.normalize:
            # Use the actual FFT flattened size that was set during training
            if hasattr(self, "fft_flat_size"):
                # Check if z has the expected shape for reshaping
                if z.shape[1] == self.fft_flat_size:
                    # Calculate the second dimension for reshape
                    # fft_flat_size = self.d * second_dim, so second_dim = fft_flat_size // self.d
                    second_dim = self.fft_flat_size // self.d
                    # Reshape z to match FFT dimensions before denormalization
                    z_reshaped = z.view(
                        z.shape[0], self.d, second_dim
                    )  # [n_samples, 2, 16]

                    # Reshape fft_std and fft_mean to match z_reshaped dimensions
                    # fft_std.shape = [2, 16], need to broadcast to [1, 2, 16]
                    fft_std_broadcast = self.fft_std.view(1, self.d, second_dim)
                    fft_mean_broadcast = self.fft_mean.view(1, self.d, second_dim)

                    z_denorm = z_reshaped * fft_std_broadcast + fft_mean_broadcast
                    z = z_denorm.view(z.shape[0], -1)  # Flatten back
            else:
                # Fallback to original behavior
                z = z * self.fft_std.view(1, -1) + self.fft_mean.view(1, -1)

        # Apply inverse FFT transform to convert back to time domain
        if self.FFT:
            z = self.FourierTransform.inverse(z)

            # Ensure the output has exactly the expected time-domain dimensions
            expected_time_dim = self.fft_size  # Should be 30
            if z.shape[1] != expected_time_dim:
                if z.shape[1] > expected_time_dim:
                    # Truncate to expected size
                    z = z[:, :expected_time_dim]
                else:
                    # Pad with zeros to expected size
                    padding = torch.zeros(z.shape[0], expected_time_dim - z.shape[1])
                    z = torch.cat([z, padding], dim=1)

        return z.detach().numpy()


# class RealNVP(nn.Module):
#     """RealNVP model without FFT transformation"""

#     def __init__(self, hidden, T, n_flows, flip=True, normalize=False):
#         super().__init__()

#         self.d = T
#         self.k = int(T / 2) + 1
#         self.normalize = normalize
#         self.FFT = False

#         if flip:
#             self.flips = [True if i % 2 else False for i in range(n_flows)]
#         else:
#             self.flips = [False for i in range(n_flows)]

#         self.bijectors = nn.ModuleList(
#             [
#                 SpectralFilter(
#                     self.d, self.k, self.FFT, hidden=hidden, flip=self.flips[_]
#                 )
#                 for _ in range(n_flows)
#             ]
#         )

#     def forward(self, x):
#         log_jacobs = []
#         for bijector, f in zip(self.bijectors, self.flips):
#             x, log_pz, lj = bijector(x, flip=f)
#             log_jacobs.append(lj)
#         return x, log_pz, sum(log_jacobs)

#     def inverse(self, z):
#         for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
#             z = bijector.inverse(z, flip=f)
#         return z.detach().numpy()

#     def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):
#         X_train = torch.from_numpy(np.array(X)).float()

#         self.d = X_train.shape[1]
#         self.k = int(np.floor(X_train.shape[1] / 2))

#         optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

#         losses = []
#         all_epochs = int(np.floor(epochs / display_step))

#         for step in range(epochs):
#             optim.zero_grad()

#             z, log_pz, log_jacob = self(X_train)
#             loss = (-log_pz - log_jacob).mean()
#             losses.append(loss.detach().numpy())

#             loss.backward()
#             optim.step()
#             scheduler.step()

#             if ((step % display_step) == 0) | (step == epochs - 1):
#                 current_epochs = int(np.floor((step + 1) / display_step))
#                 remaining_epochs = int(all_epochs - current_epochs)
#                 progress_signs = current_epochs * "|" + remaining_epochs * "-"
#                 display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"
#                 print(display_string % (step, epochs, loss))

#             if step == epochs - 1:
#                 print("Finished training!")

#         return losses

#     def sample(self, n_samples):
#         mu, cov = torch.zeros(self.d), torch.eye(self.d)
#         p_Z = MultivariateNormal(mu, cov)
#         z = p_Z.rsample(sample_shape=(n_samples,))
#         X_sample = self.inverse(z)
#         return X_sample


# class TimeFlow(nn.Module):
#     """TimeFlow model with attention filtering"""

#     def __init__(self, hidden, T, n_flows, flip=True, normalize=False):
#         super().__init__()

#         self.d = T
#         self.k = int(T / 2) + 1
#         self.normalize = normalize
#         self.FFT = False

#         if flip:
#             self.flips = [True if i % 2 else False for i in range(n_flows)]
#         else:
#             self.flips = [False for i in range(n_flows)]

#         self.bijectors = nn.ModuleList(
#             [
#                 AttentionFilter(
#                     self.d, self.k, self.FFT, hidden=hidden, flip=self.flips[_]
#                 )
#                 for _ in range(n_flows)
#             ]
#         )

#     def forward(self, x):
#         log_jacobs = []
#         for bijector, f in zip(self.bijectors, self.flips):
#             x, log_pz, lj = bijector(x, flip=f)
#             log_jacobs.append(lj)
#         return x, log_pz, sum(log_jacobs)

#     def inverse(self, z):
#         for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
#             z = bijector.inverse(z, flip=f)
#         return z.detach().numpy()

#     def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):
#         X_train = torch.from_numpy(np.array(X)).float()

#         self.d = X_train.shape[1]
#         self.k = int(np.floor(X_train.shape[1] / 2))

#         optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

#         losses = []
#         all_epochs = int(np.floor(epochs / display_step))

#         for step in range(epochs):
#             optim.zero_grad()

#             z, log_pz, log_jacob = self(X_train)
#             loss = (-log_pz - log_jacob).mean()
#             losses.append(loss.detach().numpy())

#             loss.backward()
#             optim.step()
#             scheduler.step()

#             if ((step % display_step) == 0) | (step == epochs - 1):
#                 current_epochs = int(np.floor((step + 1) / display_step))
#                 remaining_epochs = int(all_epochs - current_epochs)
#                 progress_signs = current_epochs * "|" + remaining_epochs * "-"
#                 display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"
#                 print(display_string % (step, epochs, loss))

#             if step == epochs - 1:
#                 print("Finished training!")

#         return losses

#     def sample(self, n_samples):
#         mu, cov = torch.zeros(self.d), torch.eye(self.d)
#         p_Z = MultivariateNormal(mu, cov)
#         z = p_Z.rsample(sample_shape=(n_samples,))
#         X_sample = self.inverse(z)
#         return X_sample
