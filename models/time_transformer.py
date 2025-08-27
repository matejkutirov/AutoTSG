import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import List, Tuple, Optional


class Sampling(nn.Module):
    """Sampling layer for VAE that uses (z_mean, z_log_var) to sample z."""

    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(batch_size, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class TimeSformerLayer(nn.Module):
    """TimeSformer layer combining TCN and Transformer with cross-attention."""

    def __init__(self, filters, head_size, num_heads, k_size, dilation, dropout=0.0):
        super().__init__()

        self.filters = filters
        self.head_size = head_size
        self.num_heads = num_heads
        self.k_size = k_size
        self.dilation = dilation
        self.dropout = dropout

        # TCN components
        self.tcn_conv = nn.Conv1d(
            filters, filters, k_size, padding="same", dilation=dilation
        )
        self.tcn_norm = nn.LayerNorm(filters)
        self.tcn_dropout = nn.Dropout(dropout)

        # Transformer components
        self.self_attention = nn.MultiheadAttention(
            filters, num_heads, dropout=dropout, batch_first=True
        )
        self.trans_norm1 = nn.LayerNorm(filters)
        self.trans_norm2 = nn.LayerNorm(filters)
        self.trans_ffn = nn.Sequential(
            nn.Conv1d(filters, filters, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(filters, filters, 1),
        )

        # Cross-attention components
        self.cross_attn1 = nn.MultiheadAttention(
            filters, 1, dropout=dropout, batch_first=True
        )
        self.cross_attn2 = nn.MultiheadAttention(
            filters, 1, dropout=dropout, batch_first=True
        )
        self.cross_norm1 = nn.LayerNorm(filters)
        self.cross_norm2 = nn.LayerNorm(filters)

    def forward(self, tcn_inputs, trans_inputs):
        # TCN path
        tcn_out = self.tcn_conv(tcn_inputs.transpose(1, 2)).transpose(1, 2)
        tcn_out = self.tcn_norm(tcn_out)
        tcn_out = self.tcn_dropout(tcn_out)

        # Transformer path with self-attention
        attn_out, _ = self.self_attention(trans_inputs, trans_inputs, trans_inputs)
        attn_out = self.trans_norm1(attn_out + trans_inputs)

        # Feed-forward network
        ffn_out = self.trans_ffn(attn_out.transpose(1, 2)).transpose(1, 2)
        trans_out = self.trans_norm2(ffn_out + attn_out)

        # Cross-attention: trans_out attends to tcn_out
        cross_out1, _ = self.cross_attn1(trans_out, tcn_out, tcn_out)
        chnl_trans = self.cross_norm1(cross_out1 + trans_out)

        # Cross-attention: tcn_out attends to trans_out
        cross_out2, _ = self.cross_attn2(tcn_out, trans_out, trans_out)
        chnl_tcn = self.cross_norm2(cross_out2 + tcn_out)

        return chnl_tcn, chnl_trans


class CNNEncoder(nn.Module):
    """CNN encoder for time series data."""

    def __init__(self, input_shape, latent_dim, n_filters, k_size, dropout=0.0):
        super().__init__()

        self.seq_len, self.feat_dim = input_shape
        self.latent_dim = latent_dim
        self.n_filters = n_filters
        self.k_size = k_size
        self.dropout = dropout

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.feat_dim

        for f in n_filters:
            # Calculate proper padding for stride=2 convolutions
            padding = (k_size - 1) // 2

            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, f, k_size, stride=2, padding=padding),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = f

        # Calculate flattened dimension
        self.encoder_last_dense_dim = self._calculate_flattened_dim()

        # Latent space projection
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

        # Sampling layer
        self.sampling = Sampling()

    def _calculate_flattened_dim(self):
        """Calculate the flattened dimension after convolutions."""
        with torch.no_grad():
            x = torch.randn(1, self.feat_dim, self.seq_len)
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            return x.numel()

    def forward(self, x):
        # Convert numpy array to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure input has correct shape and add debugging
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")

        batch_size, seq_len, feat_dim = x.shape
        if feat_dim != self.feat_dim:
            raise ValueError(f"Expected {self.feat_dim} features, got {feat_dim}")

        # Use permute instead of transpose for more robust reshaping
        # From [batch, seq_len, feat_dim] to [batch, feat_dim, seq_len] for Conv1d
        x = x.permute(0, 2, 1)

        # Apply convolutions
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten
        x = x.flatten(1)

        # Project to latent space
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        # Sample z
        z = self.sampling(z_mean, z_log_var)

        return z_mean, z_log_var, z


class TimeSformerDecoder(nn.Module):
    """TimeSformer decoder with deconvolution and attention layers."""

    def __init__(
        self,
        latent_dim,
        output_shape,
        hidden_layer_sizes,
        dilations,
        k_size,
        head_size,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_len, self.feat_dim = output_shape
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dilations = dilations
        self.k_size = k_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Calculate initial deconv dimension - this should be the size after all deconvolutions
        # Start with the last hidden layer size and work backwards
        self.initial_dim = hidden_layer_sizes[-1]

        # Initial dense layer
        self.initial_dense = nn.Linear(latent_dim, self.initial_dim)

        # Deconvolution layers
        self.deconv_layers = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]

        # Reverse hidden layer sizes for decoder (excluding the last one)
        reversed_filters = list(reversed(hidden_layer_sizes[:-1]))

        for f in reversed_filters:
            self.deconv_layers.append(
                nn.ConvTranspose1d(
                    in_channels, f, k_size, stride=2, padding=1, output_padding=1
                )
            )
            in_channels = f

        # Final deconv layer
        self.final_deconv = nn.ConvTranspose1d(
            in_channels, self.feat_dim, k_size, stride=2, padding=1, output_padding=1
        )

        # Final dense layers - calculate the actual size after deconvolutions
        # The size after deconvolutions should be seq_len * feat_dim
        self.final_size = self.seq_len * self.feat_dim
        self.final_dense1 = nn.Linear(self.final_size, self.final_size)
        self.final_dense2 = nn.Linear(
            self.final_size * 2, self.final_size
        )  # *2 because we concatenate tcn and trans

        # Projection layers for TimeSformer (feat_dim <-> head_size)
        self.input_projection = nn.Linear(self.feat_dim, head_size)
        self.output_projection = nn.Linear(head_size, self.feat_dim)

        # TimeSformer layers
        self.timesformer_layers = nn.ModuleList(
            [
                TimeSformerLayer(head_size, head_size, num_heads, k_size, d, dropout)
                for d in dilations
            ]
        )

    def forward(self, z):
        # Initial dense projection
        x = F.relu(self.initial_dense(z))
        x = x.view(-1, self.initial_dim, 1)

        # Apply deconvolution layers
        for deconv_layer in self.deconv_layers:
            x = F.relu(deconv_layer(x))

        # Final deconvolution
        x = self.final_deconv(x)

        # Get the actual size after deconvolutions
        actual_size = x.numel() // x.shape[0]  # Flattened size per batch

        # Reshape and apply final dense layers
        x = x.flatten(1)

        # Adjust final dense layer input size to match actual flattened size
        if x.shape[1] != self.final_size:
            # Create a new linear layer with the correct input size
            temp_dense = nn.Linear(x.shape[1], self.final_size).to(x.device)
            x = temp_dense(x)
        else:
            x = self.final_dense1(x)

        res = x.view(-1, self.seq_len, self.feat_dim)

        # Project to head_size for TimeSformer layers
        res_projected = self.input_projection(res)

        # Apply TimeSformer layers
        tcn, trans = res_projected, res_projected
        for timesformer_layer in self.timesformer_layers:
            tcn, trans = timesformer_layer(tcn, trans)

        # Project back to feat_dim
        tcn = self.output_projection(tcn)
        trans = self.output_projection(trans)

        # Concatenate and final projection
        x = torch.cat([tcn, trans], dim=-1)
        x = x.flatten(1)

        # Adjust second dense layer input size
        if x.shape[1] != self.final_size * 2:
            # Create a new linear layer with the correct input size
            temp_dense2 = nn.Linear(x.shape[1], self.final_size).to(x.device)
            x = temp_dense2(x)
        else:
            x = self.final_dense2(x)

        x = x.view(-1, self.seq_len, self.feat_dim)

        return x


class TimeTransformerVAE(nn.Module):
    """Time Transformer VAE model based on the original TensorFlow implementation."""

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        hidden_layer_sizes,
        dilations,
        k_size,
        head_size,
        num_heads,
        dropout,
        reconstruction_wt=3.0,
        **kwargs,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dilations = dilations
        self.k_size = k_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Create encoder and decoder using the original architecture
        self.encoder = CNNEncoder(
            (seq_len, feat_dim), latent_dim, hidden_layer_sizes, k_size, dropout
        )
        self.decoder = TimeSformerDecoder(
            latent_dim,
            (seq_len, feat_dim),
            hidden_layer_sizes,
            dilations,
            k_size,
            head_size,
            num_heads,
            dropout,
        )

        # Sampling layer
        self.sampling = Sampling()

        # Loss tracking
        self.total_loss_tracker = 0.0
        self.reconstruction_loss_tracker = 0.0
        self.kl_loss_tracker = 0.0
        self.step_count = 0

    def forward(self, x):
        """Forward pass through the model."""
        z_mean, z_log_var, z = self.encoder(x)
        x_decoded = self.decoder(z)
        return x_decoded

    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to output."""
        return self.decoder(z)

    def _get_reconstruction_loss(self, x, x_recons):
        # Make sure both are tensors on the same device
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=x_recons.device)
        else:
            x = x.to(device=x_recons.device, dtype=torch.float32)

        err = F.mse_loss(x, x_recons, reduction="sum")

        x_time_mean = torch.mean(x, dim=2)
        x_recons_time_mean = torch.mean(x_recons, dim=2)
        time_err = F.mse_loss(x_time_mean, x_recons_time_mean, reduction="sum")

        return err + time_err

    def train_step(self, x, optimizer):
        """Single training step."""
        optimizer.zero_grad()

        # Ensure x is a torch.Tensor on the correct device/dtype
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device=device, dtype=torch.float32)

        # Encode
        z_mean, z_log_var, z = self.encoder(x)

        # Decode (already returns a tensor)
        reconstruction = self.decoder(z)

        # Calculate losses (both are tensors now)
        reconstruction_loss = self._get_reconstruction_loss(x, reconstruction)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        # Total loss
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Update trackers
        self.total_loss_tracker = total_loss.item()
        self.reconstruction_loss_tracker = reconstruction_loss.item()
        self.kl_loss_tracker = kl_loss.item()
        self.step_count += 1

        return {
            "total_loss": self.total_loss_tracker,
            "reconstruction_loss": self.reconstruction_loss_tracker,
            "kl_loss": self.kl_loss_tracker,
        }

    def sample(self, n_samples):
        """Generate samples from the model."""
        z = torch.randn(
            n_samples, self.latent_dim, device=next(self.parameters()).device
        )
        samples = self.decoder(z)
        return samples

    def get_prior_samples_given_z(self, z):
        """Generate samples given specific latent vectors."""
        return self.decoder(z)

    def fit(
        self, train_data, epochs=1000, batch_size=16, learning_rate=1e-3, verbose=True
    ):
        """Train the model."""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        losses = []
        n_batches = len(train_data) // batch_size

        for epoch in range(epochs):
            epoch_losses = []

            # Shuffle data
            indices = torch.randperm(len(train_data))

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                batch_x = train_data[batch_indices]

                # Training step
                step_losses = self.train_step(batch_x, optimizer)
                epoch_losses.append(step_losses)

            # Calculate average epoch losses
            avg_losses = {
                key: np.mean([step[key] for step in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            losses.append(avg_losses)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Total Loss: {avg_losses['total_loss']:.4f}, "
                    f"Reconstruction Loss: {avg_losses['reconstruction_loss']:.4f}, "
                    f"KL Loss: {avg_losses['kl_loss']:.4f}"
                )

        return losses

    def save(self, filepath):
        """Save the model."""
        # Ensure directory exists
        os.makedirs(filepath, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "seq_len": self.seq_len,
                "feat_dim": self.feat_dim,
                "latent_dim": self.latent_dim,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "dilations": self.dilations,
                "k_size": self.k_size,
                "head_size": self.head_size,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "reconstruction_wt": self.reconstruction_wt,
            },
            os.path.join(filepath, "time_transformer_vae.pth"),
        )

    @classmethod
    def load(cls, filepath):
        """Load the model from file."""
        # Add .pth extension if missing
        if not filepath.endswith(".pth"):
            filepath += ".pth"

        checkpoint = torch.load(filepath, map_location="cpu")

        model = cls(
            seq_len=checkpoint["seq_len"],
            feat_dim=checkpoint["feat_dim"],
            latent_dim=checkpoint["latent_dim"],
            hidden_layer_sizes=checkpoint["hidden_layer_sizes"],
            dilations=checkpoint["dilations"],
            k_size=checkpoint["k_size"],
            head_size=checkpoint["head_size"],
            num_heads=checkpoint["num_heads"],
            dropout=checkpoint["dropout"],
            reconstruction_wt=checkpoint["reconstruction_wt"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model


# Example usage and model creation function
def create_time_transformer_vae(
    seq_len,
    feat_dim,
    latent_dim=64,
    hidden_layer_sizes=[32, 64, 128],
    dilations=[1, 2, 4],
    k_size=4,
    head_size=64,
    num_heads=3,
    dropout=0.2,
    reconstruction_wt=3.0,
):
    """Create a TimeTransformer VAE model with default parameters."""

    return TimeTransformerVAE(
        seq_len=seq_len,
        feat_dim=feat_dim,
        latent_dim=latent_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        dilations=dilations,
        k_size=k_size,
        head_size=head_size,
        num_heads=num_heads,
        dropout=dropout,
        reconstruction_wt=reconstruction_wt,
    )
