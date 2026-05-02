import torch
from torch import nn

from .base_model import BaseModel


class CNNBiGRU(BaseModel):
    def __init__(
        self,
        embed_dim: int,
        conv_config: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if conv_config is None:
            conv_config = {"num_channels": 50, "kernel_sizes": [1, 2, 3]}

        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        conv_config["num_channels"],
                        kernel_size=kernel,
                    ),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d((1,)),
                )
                for kernel in conv_config["kernel_sizes"]
            ]
        )

        rnn_hidden_size = int(
            conv_config["num_channels"] * len(conv_config["kernel_sizes"])
        )

        self.bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=rnn_hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )

        self.fc1 = nn.Linear(rnn_hidden_size, self.target_classes_len)

    def forward(self, text):
        # =======================
        # PATH 1: CNN
        # =======================
        # Conv1d expects shape: [batch_size, embed_dim, seq_len]
        reshaped_cnn_in = torch.permute(text, (0, 2, 1))

        # Squeeze out the sequence dimension left by AdaptiveMaxPool1d
        cnn_out = [conv(reshaped_cnn_in).squeeze(2) for conv in self.convolutions]
        concat_out = torch.cat(cnn_out, dim=1)  # Shape: [batch_size, rnn_hidden_size]

        # =======================
        # PATH 2: BiGRU
        # =======================
        _, hidden = self.bigru(text)

        # Grab the final hidden states from the forward and backward directions
        # hidden shape: [num_layers * 2, batch_size, hidden_size]
        bigru_out = (
            hidden[-2, :] + hidden[-1, :]
        )  # Shape: [batch_size, rnn_hidden_size]

        # =======================
        # Merge
        # =======================
        # Calculate the mean of the BiGRU features FOR EACH COMMENT individually
        # Shape becomes [batch_size, 1] to prevent cross-batch contamination
        bigru_out_mean = torch.mean(bigru_out, dim=1, keepdim=True)

        # Subtract that specific comment's BiGRU mean from its CNN features
        concat_out = concat_out - bigru_out_mean

        # Element-wise multiply the shifted CNN features with the original BiGRU features
        fc_in = concat_out * bigru_out

        # =======================
        # PREDICTION
        # =======================
        fc_out = self.fc1(fc_in)

        return fc_out
