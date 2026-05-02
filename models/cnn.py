import torch
from torch import nn

from .base_model import BaseModel


class CNN(BaseModel):
    def __init__(
        self,
        embed_dim: int,
        conv_config: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if conv_config is None:
            conv_config = {"num_channels": 50, "kernel_sizes": [1, 2, 3, 4, 5, 6]}

        self.conv_config = conv_config

        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        self.conv_config["num_channels"],
                        kernel_size=kernel,
                    ),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d((1,)),
                )
                for kernel in self.conv_config["kernel_sizes"]
            ]
        )

        self.fc1 = nn.Linear(
            len(self.conv_config["kernel_sizes"]) * self.conv_config["num_channels"],
            self.target_classes_len,
        )

    def forward(self, text):
        # CNN expects [batch_size, embed_dim, sequence_length]
        reshaped_cnn_in = torch.permute(text, (0, 2, 1))

        # Apply parallel convolutions, squeeze out the 3rd dimension, and store in a list
        cnn_out = [conv(reshaped_cnn_in).squeeze(2) for conv in self.convolutions]

        # Concatenate all features from the different n-gram kernels
        concat_out = torch.cat(cnn_out, dim=1)

        # Pass through Fully Connected layer
        fc_out = self.fc1(concat_out)

        return fc_out
