import torch
from torch import nn

from .base_model import BaseModel


class MCBiGRU(BaseModel):
    def __init__(
        self,
        embed_dim: int,
        rnn_hidden_size: int = 10,
        conv_config: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if conv_config is None:
            conv_config = {"num_channels": 128, "kernel_sizes": [1, 2, 3, 5, 6]}

        self.conv_config = conv_config

        # =======================
        # PARALLEL CNN EXTRACTORS
        # =======================
        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        conv_config["num_channels"],
                        kernel_size=kernel,
                    ),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    # Squashes the sentence down to 4 key chronological time-steps
                    nn.AdaptiveMaxPool1d((4,)),
                )
                for kernel in conv_config["kernel_sizes"]
            ]
        )

        # =======================
        # PARALLEL BiGRUs
        # =======================
        self.rnns = nn.ModuleList(
            [
                nn.GRU(
                    input_size=conv_config["num_channels"],
                    hidden_size=rnn_hidden_size,
                    num_layers=10,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.6,
                )
                for _ in conv_config["kernel_sizes"]
            ]
        )

        # =======================
        # CLASSIFIER HEAD
        # =======================
        self.fc1 = nn.Linear(
            rnn_hidden_size * 2 * len(conv_config["kernel_sizes"]),
            self.target_classes_len,
        )
        self.batch_nn = nn.BatchNorm1d(self.target_classes_len)

    def forward(self, text):
        # Conv1d expects shape: [batch_size, embed_dim, seq_len]
        reshaped_cnn_in = torch.permute(text, (0, 2, 1))

        # Run through CNNs
        # Output of each conv: [batch_size, num_channels, 4]
        cnn_out = [conv(reshaped_cnn_in) for conv in self.convolutions]

        # New shape: [batch_size, 4, num_channels]
        cnn_out = [out_.permute(0, 2, 1) for out_ in cnn_out]

        # Run each CNN output through its dedicated BiGRU
        # rnn(out_) returns (output, hidden). Grab the output [batch, 4, hidden*2]
        # [:, -1] grabs the final time step representation
        bigru_out = [self.rnns[idx](out_)[0][:, -1] for idx, out_ in enumerate(cnn_out)]

        # Concatenate all the BiGRU final states together
        concat_out = torch.cat(bigru_out, dim=1)

        # Pass to fully connected layer
        fc_out = self.fc1(concat_out)

        # Normalize logits before BCEWithLogitsLoss
        batch_n = self.batch_nn(fc_out)

        return batch_n
