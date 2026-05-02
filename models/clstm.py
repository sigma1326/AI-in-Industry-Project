import torch
from torch import nn
from models import BaseModel


class CLSTM(BaseModel):
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
                        padding="same",  # Keeps sequence length fixed
                    ),
                    nn.ReLU(),
                )
                for kernel in self.conv_config["kernel_sizes"]
            ]
        )

        # The total number of features coming out of the CNNs per word
        cnn_out_features = self.conv_config["num_channels"] * len(
            self.conv_config["kernel_sizes"]
        )
        rnn_hidden_size = cnn_out_features

        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            # dropout=0.2, # does not converge with dropout enabled
        )

        self.fc1 = nn.Linear(rnn_hidden_size, int(rnn_hidden_size / 2))
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(int(rnn_hidden_size / 2), self.target_classes_len)

    def forward(self, text):
        # CNN Phase
        # text is [batch, 50_tokens, 300_embed]
        reshaped_cnn_in = torch.permute(text, (0, 2, 1))  # [batch, 300, 50]

        cnn_out = [conv(reshaped_cnn_in) for conv in self.convolutions]

        # Concatenate all the kernel outputs together
        concat_out = torch.cat(cnn_out, dim=1)  # Shape: [batch, cnn_out_features, 50]

        # LSTMs expect [batch_size, sequence_length, features]
        lstm_in = torch.permute(
            concat_out, (0, 2, 1)
        )  # Shape: [batch, 50, cnn_out_features]

        # LSTM Phase
        lstm_out, (hidden, _) = self.lstm(lstm_in)
        lstm_out = lstm_out[:, -1]  # Grab the very last timestep's output

        # Fully Connected Phase
        fc_out = self.fc1(lstm_out)
        fc_out = self.tanh(fc_out)
        fc_out = self.fc2(fc_out)

        return fc_out


if __name__ == "__main__":
    clstm = CLSTM(
        embed_dim=300,
        conv_config={"num_channels": 50, "kernel_sizes": [1, 2, 3, 5]},
        optimizer_type="adamW",
        learning_rate=0.001,
        maximum_tokens=25,
    )

    print(clstm(torch.rand(256, 25, 300)).shape)
