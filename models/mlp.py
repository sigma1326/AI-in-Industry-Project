from pprint import pprint

import torch
from torch import nn

from models import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        embed_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        compression = 5

        def get_layers():
            # Start by flattening the [batch_size, max_tokens, embed_dim] GloVe tensor
            layers = [
                nn.Flatten(),
            ]

            # Dynamically build the shrinking linear layers
            current_size = self.maximum_tokens * embed_dim

            while current_size // compression > self.target_classes_len * compression:
                next_size = current_size // compression

                layers.append(nn.Linear(current_size, next_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))

                current_size = next_size

            # Final output layer
            layers.append(nn.Linear(current_size, self.target_classes_len))
            return layers

        self.fc_layers = nn.Sequential(*get_layers())

    def forward(self, text):
        # text shape: [batch_size, maximum_tokens, embed_dim]
        # It goes straight into the Flatten layer, no Embedding needed
        return self.fc_layers(text)


if __name__ == "__main__":
    from pprint import pprint

    # Initialize the MLP
    mlp = MLP(
        embed_dim=100,
        optimizer_type="adamW",
        learning_rate=0.001,
        maximum_tokens=50,
    ).to("cuda")

    pprint(mlp)

    # [Batch_Size, Maximum_Tokens, Embed_Dim]
    dummy_glove_batch = torch.randn(256, 50, 100).to("cuda")

    output = mlp(dummy_glove_batch)

    print("\nSuccess! Final Output Shape:", output.shape)
    # torch.Size([256, 6])
