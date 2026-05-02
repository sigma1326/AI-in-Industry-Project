from pprint import pprint

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel

from models import BaseModel


class BERT(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            self.target_classes_len,
        )

    def forward(self, text):
        text = pad_sequence(text, batch_first=True, padding_value=0)

        mask = (text != 0).float()
        outputs = self.bert(
            text,
            attention_mask=mask,
        )
        cls_output = outputs[1]  # batch, hidden
        cls_output = self.classifier(cls_output)  # batch, 6

        return cls_output


if __name__ == "__main__":
    mlp = BERT(
        optimizer_type="adam",
        learning_rate=0.001,
        maximum_tokens=50,
    ).to("cuda")
    pprint(mlp)

    mlp(torch.randint(0, 291000, (256, 50)).to("cuda"))
