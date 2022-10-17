from unicodedata import bidirectional
from torch import nn
import torch

from hw_asr.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(32, 32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        )

        self.gru = nn.GRU(
            input_size=n_feats * 32 // 2 // 2,
            hidden_size=fc_hidden,
            num_layers=4,
            bidirectional=True,
            batch_first=True,
        )

        self.projection = nn.Linear(fc_hidden * 2, n_class)

    def forward(self, spectrogram, **batch):
        spectrogram = spectrogram.transpose(1, 2).unsqueeze(1)
        feats = self.extractor(spectrogram)

        feats = torch.transpose(feats, 1, 2).reshape((feats.size(0), feats.size(2), -1))

        after_gru = self.gru(feats)[0]
        logits = self.projection(after_gru)

        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths + 1) // 2
