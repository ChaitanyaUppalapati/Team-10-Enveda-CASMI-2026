"""Notebook-compatible networks; layer names preserve imported checkpoint keys."""

import torch
from torch import nn


class CNN1DFingerprint(nn.Module):
    def __init__(self, fp_size=1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, fp_size),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MultiResolutionCNNFingerprint(nn.Module):
    def __init__(self, fp_size=1024):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(1, 16, k, padding=k // 2), nn.BatchNorm1d(16), nn.ReLU()) for k in (3, 7, 15)]
        )
        self.features = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, fp_size),
        )

    def forward(self, x):
        return self.classifier(self.features(torch.cat([branch(x) for branch in self.branches], dim=1)))


def build_model(architecture, fp_size):
    classes = {"cnn1d": CNN1DFingerprint, "multires_cnn": MultiResolutionCNNFingerprint}
    if architecture not in classes:
        raise ValueError(f"Unknown CNN architecture: {architecture}")
    return classes[architecture](fp_size)
