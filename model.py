# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class EmbeddingModel(nn.Module):
    def __init__(self, backbone="resnet50", embedding_dim=256, pretrained=True):
        super().__init__()
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier

        # Trainable head
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.head(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
