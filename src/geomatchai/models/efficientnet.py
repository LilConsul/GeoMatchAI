import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_variant="b4"):
        super().__init__()
        if model_variant == "b4":
            self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        # Remove classifier, keep features
        self.model.classifier = nn.Identity()
        self.feature_dim = 1792  # B4 output dimension

    def forward(self, x):
        return self.model(x)
