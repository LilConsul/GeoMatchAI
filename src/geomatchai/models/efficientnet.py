import torch.nn as nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_variant="b4"):
        super().__init__()
        if model_variant == "b4":
            self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

            # Feature dimension for B4 is 1792
            self.feature_dim = 1792

        # Remove classifier, keep features
        self.model.classifier = nn.Identity()

        # Set to evaluation mode for consistent inference
        self.model.eval()

        # Freeze model parameters to prevent memory leaks during inference
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
