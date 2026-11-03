"""One pretrained feature extractor and three independent binary heads."""

import timm
import torch
from torch import nn

MODEL_ID = "timm/mobilenetv3_small_100.lamb_in1k"


class SignClassifier(nn.Module):
    def __init__(self, pretrained=True, revision=None):
        super().__init__()
        source = f"hf-hub:{MODEL_ID}" + (f"@{revision}" if revision else "")
        if not pretrained:
            source = "mobilenetv3_small_100"
        self.backbone = timm.create_model(source,
                                          pretrained=pretrained, num_classes=0)
        self.heads = nn.ModuleList([nn.Linear(self.backbone.head_hidden_size, 1)
                                    for _ in range(3)])

    def forward(self, images):
        features = self.backbone(images)
        return torch.cat([head(features) for head in self.heads], dim=1)
