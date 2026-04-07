"""年龄粗分类模型封装。"""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


class AgeGroupClassifier(nn.Module):
    """基于 torchvision backbone 的年龄阶段三分类模型。"""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.5,
        num_classes: int = 3,
    ):
        super().__init__()

        backbone = backbone.lower()
        if backbone == "convnext":
            backbone = "convnext_tiny"

        self.backbone_name = backbone

        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )

        elif backbone in {"efficientnet", "efficientnet_b0"}:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        elif backbone == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        elif backbone == "convnext_tiny":
            self.backbone = models.convnext_tiny(pretrained=pretrained)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        elif backbone == "mobilenet_v3":
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(self.backbone.classifier[0].in_features, 1280),
                nn.Hardswish(),
                nn.Dropout(dropout),
                nn.Linear(1280, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        elif backbone == "regnet":
            self.backbone = models.regnet_y_400mf(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        else:
            raise ValueError(
                "Unknown backbone: {}. Choose from: resnet50, efficientnet_b0, "
                "efficientnet_b1, convnext, mobilenet_v3, regnet".format(backbone)
            )

    def forward(self, x):
        return self.backbone(x)


def get_age_group_model(
    model_name: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.5,
    num_classes: int = 3,
) -> AgeGroupClassifier:
    return AgeGroupClassifier(
        backbone=model_name,
        pretrained=pretrained,
        dropout=dropout,
        num_classes=num_classes,
    )
