"""
Multimodal fusion model: CNN image encoder (ResNet18 backbone) + tabular MLP
encoder, concatenated and fed through a regression head to predict log-price.
"""
import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """ResNet18 backbone (optionally ImageNet-pretrained) as a fixed-length feature extractor."""

    def __init__(self, embedding_dim=128, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()  # remove classification head -> 512-d features
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.projection(feats)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionRegressor(nn.Module):
    """Concatenates image + tabular embeddings and regresses log(price)."""

    def __init__(self, tabular_input_dim, image_embed_dim=128, tabular_embed_dim=32,
                 pretrained_cnn=True, freeze_backbone=True):
        super().__init__()
        self.image_encoder = ImageEncoder(image_embed_dim, pretrained=pretrained_cnn,
                                           freeze_backbone=freeze_backbone)
        self.tabular_encoder = TabularEncoder(tabular_input_dim, tabular_embed_dim)

        fused_dim = image_embed_dim + tabular_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, image, tabular):
        img_emb = self.image_encoder(image)
        tab_emb = self.tabular_encoder(tabular)
        fused = torch.cat([img_emb, tab_emb], dim=1)
        return self.head(fused).squeeze(-1)  # predicts log1p(price)
