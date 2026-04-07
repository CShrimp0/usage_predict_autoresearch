"""
年龄预测模型架构
"""
import torch
import torch.nn as nn
import torchvision.models as models


class AgePredictor(nn.Module):
    """基于ResNet50的年龄预测模型"""
    
    def __init__(self, pretrained=True, dropout=0.5):
        """
        Args:
            pretrained: 是否使用ImageNet预训练权重
            dropout: Dropout比例
        """
        super(AgePredictor, self).__init__()
        
        # 加载预训练的ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 获取最后全连接层的输入特征数
        num_features = self.backbone.fc.in_features
        
        # 替换最后的全连接层为回归头
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # 输出单个值（年龄）
        )
        
    def forward(self, x):
        """
        Args:
            x: 输入图像张量 [batch_size, 3, 224, 224]
        
        Returns:
            预测年龄 [batch_size, 1]
        """
        return self.backbone(x).squeeze(-1)  # 移除最后的维度


class EfficientNetAgePredictor(nn.Module):
    """基于EfficientNet-B0的年龄预测模型（更轻量）"""
    
    def __init__(self, pretrained=True, dropout=0.5):
        super(EfficientNetAgePredictor, self).__init__()
        
        # 加载预训练的EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # 获取最后全连接层的输入特征数
        num_features = self.backbone.classifier[1].in_features
        
        # 替换分类器为回归头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class ConvNeXtAgePredictor(nn.Module):
    """基于ConvNeXt-Tiny的年龄预测模型（2022最新CNN架构）"""
    
    def __init__(self, pretrained=True, dropout=0.5):
        super(ConvNeXtAgePredictor, self).__init__()
        
        # 加载预训练的ConvNeXt-Tiny
        try:
            from torchvision.models import convnext_tiny
            self.backbone = convnext_tiny(pretrained=pretrained)
        except ImportError:
            # 兼容旧版本
            self.backbone = models.convnext_tiny(pretrained=pretrained)
        
        # 获取分类器输入特征数
        num_features = self.backbone.classifier[2].in_features
        
        # 替换分类器为回归头
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class EfficientNetB1AgePredictor(nn.Module):
    """基于EfficientNet-B1的年龄预测模型（B0升级版）"""
    
    def __init__(self, pretrained=True, dropout=0.5):
        super(EfficientNetB1AgePredictor, self).__init__()
        
        self.backbone = models.efficientnet_b1(pretrained=pretrained)
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class MobileNetV3AgePredictor(nn.Module):
    """基于MobileNetV3-Large的年龄预测模型（移动端优化）"""
    
    def __init__(self, pretrained=True, dropout=0.5):
        super(MobileNetV3AgePredictor, self).__init__()
        
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        num_features = self.backbone.classifier[3].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class RegNetAgePredictor(nn.Module):
    """基于RegNetY-002的年龄预测模型（高效架构）"""
    
    def __init__(self, pretrained=True, dropout=0.5):
        super(RegNetAgePredictor, self).__init__()
        
        self.backbone = models.regnet_y_400mf(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class FlexibleMultimodalModel(nn.Module):
    """
    灵活的多模态模型，支持动态辅助特征维度
    
    架构：
        图像分支：CNN backbone (ResNet/EfficientNet等)
        辅助分支：全连接层
        融合层：拼接后预测
    """
    
    def __init__(self, 
                 backbone='resnet50',
                 pretrained=True,
                 aux_input_dim=0,      # 辅助特征维度（0表示纯图像模型）
                 aux_hidden_dim=32,    # 辅助分支隐藏层维度
                 dropout=0.5):
        """
        Args:
            backbone: CNN骨干网络名称
            pretrained: 是否使用预训练权重
            aux_input_dim: 辅助特征维度（动态计算）
            aux_hidden_dim: 辅助分支隐藏层大小
            dropout: Dropout比例
        """
        super().__init__()
        
        self.aux_input_dim = aux_input_dim
        self.backbone_name = backbone
        
        # ========== 图像分支 ==========
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Identity()  # 移除最后的FC层
            img_feat_dim = 2048
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone.fc = nn.Identity()
            img_feat_dim = 512
        elif backbone == 'efficientnet_b0' or backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier = nn.Identity()
            img_feat_dim = 1280
        else:
            # 默认使用ResNet50
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Identity()
            img_feat_dim = 2048
        
        # ========== 辅助特征分支 ==========
        if aux_input_dim > 0:
            self.aux_branch = nn.Sequential(
                nn.Linear(aux_input_dim, aux_hidden_dim),
                nn.BatchNorm1d(aux_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.6),  # 辅助分支用较小dropout
                nn.Linear(aux_hidden_dim, aux_hidden_dim),
                nn.BatchNorm1d(aux_hidden_dim),
                nn.ReLU(inplace=True)
            )
            fusion_dim = img_feat_dim + aux_hidden_dim
        else:
            self.aux_branch = None
            fusion_dim = img_feat_dim
        
        # ========== 融合预测头 ==========
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, aux_features=None):
        """
        前向传播
        
        Args:
            image: (B, 3, H, W) 图像张量
            aux_features: (B, aux_input_dim) 辅助特征，可选
        
        Returns:
            (B,) 年龄预测值
        """
        # 提取图像特征
        img_feat = self.backbone(image)  # (B, img_feat_dim)
        
        # 如果有辅助特征，进行融合
        if self.aux_input_dim > 0 and aux_features is not None:
            aux_feat = self.aux_branch(aux_features)  # (B, aux_hidden_dim)
            fused = torch.cat([img_feat, aux_feat], dim=1)  # (B, fusion_dim)
        else:
            fused = img_feat
        
        # 预测年龄
        output = self.fusion_head(fused).squeeze(-1)  # (B,)
        return output


def get_model(model_name='resnet50', pretrained=True, dropout=0.5, 
              aux_input_dim=0, aux_hidden_dim=32):
    """
    获取模型（支持多模态）
    
    Args:
        model_name: 模型名称 ('resnet50', 'efficientnet', 'convnext_tiny', 
                    'efficientnet_b1', 'mobilenet_v3', 'regnet')
        pretrained: 是否使用预训练权重
        dropout: Dropout比例
        aux_input_dim: 辅助特征维度（0表示纯图像模型）
        aux_hidden_dim: 辅助特征分支隐藏层维度
    
    Returns:
        模型实例
    """
    # 如果启用多模态，使用FlexibleMultimodalModel
    if aux_input_dim > 0:
        return FlexibleMultimodalModel(
            backbone=model_name,
            pretrained=pretrained,
            aux_input_dim=aux_input_dim,
            aux_hidden_dim=aux_hidden_dim,
            dropout=dropout
        )
    
    # 否则使用原始模型
    if model_name == 'resnet50':
        return AgePredictor(pretrained=pretrained, dropout=dropout)
    elif model_name == 'efficientnet' or model_name == 'efficientnet_b0':
        return EfficientNetAgePredictor(pretrained=pretrained, dropout=dropout)
    elif model_name == 'convnext_tiny':
        return ConvNeXtAgePredictor(pretrained=pretrained, dropout=dropout)
    elif model_name == 'efficientnet_b1':
        return EfficientNetB1AgePredictor(pretrained=pretrained, dropout=dropout)
    elif model_name == 'mobilenet_v3':
        return MobileNetV3AgePredictor(pretrained=pretrained, dropout=dropout)
    elif model_name == 'regnet':
        return RegNetAgePredictor(pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: resnet50, efficientnet, "
                        f"convnext_tiny, efficientnet_b1, mobilenet_v3, regnet")


if __name__ == '__main__':
    # 测试模型
    model = get_model('resnet50', pretrained=False)
    
    # 创建随机输入
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例: {output}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
