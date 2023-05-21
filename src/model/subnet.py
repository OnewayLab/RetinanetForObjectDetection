from torch import nn, Tensor


class ClassificationSubnet(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        n_anchors: int = 9,
        n_classes: int = 20,
        feature_size: int = 256,
    ):
        """初始化分类子网络

        Args:
            num_features_in: 输入特征图通道数
            n_anchors: 锚框数量
            n_classes: 类别数
            feature_size: 特征图通道数
        """
        super(ClassificationSubnet, self).__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.layers = nn.Sequential(
            nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, n_anchors * n_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        """前向传播

        Args:
            x: 来自 FPN 每一层的特征图，shape=(B, n_channels, W, H)

        Returns:
            输出层特征，shape=(B, W*H*n_anchors, n_classes)
        """
        x = self.layers(x)
        # B x C x W x H => B x W x H x C, where C = n_classes*n_anchors
        x = x.permute(0, 2, 3, 1)
        # B x W x H x C => B x W*H*n_anchors x n_classes
        x = x.reshape(x.shape[0], -1, self.n_classes)
        return x


class BoxRegressionSubnet(nn.Module):
    def __init__(
        self, num_features_in: int, num_anchors: int = 9, feature_size: int = 256
    ):
        """初始化回归子网络

        Args:
            num_features_in: 输入特征图通道数
            n_anchors: 锚框数量
            feature_size: 特征图通道数
        """
        super(BoxRegressionSubnet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """前向传播

        Args:
            x: 来自 FPN 每一层的特征图，shape=(B, n_channels, W, H)

        Returns:
            输出层特征，shape=(B, W*H*n_anchors, 4)
        """
        x = self.layers(x)
        # B x C x W x H => B x W x H x C, where C = 4*n_anchors
        x = x.permute(0, 2, 3, 1)
        # B x W x H x C => B x W*H* n_anchors x 4
        x = x.reshape(x.shape[0], -1, 4)
        return x
