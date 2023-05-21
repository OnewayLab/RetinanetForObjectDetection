from torch import nn, Tensor
from typing import List


class FeaturePyramidNet(nn.Module):
    def __init__(
        self, C3_size: int, C4_size: int, C5_size: int, feature_size: int = 256
    ):
        """初始化 FPN

        Args:
            C3_size, C4_size, C5_size: ResNet 中 C3~C5 特征图大小
            feature_size: FPN 特征图通道数
        """
        super(FeaturePyramidNet, self).__init__()

        # 卷积并上采样 C5 得到 P5
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # 卷积 C4 再加上 P5 得到P4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # 卷积 C3 再加上 P4 得到 P3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # 卷积 C5 得到 P6
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # 激活并卷积 P6 得到 P7
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=2, padding=1
        )

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """前向传播

        Args:
            inputs: 输入，ResNet C3~C5 特征图构成的列表

        Returns:
            FPN P3~P7 特征图构成的列表
        """
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsample(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsample(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
