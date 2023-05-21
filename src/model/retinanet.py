from itertools import chain
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torchvision.ops import nms
from ..anchors import AnchorGenerator, regressBoxes, clipBoxes
from .loss import FocalLoss
from .fpn import FeaturePyramidNet
from .subnet import ClassificationSubnet, BoxRegressionSubnet
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class RetinaNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        device: torch.device,
        backbone: str = "ResNet50",
        prior: float = 0.01,
        pretrained: bool = True,
        model_path: str = "./model"
    ):
        """初始化 RetinaNet

        Args:
            n_classes: 类别数
            device: 设备
            backbone: 特征提取网络，从 {"ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"} 中选择
            prior: 先验概率，用于初始化分类器输出层
            pretrained: 是否使用预训练模型
            model_path: 预训练模型路径
        """
        super(RetinaNet, self).__init__()
        self.n_classes = n_classes
        self.device = device

        # ResNet
        if backbone == "ResNet18":
            self.resnet = resnet18(n_classes, pretrained, model_path)
            fpn_sizes = [128, 256, 512]
        elif backbone == "ResNet34":
            self.resnet = resnet34(n_classes, pretrained, model_path)
            fpn_sizes = [128, 256, 512]
        elif backbone == "ResNet50":
            self.resnet = resnet50(n_classes, pretrained, model_path)
            fpn_sizes = [512, 1024, 2048]
        elif backbone == "ResNet101":
            self.resnet = resnet101(n_classes, pretrained, model_path)
            fpn_sizes = [512, 1024, 2048]
        elif backbone == "ResNet152":
            self.resnet = resnet152(n_classes, pretrained, model_path)
            fpn_sizes = [512, 1024, 2048]
        else:
            raise NotImplementedError(
                "backbone 只支持 ResNet18, ResNet34, ResNet50, ResNet101, ResNet152"
            )

        # FPN
        self.fpn = FeaturePyramidNet(*fpn_sizes)

        # class+box subnets
        self.classification_subnet = ClassificationSubnet(256, n_classes=n_classes)
        self.regression_subnet = BoxRegressionSubnet(256)

        self.anchor_generator = AnchorGenerator(device)

        self.loss = FocalLoss(device=self.device)

        # 初始化权重
        for m in chain(
            self.fpn.modules(),
            self.classification_subnet.modules(),
            self.regression_subnet.modules(),
        ):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.classification_subnet.layers[-2].weight.data.fill_(0)
        self.classification_subnet.layers[-2].bias.data.fill_(
            -math.log((1.0 - prior) / prior)
        )
        self.regression_subnet.layers[-1].weight.data.fill_(0)
        self.regression_subnet.layers[-1].bias.data.fill_(0)

        self.to(device)

    def freeze_resnet(self):
        """冻结 ResNet 参数"""
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_resnet(self):
        """解冻 ResNet 参数"""
        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(
        self, images: Tensor, annotations: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """前向传播

        Args:
            images: 输入图像，shape=(B, C, H, W)
            annotations: 标注框，shape=(B, N, 5)，其中 N 为标注框个数，5 为 (x1, y1, x2, y2, class)

        Returns:
            如果提供了标注，则返回输出层特征和损失，否则只返回输出层特征；
        """
        # ResNet
        features = self.resnet(images)

        # FPN
        features = self.fpn(features)

        # 每个层级的 FPN 特征图共享相同的分类和回归子网络
        regressions = torch.cat(
            [self.regression_subnet(feature) for feature in features], dim=1
        )  # shape=(B, N, 4)
        classifications = torch.cat(
            [self.classification_subnet(feature) for feature in features], dim=1
        )  # shape=(B, N, n_classes)

        # 生成锚框
        anchors = self.anchor_generator.generate(images.shape[-2:])  # shape=(1, N, 4)
        result = {
            "anchors": anchors,
            "regressions": regressions,
            "classifications": classifications,
        }

        # 计算损失
        if annotations is not None:
            loss = self.loss(classifications, regressions, anchors, annotations)
            result["loss"] = loss
        return result

    def infer(
        self,
        images: Tensor,
        h_scales: Optional[Tensor] = None,
        w_scales: Optional[Tensor] = None,
        threshold: float = 0.5,
    ) -> Union[Tensor, List[Tensor]]:
        """推理

        Args:
            image: 输入图像，可以是单张图像（shape=(C, H, W)）或一个批量（shape=(B, C, H, W)）
            h_scale, w_scale: 如果提供了，则用它们对边界框的高和宽进行缩放
            threshold: 预测结果的置信度阈值

        Returns:
            图像的预测结果 shape=(N, 6)，6 为 (x1, y1, x2, y2, class, score)；
            如果输入一个批量，则返回上述预测结果的一个列表
        """
        self.eval()
        with torch.no_grad():
            # 如果是单张图片，先增加一个维度
            if images.ndim == 3:
                images = images.unsqueeze(0)

            # 前向传递
            output = self.forward(images)

            # 后处理
            result = self.post_process(
                output, images.shape[-2:], h_scales, w_scales, threshold
            )

            # 如果是单张图片，直接返回张量，否则返回列表
            if images.ndim == 3:
                return result[0]
            else:
                return result

    def post_process(
        self,
        output: Dict[str, Tensor],
        input_size: Tuple[int, int],
        h_scales: Optional[Tensor] = None,
        w_scales: Optional[Tensor] = None,
        threshold: float = 0.05,
    ) -> Union[Tensor, List[Tensor]]:
        """对模型前向传递的输出进行后处理，得到预测结果

        Args:
            output: 模型前向传递的输出，包含 anchors, regressions, classifications
            input_size: 输入图像的高和宽
            h_scale, w_scale: 如果提供了，则用它们对边界框的高和宽进行缩放
            threshold: 预测结果的置信度阈值

        Returns:
            图像的预测结果 shape=(N, 6)，6 为 (x1, y1, x2, y2, class, score)；
            如果输入一个批量，则返回上述预测结果的一个列表
        """
        anchors = output["anchors"]  # shape=(B, N, 4)
        regressions = output["regressions"]  # shape=(B, N, 4)
        classifications = output["classifications"]  # shape=(B, N, n_classes)

        # 根据回归网络输出调整锚框
        anchors = regressBoxes(anchors, regressions)  # shape=(B, N, 4)
        anchors = clipBoxes(anchors, input_size)  # shape=(B, N, 4)

        # 对每张图像进行处理
        result_per_img = []  # 每张图像的结果
        for i in range(anchors.shape[0]):
            img_anchors, img_classifications = anchors[i], classifications[i]
            result_per_class = []  # 这张图像每个类别的结果
            # 对每个类别进行处理
            for j in range(img_classifications.shape[1]):
                scores = img_classifications[:, j]
                # 选取置信度大于阈值的锚框
                scores_over_thresh = scores > threshold
                if scores_over_thresh.any():  # 如果没有，就跳过
                    scores = scores[scores_over_thresh]
                    class_anchors = img_anchors[scores_over_thresh]
                    # 用非极大抑制算法去除重叠度高的框
                    indices = nms(class_anchors, scores, 0.5)
                    class_anchors = class_anchors[indices]  # shape=(N,4)
                    classes = torch.tensor([j] * indices.shape[0], device = self.device)
                    classes = classes.unsqueeze(1)  # shape=(N, 1)
                    scores = scores[indices].unsqueeze(1)  # shape=(N, 1)
                    # 用 h_scale 和 w_scale 对边界框进行缩放
                    if h_scales is not None:
                        class_anchors[:, [1, 3]] *= h_scales[i]
                    if w_scales is not None:
                        class_anchors[:, [0, 2]] *= w_scales[i]
                    # 拼接得到结果
                    result = torch.cat(
                        (class_anchors, classes, scores), dim=1
                    )  # shape=(N, 6)
                    result_per_class.append(result)
            if len(result_per_class) == 0:  # 如果这张图像没有检测到任何物体，就添加一个空的张量
                result_per_class = torch.zeros((0, 6), device=self.device)
            else:  # 否则就拼接得到这张图像的结果
                result_per_class = torch.cat(result_per_class, dim=0)  # shape=(N, 6)
            result_per_img.append(result_per_class)

        return result_per_img
