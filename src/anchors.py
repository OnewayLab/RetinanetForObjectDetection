from typing import List, Optional, Tuple
import numpy as np
import torch


class AnchorGenerator:
    def __init__(
        self,
        device: torch.device,
        pyramid_levels: List[int] = [3, 4, 5, 6, 7],
        strides: Optional[List[int]] = None,
        sizes: Optional[List[int]] = None,
        ratios: List[float] = [0.5, 1, 2],
        scales: List[float] = [2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
    ):
        """初始化锚框生成器

        Args:
            device: 生成的锚框张量所处设备
            pyramid_levels: FPN 特征图层级，默认为 P3 到 P7
            strides: 步长，默认根据特征图层级自动计算（8, 16, 32, 64, 128）
            sizes: 锚框的大小，默认根据特征图层级自动计算（32, 64, 128, 256, 512）
            ratios: 锚框的高宽比，默认为 0.5, 1, 2
            scales: 锚框的缩放比，默认为 2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)
        """
        super(AnchorGenerator, self).__init__()
        self.device = device
        self.pyramid_levels = pyramid_levels
        self.strides = strides if strides else [2**x for x in self.pyramid_levels]
        self.sizes = sizes if sizes else [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = ratios
        self.scales = scales

    def generate(self, image_shape: Tuple[int, int]) -> torch.Tensor:
        """根据原始图像生成 FPN 中 P3 到 P7 特征图上的锚框

        Args:
            image_shape: 原始图像的大小，(h, w)

        Returns:
            所有锚框，shape=(B, num_anchors, 4)
        """
        image_shape = np.array(image_shape)
        image_shapes = [
            (image_shape + 2**x - 1) // (2**x) for x in self.pyramid_levels
        ]  # 每个特征图的大小

        # 计算每个金字塔层级的锚框
        all_anchors = np.zeros((0, 4), dtype=np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = self._generate_anchors(
                base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales
            )
            shifted_anchors = self._shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        return torch.from_numpy(all_anchors.astype(np.float32)).to(self.device)

    def _generate_anchors(
        self, base_size: int, ratios: List[int], scales: List[int]
    ) -> np.ndarray:
        """生成一组锚框

        Args:
            base_size: 基础大小
            ratios: 高宽比
            scales: 缩放比

        Returns:
            生成的锚框，shape=(num_anchors, 4)
        """
        num_anchors = len(ratios) * len(scales)
        anchors = np.zeros((num_anchors, 4))

        # 从基础大小开始缩放
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        # 计算每个锚框的面积
        areas = anchors[:, 2] * anchors[:, 3]

        # 根据高宽比调整锚框的大小
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # (x_ctr, y_ctr, w, h) => (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    def _shift(
        self, shape: Tuple[int, int], stride: int, anchors: np.ndarray
    ) -> np.ndarray:
        """将锚框移动到特征图上的每个位置

        Args:
            shape: 特征图的大小，(h, w)
            stride: 步长
            anchors: 锚框，shape=(num_anchors, 4)

        Returns:
            所有移动后的锚框，shape=(K*A, 4)
        """
        # 到每个位置需要移动的距离
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        # 生成网格
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack(
            (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
        ).transpose()

        # 生成所有锚框
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
            (1, 0, 2)
        )
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors


def regressBoxes(
    anchors: torch.Tensor,
    deltas: torch.Tensor,
    mean=[0, 0, 0, 0],
    std=[0.1, 0.1, 0.2, 0.2],
) -> torch.Tensor:
    """根据预测的偏移量对锚框进行调整

    Args:
        anchors: 锚框，shape=(1, num_anchors, 4)
        deltas: 预测的偏移量，shape=(B, num_anchors, 4)
        mean: 均值
        std: 方差

    Returns:
        调整后的锚框，shape=(B, num_anchors, 4)
    """
    # 计算锚框的中心点和宽高
    widths = anchors[:, :, 2] - anchors[:, :, 0]
    heights = anchors[:, :, 3] - anchors[:, :, 1]
    ctr_x = anchors[:, :, 0] + 0.5 * widths
    ctr_y = anchors[:, :, 1] + 0.5 * heights

    # 计算预测的偏移量
    dx = deltas[:, :, 0] * std[0] + mean[0]
    dy = deltas[:, :, 1] * std[1] + mean[1]
    dw = deltas[:, :, 2] * std[2] + mean[2]
    dh = deltas[:, :, 3] * std[3] + mean[3]

    # 计算预测的锚框
    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
    pred_boxes = torch.stack(
        [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2
    )
    return pred_boxes


def clipBoxes(boxes: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    """将预测的锚框裁剪到图像内

    Args:
        boxes: 预测的锚框，shape=(B, num_anchors, 4)
        img_shape: 图像的大小，(h, w)

    Returns:
        裁剪后的锚框，shape=(B, num_anchors, 4)
    """
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0, max=img_shape[1])
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0, max=img_shape[0])
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], min=0, max=img_shape[1])
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], min=0, max=img_shape[0])
    return boxes
