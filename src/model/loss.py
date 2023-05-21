import torch
from torch import nn, Tensor
from ..metrics import calculate_iou


class FocalLoss(nn.Module):
    def __init__(self, device: torch.device, gamma: float = 2.0, alpha: float = 0.25):
        """初始化 Focal Loss

        Args:
            device: 训练设备
            gamma, alpha: 超参数
        """
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        classifications: Tensor,
        regressions: Tensor,
        anchors: Tensor,
        annotations: Tensor,
    ):
        """前向传播

        Args:
            classifications: 分类子网络的输出特征，shape=(B, N, C)
            regressions: 回归子网络的输出特征，shape=(B, N, 4)
            anchors: 生成的锚框，shape=(1, N, 4)
            annotations: 标注框，shape=(B, M, 5)

        Returns:
            分类损失和回归损失均值之和
        """
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        # 获取锚框的中心和宽高
        anchor = anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        # 遍历每一张图片
        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]  # 去掉填充的标注框
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:  # 如果没有标注框，那么所有锚框都是负类
                alpha_factor = torch.full(
                    classification.shape, 1.0 - self.alpha, device=self.device
                )
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
                bce = -torch.log(1.0 - classification)
                cls_loss = focal_weight * bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().cuda())
                continue

            IoU = calculate_iou(
                anchors[0, :, :], bbox_annotation[:, :4]
            )  # shape=(num_anchors, num_annotations)

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # 计算分类损失
            targets = torch.full(
                classification.shape, -1, dtype=torch.float, device=self.device
            )
            targets[IoU_max < 0.4, :] = 0  # 背景类
            positive_indices = IoU_max > 0.5  # 正样本
            num_positive_anchors = positive_indices.sum()  # 正样本数量
            assigned_annotations = bbox_annotation[IoU_argmax, :]  # 分配到各个锚框的标注框

            targets[positive_indices, :] = 0
            targets[
                positive_indices, assigned_annotations[positive_indices, 4].long()
            ] = 1  # 分配的标注框对应的类别为正类

            alpha_factor = torch.full(targets.shape, self.alpha, device=self.device)
            alpha_factor = torch.where(targets == 1.0, alpha_factor, 1.0 - alpha_factor)
            focal_weight = torch.where(
                targets == 1.0, 1.0 - classification, classification
            )
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(
                targets * torch.log(classification)
                + (1.0 - targets) * torch.log(1.0 - classification)
            )

            cls_loss = focal_weight * bce
            cls_loss[targets == -1.0] = 0
            classification_losses.append(
                cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
            )

            # 计算回归损失
            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()
                targets = targets / torch.tensor(
                    [[0.1, 0.1, 0.2, 0.2]], device=self.device
                )

                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    regression_diff < 1.0 / 9.0,
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0,
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(
                    torch.tensor(0, dtype=torch.float, device=self.device)
                )

        # 计算均值
        classification_loss = torch.stack(classification_losses)
        regression_loss = torch.stack(regression_losses)
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        return loss
