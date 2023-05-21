from typing import List
import torch
from torch import Tensor


def calculate_iou(a: Tensor, b: Tensor) -> Tensor:
    """计算两组框的 IoU

    Args:
        a: shape: (N, 4)，每一行的格式为 (xmin, ymin, xmax, ymax)
        b: shape: (M, 4)，每一行的格式为 (xmin, ymin, xmax, ymax)

    Returns:
        IoU，shape: (N, M)
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(
        torch.unsqueeze(a[:, 0], 1), b[:, 0]
    )
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(
        torch.unsqueeze(a[:, 1], 1), b[:, 1]
    )
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = (
        torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
        + area
        - iw * ih
    )
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU


def calculate_ap(
    predictions: List[Tensor], annotations: List[Tensor], n_classes: int
) -> float:
    """计算平均准确率（AP）

    Args:
        predictions: 预测结果，每个张量 shape=(n_boxes, 6)，6 为 (xmin, ymin, xmax, ymax, class, score)
        annotations: 真实标注，每个张量 shape=(m_boxes, 5)，5 为 (xmin, ymin, xmax, ymax, class)
        n_classes: 类别数

    Returns:
        AP
    """
    average_precisions = torch.zeros(n_classes)

    # 分别计算每个类别
    for label in range(n_classes):
        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0

        for prediction, annotation in zip(predictions, annotations):
            prediction = prediction[prediction[:, 4] == label]
            annotation = annotation[annotation[:, 4] == label]
            num_annotations += annotation.shape[0]
            scores.append(prediction[:, 5].cpu())
            if annotation.shape[0] == 0:  # 如果没有标注框，那么所有预测框都是假阳性
                false_positives.append(torch.ones(prediction.shape[0]))
                true_positives.append(torch.zeros(prediction.shape[0]))
            else:  # 如果有标注框，那么遍历所有预测框，给它们找配对的标注框
                ious = calculate_iou(prediction, annotation)  # 计算每个预测框与每个标注框的 IoU
                max_ious, indices = ious.max(dim=1)  # 每个预测框对应的最高 IoU 和下标
                max_ious = max_ious.cpu()
                indices = indices.cpu()
                detected_annotations = set()
                fp = torch.zeros(prediction.shape[0])
                tp = torch.zeros(prediction.shape[0])
                for i in range(prediction.shape[0]):
                    max_iou, idx = max_ious[i], indices[i].item()
                    if max_iou >= 0.5 and idx not in detected_annotations:  # 真阳性
                        fp[i] = 0
                        tp[i] = 1
                        detected_annotations.add(idx)
                    else:  # 假阳性
                        fp[i] = 1
                        tp[i] = 0
                false_positives.append(fp)
                true_positives.append(tp)

        # 如果整个类别都没有标注，那么 AP 为 0
        if num_annotations == 0:
            average_precisions[label] = 0
        else:
            false_positives = torch.cat(false_positives, dim=0)
            true_positives = torch.cat(true_positives, dim=0)
            scores = torch.cat(scores, dim=0)
            # 按置信度从大到小排序
            indices = torch.argsort(scores, descending=True)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]
            # 累加
            false_positives = torch.cumsum(false_positives, dim=0)
            true_positives = torch.cumsum(true_positives, dim=0)
            # 计算召回率和准确率
            recall = true_positives / num_annotations
            precision = true_positives / torch.clamp(
                true_positives + false_positives, min=1e-16
            )
            # 计算 AP
            average_precision = _calculate_ap(recall, precision)
            average_precisions[label] = average_precision

    return average_precisions


def _calculate_ap(recalls: Tensor, precisions: Tensor):
    """计算平均准确率（AP）

    Args:
        recall: 召回率
        precision: 准确率

    Returns:
        AP
    """
    # first append sentinel values at the end
    mrec = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
    mpre = torch.cat((torch.tensor([0.0]), precisions, torch.tensor([0.0])))

    # compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = torch.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
