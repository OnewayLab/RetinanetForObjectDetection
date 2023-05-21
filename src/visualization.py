import numpy as np
from typing import List
import cv2

# 用来把分割结果映射成彩色的掩码
COLOR = ((120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
         (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
         (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
         (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
         (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
         (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
         (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
         (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
         (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
         (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7))


def visualize_result(
    image: np.ndarray, bboxes: np.ndarray, id2label: List[str], threshold: float = 0.5
) -> np.ndarray:
    """可视化结果，在原图上叠加目标框和类别标签

    Args:
        image: 原图，shape=(H, W, 3)
        bboxes: 目标框，shape=(N, 5) 或 (N, 6)，每行为 (xmin, ymin, xmax, ymax, label, [score])
        id2label: 标签名列表
        threshold: 分数阈值，只有当 score > threshold 时才会被可视化

    Returns:
        可视化后的结果，shape=(H, W, 3)
    """
    image = image.copy()
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, label = bbox[:5].astype(int)
        if bbox.shape[0] == 5:
            score = None
        else:
            score = bbox[5]
        if score is not None and score < threshold:  # 忽略低分目标框
            continue
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR[label], 2)
        cv2.putText(
            image,
            f"{id2label[label]}={score:.4f}" if score is not None else f"{id2label[label]}",
            (xmin, ymin - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR[label],
            1,
        )
    return image
