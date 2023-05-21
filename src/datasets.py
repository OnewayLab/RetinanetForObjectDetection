import os
from typing import Dict, List, Tuple, Union
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import xml.dom.minidom as xmldom
import albumentations as A


class VOCDataset(Dataset):
    """VOC2012 数据集"""

    def __init__(self, root, split="train", input_size=608):
        """初始化数据集

        Args:
            root: 数据集路径
            split: 训练集 "train"，验证集 "val"，测试集 "test"
            transform: 图像的预处理
        """
        self.root = root
        self.split = split
        self.MEAN = np.array([116.54703538, 111.75323747, 103.57417823]) / 255
        self.STD = np.array([60.96688134, 59.94961114, 61.13129536]) / 255

        # 读取数据列表
        if split == "train":
            sample_list_path = os.path.join(self.root, "train.txt")
        elif split == "val":
            sample_list_path = os.path.join(self.root, "val.txt")
        elif split == "test":
            sample_list_path = os.path.join(self.root, "test.txt")
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(sample_list_path) as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        # 读取标签名
        with open(os.path.join(root, "labels.txt")) as f:
            self.id2label = [line.strip() for line in f.readlines()]
        self.label2id = {label: i for i, label in enumerate(self.id2label)}
        self.num_classes = len(self.id2label)

        # 数据增强与预处理
        if split == "train":
            self.transform = A.Compose([
            A.RandomResizedCrop(
                width=input_size,
                height=input_size,
                scale=(0.5, 2.0),
                ratio=(0.75, 1.3333333333333333)
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=self.MEAN, std=self.STD),
            A.Resize(width=input_size, height=input_size),
        ], bbox_params=A.BboxParams(format='pascal_voc'))
        else:
            self.transform = A.Compose([
            A.Normalize(mean=self.MEAN, std=self.STD),
            A.Resize(width=input_size, height=input_size),
        ], bbox_params=A.BboxParams(format='pascal_voc'))

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, float]]:
        """获取一个数据样本

        Args:
            idx: 数据索引

        Returns:
            训练集返回变换后的图像和目标框；
            验证集返回变换后的图像、目标框和缩放比；
            测试集返回变换后的图像、目标框、缩放比和原始图像；
            变换后图像 shape=(3, H, W)，目标框 shape=(N, 5)，原始图像 shape=(H, W, 3)
        """
        sample_name = self.sample_list[idx]
        image = cv2.imread(os.path.join(self.root, "JPEGImages", f"{sample_name}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image  # HWC
        annotations = xmldom.parse(
            os.path.join(self.root, "Annotations", f"{sample_name}.xml")
        ).documentElement.getElementsByTagName("object")
        bboxes = []
        for annotation in annotations:  # 一张图片可能有多个目标
            name = annotation.getElementsByTagName("name")[0].firstChild.data
            label = self.label2id[name]
            bbox = annotation.getElementsByTagName("bndbox")[0]
            xmin = int(float(bbox.getElementsByTagName("xmin")[0].firstChild.data))
            ymin = int(float(bbox.getElementsByTagName("ymin")[0].firstChild.data))
            xmax = int(float(bbox.getElementsByTagName("xmax")[0].firstChild.data))
            ymax = int(float(bbox.getElementsByTagName("ymax")[0].firstChild.data))
            bboxes.append((xmin, ymin, xmax, ymax, label))
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        bboxes = np.array(bboxes)
        data = {"input": image, "annotation": bboxes}
        if self.split == "val" or self.split == "test":
            data["h_scale"] = original_image.shape[0] / image.shape[1]
            data["w_scale"] = original_image.shape[1] / image.shape[2]
        if self.split == "test":
            data["image"] = original_image
        return data


def collator(
    data: List[Dict[str, Union[np.ndarray, float]]]
) -> Dict[str, torch.Tensor]:
    """将数据样本转换为一个批量

    Args:
        data: 数据样本的列表

    Returns:

    """
    # 输入图像
    inputs = [torch.from_numpy(d["input"]) for d in data]
    inputs = torch.stack(inputs, dim=0)

    # 目标框
    annotations = [d["annotation"] for d in data]
    max_bboxes = max(annot.shape[0] for annot in annotations)  # 这个 batch 中一张图最多有多少个目标框
    if max_bboxes > 0:
        annotation_batch = torch.full((len(annotations), max_bboxes, 5), -1.0)
        for idx, annot in enumerate(annotations):
            if annot.shape[0] > 0:
                annotation_batch[idx, : annot.shape[0], :] = torch.from_numpy(annot)
    else:
        annotation_batch = torch.full((len(annotations), 1, 5), -1.0)

    batch = {"inputs": inputs, "annotations": annotation_batch}

    # 缩放比
    if "h_scale" in data[0]:
        h_scales = [d["h_scale"] for d in data]
        w_scales = [d["w_scale"] for d in data]
        batch["h_scales"] = torch.tensor(h_scales, dtype=torch.float32)
        batch["w_scales"] = torch.tensor(w_scales, dtype=torch.float32)

    # 原图
    if "image" in data[0]:
        batch["images"] = [d["image"] for d in data if "image" in d]

    return batch
