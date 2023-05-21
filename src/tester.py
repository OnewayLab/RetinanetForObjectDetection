import os
import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
from .metrics import calculate_ap
from .visualization import visualize_result
from .datasets import collator


def test(
    model: torch.nn.Module,
    device: torch.device,
    test_data: Dataset,
    batch_size: int,
    output_path: str,
):
    """测试模型

    Args:
        model: 模型
        device: 设备
        test_data: 测试集
        batch_size: 批大小
        output_path: 输出路径
    """
    os.makedirs(output_path, exist_ok=True)
    logger = logging.getLogger("test")
    handler = logging.FileHandler(os.path.join(output_path, "test.log"), "w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    writer = tensorboard.SummaryWriter(
        os.path.join(output_path, "tensorboard"), filename_suffix="_test"
    )
    logger.info(f"Test set size: {len(test_data)}")

    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collator, shuffle=False, num_workers=8, pin_memory=False)

    # 测试
    logger.info("Start Testing")
    model.eval()
    test_predictions = []
    test_annotations = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            inputs = batch["inputs"].to(device)
            labels = batch["annotations"].to(device)
            h_scales = batch["h_scales"].to(device)
            w_scales = batch["w_scales"].to(device)
            images = batch["images"]
            predictions = model.infer(inputs, h_scales, w_scales, threshold=0.05)
            labels[:, :, [0, 2]] *= w_scales.view(-1, 1, 1)
            labels[:, :, [1, 3]] *= h_scales.view(-1, 1, 1)
            # 保存预测结果用于计算评价指标
            test_predictions += predictions
            test_annotations += [label for label in labels]
            # 保存每个批量第一个样本的预测结果
            image = images[0]
            prediction = predictions[0].cpu().numpy()
            label = labels[0].cpu().numpy()
            visual_prediction = visualize_result(image, prediction, test_data.id2label)
            visual_label = visualize_result(image, label, test_data.id2label)
            writer.add_image(f"test/{i}/prediction", visual_prediction, 0, dataformats="HWC")
            writer.add_image(f"test/{i}/ground_truth", visual_label, 0, dataformats="HWC")
    test_ap = calculate_ap(test_predictions, test_annotations, test_data.num_classes)
    logger.info(f"Test set AP:")
    for i in range(test_data.num_classes):
        logger.info(f"\t{test_data.id2label[i]}: {test_ap[i]}")
    logger.info(f"\tmAP: {test_ap.mean()}")
    writer.close()
