import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.datasets import VOCDataset
from src.trainer import train
from src.tester import test
from src.model.retinanet import RetinaNet


# 训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        命令行参数
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backbone", type=str, default="ResNet50", help="Backbone")
    parser.add_argument("-t", "--test", action='store_true', default=False, help="Test only")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("-mp", "--model_path", type=str, default="./model", help="Pretrained model path")
    # parser.add_argument("-d", "--data_path", type=str, default="./data/VOC2012", help="Data path")  # 完整数据集
    parser.add_argument("-d", "--data_path", type=str, default="./data/VOC2012-sample", help="Data path")  # 抽样得到的小型数据集
    parser.add_argument("-is", "--input_size", type=int, default=608, help="Size of model input")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="Batch size of training")
    parser.add_argument("-ebs", "--eval_batch_size", type=int, default=32, help="Batch size of evaluation")
    parser.add_argument("-ts1", "--stage1_total_steps", type=int, default=4, help="Total training steps")
    parser.add_argument("-ts2", "--stage2_total_steps", type=int, default=4, help="Total training steps")
    parser.add_argument("-es", "--eval_steps", type=int, default=2, help="Evaluation steps")
    parser.add_argument("-p", "--patience", type=int, default=8, help="Patience")
    parser.add_argument("-lr1", "--stage1_learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-lr2", "--stage2_learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("-op", "--output_path", type=str, default="./output", help="Output path")

    args = parser.parse_args()
    return args


def main(args):
    # 设置日志格式
    os.makedirs(args.output_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_path, "main.log"),
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(args)

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 加载数据集
    logging.info("Loading Data")
    train_dataset = VOCDataset(args.data_path, "train", input_size=args.input_size)
    val_dataset = VOCDataset(args.data_path, "val", input_size=args.input_size)
    test_dataset = VOCDataset(args.data_path, "test", input_size=args.input_size)
    logging.info(
        f"\tTraining dataset size: {len(train_dataset)}\n"
        f"\tValidation dataset size: {len(val_dataset)}\n"
        f"\tTest dataset size: {len(test_dataset)}\n"
        f"\tNumber of classes: {train_dataset.num_classes}"
    )

    # 定义模型
    model = RetinaNet(train_dataset.num_classes, DEVICE, args.backbone, model_path=args.model_path)

    # 训练模型
    if not args.test:
        # Stage1: 冻结 ResNet
        logging.info("Stage 1: Freeze ResNet")
        output_path = os.path.join(args.output_path, "Stage1")
        model.freeze_resnet()
        train(
            model,
            DEVICE,
            train_dataset,
            val_dataset,
            args.batch_size,
            args.eval_batch_size,
            args.stage1_total_steps,
            args.eval_steps,
            args.patience,
            args.stage1_learning_rate,
            optimizer="AdamW",
            lr_scheduler="LinearLR",
            output_path=output_path,
        )
        # 加载最好的模型
        model.load_state_dict(
            torch.load(os.path.join(output_path, "best_model.pt"), map_location=DEVICE)
        )
        logging.info("Best model loaded!")
        # Stage2: 微调整个网络
        logging.info("Stage 2: Train the entire network")
        output_path = os.path.join(args.output_path, "Stage2")
        model.unfreeze_resnet()
        train(
            model,
            DEVICE,
            train_dataset,
            val_dataset,
            args.batch_size,
            args.eval_batch_size,
            args.stage2_total_steps,
            args.eval_steps,
            args.patience,
            args.stage2_learning_rate,
            optimizer="AdamW",
            lr_scheduler="LinearLR",
            output_path=output_path,
        )

    # 加载最好的模型
    output_path = os.path.join(args.output_path, "Stage2")
    model.load_state_dict(
        torch.load(os.path.join(output_path, "best_model.pt"), map_location=DEVICE)
    )
    logging.info("Best model loaded!")

    # 测试模型
    test(model, DEVICE, test_dataset, args.eval_batch_size, args.output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
